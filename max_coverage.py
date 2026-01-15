# max_coverage.py (WITH VIS + OPT SOLUTION SUPPORT)
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple, Union

try:
    from pyspark import SparkConf, SparkContext  # type: ignore
    _HAS_PYSPARK = True
except Exception:  # pragma: no cover
    SparkConf = None  # type: ignore
    SparkContext = None  # type: ignore
    _HAS_PYSPARK = False

Mask = int  # Python int as bitset


# -----------------------------
# Helpers
# -----------------------------
def popcount(x: int) -> int:
    return x.bit_count()


def f_value(covered: Mask) -> int:
    return popcount(covered)


def marginal_gain(set_mask: Mask, covered: Mask) -> int:
    return popcount(set_mask & ~covered)


def approx_py_object_size_mb(obj) -> float:
    import sys

    seen = set()
    stack = [obj]
    total = 0
    while stack:
        x = stack.pop()
        oid = id(x)
        if oid in seen:
            continue
        seen.add(oid)
        total += sys.getsizeof(x)
        if isinstance(x, dict):
            stack.extend(list(x.keys()))
            stack.extend(list(x.values()))
        elif isinstance(x, (list, tuple, set, frozenset)):
            stack.extend(list(x))
    return total / (1024 * 1024)


# -----------------------------
# Instance generation (points -> balls -> bitmasks)
# -----------------------------
def build_ball_sets_bitmask(
    X,
    m: int,
    radius: Union[float, Sequence[float]],
    *,
    seed: int = 0,
    centers_from_points: bool = True,
) -> List[Mask]:
    # Backward-compatible wrapper: only masks.
    masks, _, _ = build_ball_sets_bitmask_with_meta(
        X, m=m, radius=radius, seed=seed, centers_from_points=centers_from_points
    )
    return masks


def build_ball_sets_bitmask_with_meta(
    X,
    m: int,
    radius: Union[float, Sequence[float]],
    *,
    seed: int = 0,
    centers_from_points: bool = True,
):
    """
    Build m candidate sets ("balls") covering points within radius.

    Returns:
      - masks: List[int] bitmask per ball (covered points)
      - centers: (m,d) ndarray of centers
      - radii: (m,) ndarray radii
    """
    import numpy as np

    n = X.shape[0]
    rng = np.random.default_rng(seed)

    if centers_from_points:
        center_idx = rng.choice(n, size=m, replace=(m > n))
        centers = X[center_idx]
    else:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        centers = rng.uniform(mins, maxs, size=(m, X.shape[1]))

    if isinstance(radius, (list, tuple, np.ndarray)):
        if len(radius) != m:
            raise ValueError("If radius is a sequence, it must have length m.")
        radii = np.asarray(radius, dtype=float)
    else:
        radii = np.full(m, float(radius), dtype=float)

    masks: List[int] = []
    for c, r in zip(centers, radii):
        r2 = float(r) * float(r)
        d2 = ((X - c) ** 2).sum(axis=1)
        idx = np.where(d2 <= r2)[0]
        mask = 0
        for p in idx.tolist():
            mask |= (1 << p)
        masks.append(mask)

    return masks, centers, radii


def sample_radii(m: int, *, mode: str, r: float, r_min: float, r_max: float, seed: int) -> List[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    if mode == "fixed":
        return [float(r)] * m
    if mode == "uniform":
        return rng.uniform(r_min, r_max, size=m).astype(float).tolist()
    raise ValueError("mode must be one of: fixed, uniform")


# -----------------------------
# Stats
# -----------------------------
@dataclass
class AlgoRunDiagnostics:
    rounds_logical: int
    time_sec: float
    f_value: int
    chosen_count: int
    spark_jobs: Optional[int] = None
    spark_stages: Optional[int] = None


# -----------------------------
# Algorithm 1: Greedy
# -----------------------------
def greedy_max_coverage(
    sets: Sequence[Mask],
    k: int,
    *,
    sc: Optional["SparkContext"] = None,
    master: str = "local[*]",
    app_name: str = "MaxCoverage_Greedy",
    stop_spark: bool = True,
) -> Tuple[List[int], Mask, AlgoRunDiagnostics]:
    t0 = time.perf_counter()
    rounds = 0

    created_sc = False
    rdd = None

    if _HAS_PYSPARK:
        if sc is None:
            conf = SparkConf().setAppName(app_name).setMaster(master)
            sc = SparkContext.getOrCreate(conf=conf)
            created_sc = True
        rdd = sc.parallelize(list(enumerate(sets))).cache()

    covered: Mask = 0
    chosen: List[int] = []
    chosen_set = set()

    for _ in range(k):
        rounds += 1

        if _HAS_PYSPARK and sc is not None and rdd is not None:
            bc_cov = sc.broadcast(covered)
            bc_chosen = sc.broadcast(chosen_set)

            def score(pair):
                idx, sm = pair
                if idx in bc_chosen.value:
                    return (-1, idx)
                g = marginal_gain(sm, bc_cov.value)
                return (g, idx)

            best_gain, best_idx = rdd.map(score).max()
            bc_cov.destroy()
            bc_chosen.destroy()

            if best_gain <= 0:
                break

            chosen.append(best_idx)
            chosen_set.add(best_idx)
            covered |= sets[best_idx]
        else:
            best_gain, best_idx = -1, -1
            for idx, sm in enumerate(sets):
                if idx in chosen_set:
                    continue
                g = marginal_gain(sm, covered)
                if g > best_gain:
                    best_gain, best_idx = g, idx
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            covered |= sets[best_idx]

    t1 = time.perf_counter()

    if rdd is not None:
        rdd.unpersist()

    if _HAS_PYSPARK and stop_spark and created_sc and sc is not None:
        sc.stop()

    diag = AlgoRunDiagnostics(
        rounds_logical=rounds,
        time_sec=(t1 - t0),
        f_value=f_value(covered),
        chosen_count=len(chosen),
    )
    return chosen, covered, diag


# -----------------------------
# Algorithm 2: Lecture-7 Threshold + guesses
# -----------------------------
def threshold_max_coverage_lecture7(
    sets: Sequence[Mask],
    k: int,
    epsilon: float,
    *,
    sc: Optional["SparkContext"] = None,
    master: str = "local[*]",
    app_name: str = "MaxCoverage_Threshold",
    stop_spark: bool = True,
) -> Tuple[List[int], Mask, AlgoRunDiagnostics, int]:
    t0 = time.perf_counter()
    rounds = 0

    f_e = max((popcount(sm) for sm in sets), default=0)

    if k <= 1:
        y = 0
    else:
        y = int(math.ceil(math.log(k) / math.log(1.0 + epsilon)))
    num_guesses = y + 1

    created_sc = False
    rdd = None

    if _HAS_PYSPARK:
        if sc is None:
            conf = SparkConf().setAppName(app_name).setMaster(master)
            sc = SparkContext.getOrCreate(conf=conf)
            created_sc = True
        rdd = sc.parallelize(list(enumerate(sets))).cache()

    best_cov: Mask = 0
    best_chosen: List[int] = []
    best_val = 0

    for j in range(y + 1):
        covered: Mask = 0
        chosen_set = set()
        chosen: List[int] = []

        tau = ((1.0 + epsilon) ** j) * f_e / (2.0 * max(1, k))

        for _ in range(k):
            rounds += 1

            if _HAS_PYSPARK and sc is not None and rdd is not None:
                bc_cov = sc.broadcast(covered)
                bc_chosen = sc.broadcast(chosen_set)

                def score(pair):
                    idx, sm = pair
                    if idx in bc_chosen.value:
                        return (-1, idx)
                    g = marginal_gain(sm, bc_cov.value)
                    if g >= tau:
                        return (g, idx)
                    return (-1, idx)

                best_gain, best_idx = rdd.map(score).max()
                bc_cov.destroy()
                bc_chosen.destroy()

                if best_gain < 0:
                    break
                chosen.append(best_idx)
                chosen_set.add(best_idx)
                covered |= sets[best_idx]
            else:
                best_gain, best_idx = -1, -1
                for idx, sm in enumerate(sets):
                    if idx in chosen_set:
                        continue
                    g = marginal_gain(sm, covered)
                    if g >= tau and g > best_gain:
                        best_gain, best_idx = g, idx
                if best_gain < 0:
                    break
                chosen.append(best_idx)
                chosen_set.add(best_idx)
                covered |= sets[best_idx]

        val = f_value(covered)
        if val > best_val:
            best_val = val
            best_cov = covered
            best_chosen = chosen

    t1 = time.perf_counter()

    if rdd is not None:
        rdd.unpersist()

    if _HAS_PYSPARK and stop_spark and created_sc and sc is not None:
        sc.stop()

    diag = AlgoRunDiagnostics(
        rounds_logical=rounds,
        time_sec=(t1 - t0),
        f_value=best_val,
        chosen_count=len(best_chosen),
    )
    return best_chosen, best_cov, diag, num_guesses


# -----------------------------
# Brute force OPT (small m,k only)
# -----------------------------
def brute_force_optimum(sets: Sequence[Mask], k: int) -> int:
    best = 0
    for comb in combinations(range(len(sets)), k):
        cov = 0
        for idx in comb:
            cov |= sets[idx]
        best = max(best, popcount(cov))
    return best


def brute_force_optimum_solution(sets: Sequence[Mask], k: int) -> Tuple[int, List[int], Mask]:
    """
    Returns (opt_value, opt_indices, opt_covered_mask).
    Only feasible for small m,k.
    """
    best = -1
    best_comb: Tuple[int, ...] = tuple()
    best_cov: Mask = 0
    for comb in combinations(range(len(sets)), k):
        cov = 0
        for idx in comb:
            cov |= sets[idx]
        val = popcount(cov)
        if val > best:
            best = val
            best_comb = comb
            best_cov = cov
    return best, list(best_comb), best_cov
