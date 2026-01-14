from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

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
    # new points added by set_mask beyond covered
    return popcount(set_mask & ~covered)


# -----------------------------
# Instance generation (points -> balls -> bitmasks)
# -----------------------------
def build_ball_sets_bitmask(
    X,
    m: int,
    radius: float,
    *,
    seed: int = 0,
    centers_from_points: bool = True,
) -> List[Mask]:
    """
    Build m candidate sets ("balls") that cover points within radius.
    Returns List[int] where each int is a bitmask of covered point indices.
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

    r2 = radius * radius
    sets: List[int] = []

    for c in centers:
        d2 = ((X - c) ** 2).sum(axis=1)
        idx = np.where(d2 <= r2)[0]
        mask = 0
        for p in idx.tolist():
            mask |= (1 << p)
        sets.append(mask)

    return sets


# -----------------------------
# Stats (optional, not required by runner)
# -----------------------------
@dataclass
class CoverageStats:
    algo: str
    dataset: str
    n_points: int
    m_sets: int
    k: int
    epsilon: float
    radius: float
    num_guesses: int
    rounds: int
    total_time_sec: float
    f_value: int
    chosen_count: int
    opt_value: Optional[int] = None
    ratio_to_opt: Optional[float] = None


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
) -> Tuple[List[int], Mask, int, int, float]:
    """
    Returns (chosen_indices, covered_mask, f(S), rounds, time_sec)
    rounds ~= number of global passes (about chosen_count)
    """
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

    return chosen, covered, f_value(covered), rounds, (t1 - t0)


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
) -> Tuple[List[int], Mask, int, int, float, int]:
    """
    Implements Lecture 7 structure:

    1) f_e = max singleton value
    2) y = ceil( log(k) / log(1+eps) )
    3) For each guess j=0..y:
         tau_j = ((1+eps)^j * f_e) / (2k)
         build S^j up to k picks, only adding elements with marginal gain >= tau_j
       return best over j

    Returns (chosen_best, covered_best, f_best, rounds, time_sec, num_guesses)
    where num_guesses = y+1
    """
    t0 = time.perf_counter()
    rounds = 0

    # best singleton value f(e)
    f_e = max((popcount(sm) for sm in sets), default=0)

    # y = ceil(log k / log(1+eps))  (Lecture 7)
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

    return best_chosen, best_cov, best_val, rounds, (t1 - t0), num_guesses


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
