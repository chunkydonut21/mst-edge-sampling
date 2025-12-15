

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

try:
    from pyspark import SparkConf, SparkContext  # type: ignore
    _HAS_PYSPARK = True
except Exception:  # pragma: no cover
    SparkConf = None  # type: ignore
    SparkContext = None  # type: ignore
    _HAS_PYSPARK = False

# Optional plotting via the repo's Plotter.py (only used if save_plot=True)
try:
    from Plotter import Plotter  # type: ignore
    _HAS_PLOTTER = True
except Exception:  # pragma: no cover
    Plotter = None  # type: ignore
    _HAS_PLOTTER = False


# Edge representation: (u, v, w) with u < v for uniqueness.
Edge = Tuple[int, int, float]


class DSU:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_msf(n: int, edges: Sequence[Edge]) -> List[Edge]:
    """Minimum spanning forest of (possibly disconnected) graph on vertex ids 0..n-1."""
    dsu = DSU(n)
    out: List[Edge] = []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(u, v):
            out.append((u, v, float(w)))
            if len(out) == n - 1:
                break
    return out


def edges_dict_to_list(E: Dict[int, Dict[int, float]]) -> List[Edge]:
    """Convert a half-adjacency dict into a unique undirected edge list."""
    edges: List[Edge] = []
    for u, nbrs in E.items():
        for v, w in nbrs.items():
            uu, vv = (u, v) if u < v else (v, u)
            edges.append((uu, vv, float(w)))
    return edges


@dataclass
class EdgeSamplingStats:
    rounds: int
    y_threshold: int
    per_round_m: List[int]
    per_round_x: List[int]
    per_round_max_bucket: List[int]
    total_time_sec: float


def _make_universal_hash(n: int, x: int, seed: int):
    """2-universal hash h(e) in {0..x-1} via linear hashing mod a big prime."""
    p = 2147483647  # prime
    rng = random.Random(seed)
    a = rng.randint(1, p - 1)
    b = rng.randint(0, p - 1)

    def h(edge: Edge) -> int:
        u, v, _ = edge
        if u > v:
            u, v = v, u
        key = u * n + v
        return ((a * key + b) % p) % x

    return h


def _ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _save_mst_csv(mst: Sequence[Edge], outdir: str, tag: str) -> str:
    path = os.path.join(outdir, f"mst_{tag}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("u,v,w\n")
        for u, v, w in mst:
            f.write(f"{u},{v},{w}\n")
    return path


def _save_stats_json(stats: EdgeSamplingStats, outdir: str, tag: str) -> str:
    path = os.path.join(outdir, f"stats_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)
    return path


def edge_sampling_mst(
    n: int,
    edges: List[Edge],
    epsilon: float,
    *,
    master: str = "local[*]",
    app_name: str = "MST_EdgeSampling",
    seed: int = 0,
    stop_spark: bool = True,
    outdir: Optional[str] = None,
    tag: str = "edge",
    save_mst: bool = True,
    save_stats: bool = True,
) -> Tuple[List[Edge], EdgeSamplingStats]:
    """Edge-sampling MPC MST algorithm (Algorithm MST-Dense-1). Returns (mst_edges, stats).

    If outdir is provided, saves stats + mst edge list (CSV) unless disabled.
    """
    if n <= 1:
        stats = EdgeSamplingStats(0, 0, [], [], [], 0.0)
        if outdir:
            outdir = _ensure_dir(outdir)
            if save_stats:
                _save_stats_json(stats, outdir, tag)
        return [], stats

    outdir = _ensure_dir(outdir)
    y = int(math.ceil(n ** (1.0 + epsilon)))

    rounds = 0
    per_round_m: List[int] = []
    per_round_x: List[int] = []
    per_round_max_bucket: List[int] = []

    sc = None
    if _HAS_PYSPARK:
        conf = SparkConf().setAppName(app_name).setMaster(master)
        sc = SparkContext.getOrCreate(conf=conf)

    t0 = time.perf_counter()
    cur_edges = edges

    while len(cur_edges) > y:
        rounds += 1
        m = len(cur_edges)
        x = int(math.ceil(m / y))
        per_round_m.append(m)
        per_round_x.append(x)

        h = _make_universal_hash(n, x, seed + rounds)

        if _HAS_PYSPARK:
            assert sc is not None
            pair_rdd = (
                sc.parallelize(cur_edges)
                .map(lambda e: (h(e), e))
                .partitionBy(x)
            )
            pair_rdd.persist()

            bucket_sizes = pair_rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
            per_round_max_bucket.append(max(bucket_sizes) if bucket_sizes else 0)

            def mst_in_partition(it: Iterable[Tuple[int, Edge]]):
                es = [e for _, e in it]
                for edge in kruskal_msf(n, es):
                    yield edge

            union_edges = pair_rdd.mapPartitions(mst_in_partition).collect()
            pair_rdd.unpersist()
        else:
            buckets: List[List[Edge]] = [[] for _ in range(x)]
            for e in cur_edges:
                buckets[h(e)].append(e)
            per_round_max_bucket.append(max((len(b) for b in buckets), default=0))
            union_edges: List[Edge] = []
            for b in buckets:
                union_edges.extend(kruskal_msf(n, b))

        cur_edges = union_edges

    final_mst = kruskal_msf(n, cur_edges)
    t1 = time.perf_counter()

    if _HAS_PYSPARK and stop_spark and sc is not None:
        sc.stop()

    stats = EdgeSamplingStats(
        rounds=rounds,
        y_threshold=y,
        per_round_m=per_round_m,
        per_round_x=per_round_x,
        per_round_max_bucket=per_round_max_bucket,
        total_time_sec=(t1 - t0),
    )

    if outdir:
        if save_stats:
            _save_stats_json(stats, outdir, tag)
        if save_mst:
            _save_mst_csv(final_mst, outdir, tag)

    return final_mst, stats


def mst_to_k_clusters(n: int, mst_edges: Sequence[Edge], k: int) -> List[int]:
    """MST clustering by removing the (k-1) largest edges."""
    if k <= 1:
        return [0] * n
    kept = sorted(mst_edges, key=lambda e: e[2])[: max(0, len(mst_edges) - (k - 1))]
    dsu = DSU(n)
    for u, v, _ in kept:
        dsu.union(u, v)
    comp = [dsu.find(i) for i in range(n)]
    mapping: Dict[int, int] = {}
    out: List[int] = []
    nxt = 0
    for c in comp:
        if c not in mapping:
            mapping[c] = nxt
            nxt += 1
        out.append(mapping[c])
    return out


def compare_on_one_dataset(
    X,  # (n, d)
    y_true: Optional[Sequence[int]],
    *,
    epsilon: float = 1 / 8,
    edge_keep_fraction: float = 1.0,
    seed: int = 0,
    outdir: Optional[str] = None,
    tag: str = "edge",
    save_plot: bool = False,
):
    """Build dense graph from points, run edge-sampling MST, optionally save to outdir."""
    import scipy.spatial

    outdir = _ensure_dir(outdir)
    n = X.shape[0]

    dm = scipy.spatial.distance_matrix(X, X, threshold=1000000)
    edges: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, float(dm[i, j])))

    if edge_keep_fraction < 1.0:
        rng = random.Random(seed)
        rng.shuffle(edges)
        edges = edges[: int(len(edges) * edge_keep_fraction)]

    mst, stats = edge_sampling_mst(
        n,
        edges,
        epsilon,
        seed=seed,
        outdir=outdir,
        tag=tag,
        save_mst=True,
        save_stats=True,
    )

    clustering_score = None
    if y_true is not None:
        try:
            from sklearn.metrics import adjusted_rand_score
            k = len(set(y_true))
            y_pred = mst_to_k_clusters(n, mst, k)
            clustering_score = adjusted_rand_score(y_true, y_pred)
            if outdir:
                with open(os.path.join(outdir, f"clustering_{tag}.json"), "w", encoding="utf-8") as f:
                    json.dump({"ari": clustering_score, "k": k}, f, indent=2)
        except Exception:
            clustering_score = None

    # Optional plot (only if you explicitly request it)
    if save_plot and outdir and _HAS_PLOTTER:
        # Plotter expects a folder ending with "/"
        pdir = outdir if outdir.endswith(os.sep) else outdir + os.sep
        plotter = Plotter(pdir)
        # Plotter in repo typically expects (X, mst_edges, name)
        # We'll mimic "final" plot naming by using tag as dataset name.
        try:
            plotter.plot_final_mst(X, mst, tag)  # if your Plotter has this
        except Exception:
            # fallback: many versions use plot_mst(...)
            try:
                plotter.plot_mst(X, mst, f"{tag}_final")
            except Exception:
                pass

    return mst, stats, clustering_score
