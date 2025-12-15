"""Edge-sampling MPC MST (Lecture 2) + benchmarking wrapper.

This file is meant to be dropped into the same folder as the provided repo code.

It implements Algorithm "MST-Dense-1" from the Lecture 2 handout (edge sampling):
  - If |E| <= y = n^(1+epsilon): compute MST locally.
  - Else: hash-partition edges into x = ceil(|E|/y) buckets, compute an MST/MSF per bucket,
          recurse on the union of these MST edges.

It also contains a small benchmarking harness to compare against the *provided* vertex-sampling
implementation in PysparkMSTfordensegraphsfast.py.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

try:
    from pyspark import SparkConf, SparkContext  # type: ignore
    _HAS_PYSPARK = True
except Exception:  # pragma: no cover
    # Allows basic correctness testing without Spark.
    SparkConf = None  # type: ignore
    SparkContext = None  # type: ignore
    _HAS_PYSPARK = False


# Edge representation used across the repo: (u, v, w) with u < v for uniqueness.
Edge = Tuple[int, int, float]


def remap_vertices_edges(vertices: Sequence[int], edges: Sequence[Edge]) -> Tuple[int, List[Edge], List[int]]:
    """Remap arbitrary vertex ids to 0..n-1.

    Returns (n, remapped_edges, id_from_new_index).
    """
    uniq = list(dict.fromkeys(vertices))
    id_to_new = {vid: i for i, vid in enumerate(uniq)}
    remapped: List[Edge] = []
    for u, v, w in edges:
        if u not in id_to_new or v not in id_to_new:
            continue
        uu, vv = id_to_new[u], id_to_new[v]
        if uu == vv:
            continue
        if uu > vv:
            uu, vv = vv, uu
        remapped.append((uu, vv, float(w)))
    return len(uniq), remapped, uniq


class DSU:
    """Disjoint Set Union / Union-Find."""

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
            out.append((u, v, w))
            if len(out) == n - 1:
                # early exit for connected graphs
                break
    return out


def edges_dict_to_list(E: Dict[int, Dict[int, float]]) -> List[Edge]:
    """Convert the repo's half-adjacency dict into a unique undirected edge list."""
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


def _make_universal_hash(n: int, x: int, seed: int) -> "callable":
    """2-universal hash h(e) in {0..x-1} via linear hashing mod a big prime."""
    # Prime > n^2 for safety; 2^31-1 is prime.
    p = 2147483647
    rng = random.Random(seed)
    a = rng.randint(1, p - 1)
    b = rng.randint(0, p - 1)

    def h(edge: Edge) -> int:
        u, v, _ = edge
        if u > v:
            u, v = v, u
        key = u * n + v  # in [0, n^2)
        return ((a * key + b) % p) % x

    return h


def edge_sampling_mst(
    n: int,
    edges: List[Edge],
    epsilon: float,
    *,
    master: str = "local[*]",
    app_name: str = "MST_EdgeSampling",
    seed: int = 0,
    stop_spark: bool = True,
) -> Tuple[List[Edge], EdgeSamplingStats]:
    """Edge-sampling MPC MST algorithm (Algorithm MST-Dense-1).

    Returns: (mst_edges, stats)
    """
    if n <= 1:
        return [], EdgeSamplingStats(0, 0, [], [], [], 0.0)

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

    # Main recursion/loop
    while len(cur_edges) > y:
        rounds += 1
        m = len(cur_edges)
        x = int(math.ceil(m / y))
        per_round_m.append(m)
        per_round_x.append(x)

        h = _make_universal_hash(n, x, seed + rounds)

        if _HAS_PYSPARK:
            assert sc is not None
            # Key edges by bucket id, then hash-partition (one bucket per partition).
            pair_rdd = (
                sc.parallelize(cur_edges)
                .map(lambda e: (h(e), e))
                .partitionBy(x)
            )
            pair_rdd.persist()

            # Space proxy: bucket sizes (edges per partition).
            bucket_sizes = pair_rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
            per_round_max_bucket.append(max(bucket_sizes) if bucket_sizes else 0)

            # Compute an MST/MSF per bucket; union them.
            def mst_in_partition(it: Iterable[Tuple[int, Edge]]):
                es = [e for _, e in it]
                for edge in kruskal_msf(n, es):
                    yield edge

            union_edges = pair_rdd.mapPartitions(mst_in_partition).collect()
            pair_rdd.unpersist()
        else:
            # Local fallback: partition edges into buckets using the same hash.
            buckets: List[List[Edge]] = [[] for _ in range(x)]
            for e in cur_edges:
                buckets[h(e)].append(e)
            per_round_max_bucket.append(max((len(b) for b in buckets), default=0))
            union_edges: List[Edge] = []
            for b in buckets:
                union_edges.extend(kruskal_msf(n, b))

        # Next epoch input.
        cur_edges = union_edges

    # Final MST on reduced edge set (single machine step in the lecture algorithm).
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
    return final_mst, stats


def mst_to_k_clusters(n: int, mst_edges: Sequence[Edge], k: int) -> List[int]:
    """MST clustering by removing the (k-1) largest edges."""
    if k <= 1:
        return [0] * n
    # Remove largest (k-1) edges
    kept = sorted(mst_edges, key=lambda e: e[2])[: max(0, len(mst_edges) - (k - 1))]
    dsu = DSU(n)
    for u, v, _ in kept:
        dsu.union(u, v)
    # Compress component ids to 0..k-1-ish
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
):
    """Small helper: build dense graph from points, run edge-sampling MST.

    Vertex-sampling baseline is intentionally not run here, because the repo's baseline
    already provides it (create_mst in PysparkMSTfordensegraphsfast.py).
    """
    import numpy as np
    import scipy.spatial

    n = X.shape[0]
    # Build complete graph edge list (upper triangle)
    dm = scipy.spatial.distance_matrix(X, X, threshold=1000000)
    edges: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, float(dm[i, j])))

    if edge_keep_fraction < 1.0:
        rng = random.Random(seed)
        rng.shuffle(edges)
        edges = edges[: int(len(edges) * edge_keep_fraction)]

    mst, stats = edge_sampling_mst(n, edges, epsilon, seed=seed)

    clustering_score = None
    if y_true is not None:
        try:
            from sklearn.metrics import adjusted_rand_score

            # If you know the true number of clusters (e.g., blobs/moons/circles), pass it.
            k = len(set(y_true))
            y_pred = mst_to_k_clusters(n, mst, k)
            clustering_score = adjusted_rand_score(y_true, y_pred)
        except Exception:
            clustering_score = None

    return mst, stats, clustering_score
