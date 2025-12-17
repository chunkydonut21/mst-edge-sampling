# sanity_edge.py
from __future__ import annotations

import os
import json
import csv
import time
from argparse import ArgumentParser

import tracemalloc
from sklearn.metrics import adjusted_rand_score

from EdgeSamplingMST_PySpark import compare_on_one_dataset, mst_to_k_clusters


def save_mst_csv(path: str, mst_edges):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "w"])
        for (u, v, ww) in mst_edges:
            w.writerow([int(u), int(v), float(ww)])


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1 / 8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="Results/edge_sampling_strict")
    parser.add_argument("--test", action="store_true", help="Only run first 2 datasets")
    parser.add_argument("--save-plots", action="store_true", help="Save MST plot PNGs")
    args = parser.parse_args()

    from mstfordensegraphs import get_clustering_data  # repo helper
    if args.save_plots:
        from Plotter import Plotter  # repo plotter

    os.makedirs(args.outdir, exist_ok=True)

    datasets = get_clustering_data()
    names = ["TwoCircles", "TwoMoons", "Varied", "Aniso", "Blobs", "Random"]

    plotter = None
    if args.save_plots:
        plotter = Plotter(None, None, args.outdir.rstrip("/") + "/")

    for i, ds in enumerate(datasets):
        if args.test and i >= 2:
            break

        name = names[i]
        X = ds[0][0]  # (n,d) points
        y_true = ds[0][1] if len(ds[0]) > 1 else None  # labels if present
        n = int(X.shape[0])

        t0 = time.perf_counter()
        tracemalloc.start()
        mst, stats, ari = compare_on_one_dataset(
            X,
            y_true=y_true,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        space_usage_mb = {
            "current_mb": current_mem / (1024 * 1024),
            "peak_mb": peak_mem / (1024 * 1024),
        }
        t1 = time.perf_counter()

        # Save MST
        mst_csv = os.path.join(args.outdir, f"mst_{name}_n{n}_eps{args.epsilon}.csv")
        save_mst_csv(mst_csv, mst)

        # Save clustering assignment (for reporting / debugging)
        clustering_path = os.path.join(args.outdir, f"clustering_{name}_n{n}_eps{args.epsilon}.json")
        clustering = None
        if y_true is not None:
            k = len(set(y_true))
            clustering = mst_to_k_clusters(n, mst, k)
            save_json(clustering_path, {"dataset": name, "k": k, "labels": clustering})
        else:
            save_json(clustering_path, {"dataset": name, "k": None, "labels": None})

        # Save stats
        stats_json = os.path.join(args.outdir, f"stats_{name}_n{n}_eps{args.epsilon}.json")
        save_json(
            stats_json,
            {
                "dataset": name,
                "n": n,
                "epsilon": args.epsilon,
                "rounds": stats.rounds,
                "y_threshold": stats.y_threshold,
                "per_round_m": stats.per_round_m,
                "per_round_x": stats.per_round_x,
                "per_round_max_bucket": stats.per_round_max_bucket,
                "time_sec_algorithm": stats.total_time_sec,  # inside algorithm
                "time_sec_wall": (t1 - t0),                 # end-to-end wrapper time
                "mst_edges": len(mst),
                "ari": ari,
                "space_usage_mb": space_usage_mb,
            },
        )

        # Save plot (PNG)
        if plotter is not None:
            plotter.set_dataset(name)
            plotter.set_vertex_coordinates(X.tolist())
            plotter.update_string()
            plotter.reset_round()
            plotter.plot_mst_2d(list(mst), intermediate=False, plot_cluster=False)

            if y_true is not None:
                k = len(set(y_true))
                plotter.plot_mst_2d(list(mst), intermediate=False, plot_cluster=True, num_clusters=k)

        print(
            f"[{name}] MST edges={len(mst)} expected={n-1} "
            f"rounds={stats.rounds} time={stats.total_time_sec:.2f}s ARI={ari}"
        )


if __name__ == "__main__":
    main()
