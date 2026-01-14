# run_report2.py
from __future__ import annotations

import os
import json
import time
import tracemalloc
from argparse import ArgumentParser

try:
    from pyspark import SparkConf, SparkContext  # type: ignore
    _HAS_PYSPARK = True
except Exception:
    SparkConf = None  # type: ignore
    SparkContext = None  # type: ignore
    _HAS_PYSPARK = False

from max_coverage import (
    build_ball_sets_bitmask,
    greedy_max_coverage,
    threshold_max_coverage_lecture7,
    brute_force_optimum,
)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def measure_peak_memory_mb(fn):
    """
    Run fn() under tracemalloc and return (result, current_mb, peak_mb).
    NOTE: tracemalloc measures Python allocations (not Spark JVM heap).
    """
    tracemalloc.start()
    out = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, current / (1024 * 1024), peak / (1024 * 1024)


def main():
    parser = ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.25, help="epsilon for threshold algorithm (Lecture 7)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="Results/report2_coverage")
    parser.add_argument("--test", action="store_true", help="Only run first 2 datasets")
    parser.add_argument("--dataset-only", type=str, default="", help="Run only one dataset name (e.g., TwoMoons)")

    # coverage instance knobs
    parser.add_argument("--m", type=int, default=200, help="number of candidate sets/balls")
    parser.add_argument("--k", type=int, default=20, help="pick k sets")
    parser.add_argument("--radius", type=float, default=0.7, help="ball radius in data space")

    # Spark master similar to your old style
    parser.add_argument("--master", type=str, default="local[*]", help="Spark master, e.g. local[*], local[4]")

    # optional optimum (only do on tiny m,k)
    parser.add_argument("--compute-opt", action="store_true", help="Bruteforce OPT (only feasible for small m,k)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Create ONE SparkContext for the whole run (reused across datasets + both algos)
    sc = None
    if _HAS_PYSPARK:
        conf = SparkConf().setAppName("Report2_MaxCoverage").setMaster(args.master)
        sc = SparkContext.getOrCreate(conf=conf)

    # reuse your repo’s dataset generator from report 1
    from mstfordensegraphs import get_clustering_data  # repo helper

    datasets = get_clustering_data()
    names = ["TwoCircles", "TwoMoons", "Varied", "Aniso", "Blobs", "Random"]

    try:
        for i, ds in enumerate(datasets):
            if args.test and i >= 2:
                break

            name = names[i]
            if args.dataset_only and name != args.dataset_only:
                continue

            X = ds[0][0]  # (n,d)
            n = int(X.shape[0])

            sets = build_ball_sets_bitmask(
                X,
                m=args.m,
                radius=args.radius,
                seed=args.seed,
                centers_from_points=True,
            )

            opt = None
            if args.compute_opt:
                if args.m <= 25 and args.k <= 6:
                    opt = brute_force_optimum(sets, args.k)
                else:
                    print(f"[{name}] Skipping OPT: too large (m={args.m}, k={args.k}). Use m<=25,k<=6.")

            # -------------------------
            # Run Greedy
            # -------------------------
            t0 = time.perf_counter()

            def _run_greedy():
                return greedy_max_coverage(
                    sets,
                    args.k,
                    sc=sc,
                    master=args.master,
                    app_name=f"R2_Coverage_Greedy_{name}",
                    stop_spark=False,
                )

            (chosen_g, covered_g, f_g, rounds_g, algo_time_g), cur_mb_g, peak_mb_g = measure_peak_memory_mb(_run_greedy)
            t1 = time.perf_counter()

            stats_g = {
                "problem": "maximum_coverage",
                "algo": "greedy",
                "dataset": name,
                "n_points": n,
                "m_sets": args.m,
                "k": args.k,
                "radius": args.radius,
                "epsilon": args.epsilon,
                "rounds": rounds_g,
                "f_value": f_g,
                "chosen_count": len(chosen_g),
                "time_sec_algorithm": algo_time_g,
                "time_sec_wall": (t1 - t0),
                "space_usage_mb": {"current_mb": cur_mb_g, "peak_mb": peak_mb_g},
                "opt_value": opt,
                "ratio_to_opt": (f_g / opt) if opt else None,
            }

            save_json(
                os.path.join(args.outdir, f"stats_{name}_n{n}_m{args.m}_k{args.k}_greedy.json"),
                stats_g,
            )

            # -------------------------
            # Run Threshold (Lecture 7)
            # -------------------------
            t2 = time.perf_counter()

            def _run_threshold():
                return threshold_max_coverage_lecture7(
                    sets,
                    args.k,
                    args.epsilon,
                    sc=sc,
                    master=args.master,
                    app_name=f"R2_Coverage_Threshold_{name}",
                    stop_spark=False,
                )

            (chosen_t, covered_t, f_t, rounds_t, algo_time_t, num_guesses), cur_mb_t, peak_mb_t = measure_peak_memory_mb(_run_threshold)
            t3 = time.perf_counter()

            stats_t = {
                "problem": "maximum_coverage",
                "algo": "threshold_lecture7",
                "dataset": name,
                "n_points": n,
                "m_sets": args.m,
                "k": args.k,
                "radius": args.radius,
                "epsilon": args.epsilon,
                "num_guesses": num_guesses,
                "rounds": rounds_t,
                "f_value": f_t,
                "chosen_count": len(chosen_t),
                "time_sec_algorithm": algo_time_t,
                "time_sec_wall": (t3 - t2),
                "space_usage_mb": {"current_mb": cur_mb_t, "peak_mb": peak_mb_t},
                "opt_value": opt,
                "ratio_to_opt": (f_t / opt) if opt else None,
            }

            save_json(
                os.path.join(args.outdir, f"stats_{name}_n{n}_m{args.m}_k{args.k}_threshold.json"),
                stats_t,
            )

            print(
                f"[{name}] Greedy f={f_g} rounds={rounds_g} time={algo_time_g:.2f}s | "
                f"Threshold f={f_t} guesses={num_guesses} rounds={rounds_t} time={algo_time_t:.2f}s"
            )

    finally:
        # Stop Spark exactly once
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
