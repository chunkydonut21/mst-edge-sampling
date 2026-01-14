# run_report2.py (IMPROVED)
from __future__ import annotations

import json
import os
import time
import tracemalloc
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Tuple

try:
    from pyspark import SparkConf, SparkContext  # type: ignore
    _HAS_PYSPARK = True
except Exception:
    SparkConf = None  # type: ignore
    SparkContext = None  # type: ignore
    _HAS_PYSPARK = False

from max_coverage import (
    AlgoRunDiagnostics,
    approx_py_object_size_mb,
    build_ball_sets_bitmask,
    brute_force_optimum,
    greedy_max_coverage,
    sample_radii,
    threshold_max_coverage_lecture7,
)


def save_json(path: str, obj: Any):
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


def spark_jobs_stages_for_group(sc: SparkContext, group_id: str) -> Tuple[int, int]:
    """
    Practical proxy for 'communication rounds' in Spark:
    count jobs + stages triggered by the algorithm call.
    """
    tracker = sc.statusTracker()
    job_ids = tracker.getJobIdsForGroup(group_id) or []
    n_jobs = len(job_ids)
    stages = set()
    for jid in job_ids:
        info = tracker.getJobInfo(jid)
        if info is not None:
            for sid in info.stageIds:
                stages.add(int(sid))
    return n_jobs, len(stages)


def run_with_job_group(sc: Optional[SparkContext], group_id: str, desc: str, fn):
    """
    Wrap a function call in a Spark job group so we can later query job/stage counts.
    """
    if sc is not None:
        sc.setJobGroup(group_id, desc)
    try:
        return fn()
    finally:
        if sc is not None:
            sc.clearJobGroup()


def main():
    parser = ArgumentParser()

    # algorithm knob
    parser.add_argument("--epsilon", type=float, default=0.25, help="epsilon for threshold algorithm (Lecture 7)")

    # dataset selection
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="Results/report2_coverage")
    parser.add_argument("--test", action="store_true", help="Only run first 2 datasets")
    parser.add_argument("--dataset-only", type=str, default="", help="Run only one dataset name (e.g., TwoMoons)")

    # coverage instance knobs
    parser.add_argument("--m", type=int, default=200, help="number of candidate sets/balls")
    parser.add_argument("--k", type=int, default=20, help="pick k sets")
    parser.add_argument("--radius", type=float, default=0.7, help="default radius (fixed radii mode)")
    parser.add_argument("--radii-mode", type=str, default="fixed", choices=["fixed", "uniform"], help="fixed or varying radii")
    parser.add_argument("--radius-min", type=float, default=0.4, help="min radius for uniform mode")
    parser.add_argument("--radius-max", type=float, default=1.0, help="max radius for uniform mode")

    # Spark master
    parser.add_argument("--master", type=str, default="local[*]", help="Spark master, e.g. local[*], local[4]")

    # OPT controls:
    # - always compute OPT when feasible, and report ratio_to_opt
    parser.add_argument(
        "--compute-opt",
        action="store_true",
        help="Compute brute force OPT when feasible (m<=25,k<=6).",
    )

    # add a built-in "small OPT benchmark" regime
    parser.add_argument(
        "--also-run-opt-benchmark",
        action="store_true",
        help="In addition to your main (m,k), also run a small (m_opt,k_opt) regime to compare vs OPT.",
    )
    parser.add_argument("--m-opt", type=int, default=25, help="m for OPT-benchmark regime (must be <=25)")
    parser.add_argument("--k-opt", type=int, default=6, help="k for OPT-benchmark regime (must be <=6)")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Create ONE SparkContext for the whole run
    sc = None
    if _HAS_PYSPARK:
        conf = SparkConf().setAppName("Report2_MaxCoverage").setMaster(args.master)
        sc = SparkContext.getOrCreate(conf=conf)

    # datasets
    from mstfordensegraphs import get_clustering_data  # repo helper

    datasets = get_clustering_data()
    names = ["TwoCircles", "TwoMoons", "Varied", "Aniso", "Blobs", "Random"]

    def build_instance(X, m: int) -> Tuple[list, Dict[str, Any]]:
        radii = sample_radii(
            m,
            mode=args.radii_mode,
            r=args.radius,
            r_min=args.radius_min,
            r_max=args.radius_max,
            seed=args.seed,
        )
        sets = build_ball_sets_bitmask(
            X,
            m=m,
            radius=radii,  # varying or fixed depending on radii-mode
            seed=args.seed,
            centers_from_points=True,
        )
        meta = {
            "m_sets": m,
            "radii_mode": args.radii_mode,
            "radius_fixed": args.radius,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
        }
        return sets, meta

    def maybe_opt(sets, k: int) -> Optional[int]:
        if not args.compute_opt:
            return None
        if len(sets) <= 25 and k <= 6:
            return brute_force_optimum(sets, k)
        return None

    try:
        for i, ds in enumerate(datasets):
            if args.test and i >= 2:
                break

            name = names[i]
            if args.dataset_only and name != args.dataset_only:
                continue

            X = ds[0][0]  # (n,d)
            n = int(X.shape[0])

            # -------------------------
            # Main regime (your chosen m,k)
            # -------------------------
            sets, inst_meta = build_instance(X, args.m)
            py_sets_size_mb = approx_py_object_size_mb(sets)

            opt = maybe_opt(sets, args.k)

            # Greedy
            group_g = f"R2_Greedy_{name}_main"
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

            (chosen_g, covered_g, diag_g), cur_mb_g, peak_mb_g = measure_peak_memory_mb(
                lambda: run_with_job_group(sc, group_g, "Greedy main", _run_greedy)
            )
            t1 = time.perf_counter()

            jobs_g, stages_g = (None, None)
            if sc is not None:
                jobs_g, stages_g = spark_jobs_stages_for_group(sc, group_g)
                diag_g.spark_jobs = jobs_g
                diag_g.spark_stages = stages_g

            stats_g = {
                "problem": "maximum_coverage_points",
                "algo": "greedy",
                "regime": "main",
                "dataset": name,
                "n_points": n,
                "k": args.k,
                **inst_meta,
                "rounds_logical": diag_g.rounds_logical,
                "spark_jobs": diag_g.spark_jobs,
                "spark_stages": diag_g.spark_stages,
                "f_value": diag_g.f_value,
                "chosen_count": diag_g.chosen_count,
                "time_sec_algorithm": diag_g.time_sec,
                "time_sec_wall": (t1 - t0),
                "python_space_usage_mb": {
                    "sets_list_estimate_mb": py_sets_size_mb,
                    "tracemalloc_current_mb": cur_mb_g,
                    "tracemalloc_peak_mb": peak_mb_g,
                },
                "opt_value": opt,
                "ratio_to_opt": (diag_g.f_value / opt) if opt else None,
            }
            save_json(os.path.join(args.outdir, f"stats_{name}_main_greedy.json"), stats_g)

            # Threshold
            group_t = f"R2_Threshold_{name}_main"
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

            (chosen_t, covered_t, diag_t, num_guesses), cur_mb_t, peak_mb_t = measure_peak_memory_mb(
                lambda: run_with_job_group(sc, group_t, "Threshold main", _run_threshold)
            )
            t3 = time.perf_counter()

            jobs_t, stages_t = (None, None)
            if sc is not None:
                jobs_t, stages_t = spark_jobs_stages_for_group(sc, group_t)
                diag_t.spark_jobs = jobs_t
                diag_t.spark_stages = stages_t

            stats_t = {
                "problem": "maximum_coverage_points",
                "algo": "threshold_lecture7",
                "regime": "main",
                "dataset": name,
                "n_points": n,
                "k": args.k,
                "epsilon": args.epsilon,
                "num_guesses": num_guesses,
                **inst_meta,
                "rounds_logical": diag_t.rounds_logical,
                "spark_jobs": diag_t.spark_jobs,
                "spark_stages": diag_t.spark_stages,
                "f_value": diag_t.f_value,
                "chosen_count": diag_t.chosen_count,
                "time_sec_algorithm": diag_t.time_sec,
                "time_sec_wall": (t3 - t2),
                "python_space_usage_mb": {
                    "sets_list_estimate_mb": py_sets_size_mb,
                    "tracemalloc_current_mb": cur_mb_t,
                    "tracemalloc_peak_mb": peak_mb_t,
                },
                "opt_value": opt,
                "ratio_to_opt": (diag_t.f_value / opt) if opt else None,
            }
            save_json(os.path.join(args.outdir, f"stats_{name}_main_threshold.json"), stats_t)

            print(
                f"[{name}][main] Greedy f={diag_g.f_value} rounds={diag_g.rounds_logical} "
                f"jobs={diag_g.spark_jobs} stages={diag_g.spark_stages} time={diag_g.time_sec:.2f}s | "
                f"Threshold f={diag_t.f_value} guesses={num_guesses} rounds={diag_t.rounds_logical} "
                f"jobs={diag_t.spark_jobs} stages={diag_t.spark_stages} time={diag_t.time_sec:.2f}s"
            )

            # -------------------------
            # Optional OPT benchmark regime (small m,k) for ratio-to-opt plots
            # -------------------------
            if args.also_run_opt_benchmark:
                if args.m_opt > 25 or args.k_opt > 6:
                    raise ValueError("--m-opt must be <=25 and --k-opt must be <=6 for brute force OPT.")

                sets_opt, inst_meta_opt
                sets_opt, inst_meta_opt = build_instance(X, args.m_opt)
                py_sets_opt_size_mb = approx_py_object_size_mb(sets_opt)

                opt2 = brute_force_optimum(sets_opt, args.k_opt)

                # Greedy small
                group_g2 = f"R2_Greedy_{name}_opt"
                (chosen_g2, covered_g2, diag_g2), cur_mb_g2, peak_mb_g2 = measure_peak_memory_mb(
                    lambda: run_with_job_group(
                        sc,
                        group_g2,
                        "Greedy OPT-benchmark",
                        lambda: greedy_max_coverage(
                            sets_opt,
                            args.k_opt,
                            sc=sc,
                            master=args.master,
                            app_name=f"R2_Coverage_Greedy_{name}_OPT",
                            stop_spark=False,
                        ),
                    )
                )
                if sc is not None:
                    jg2, sg2 = spark_jobs_stages_for_group(sc, group_g2)
                    diag_g2.spark_jobs, diag_g2.spark_stages = jg2, sg2

                stats_g2 = {
                    "problem": "maximum_coverage_points",
                    "algo": "greedy",
                    "regime": "opt_benchmark",
                    "dataset": name,
                    "n_points": n,
                    "k": args.k_opt,
                    **inst_meta_opt,
                    "rounds_logical": diag_g2.rounds_logical,
                    "spark_jobs": diag_g2.spark_jobs,
                    "spark_stages": diag_g2.spark_stages,
                    "f_value": diag_g2.f_value,
                    "chosen_count": diag_g2.chosen_count,
                    "python_space_usage_mb": {
                        "sets_list_estimate_mb": py_sets_opt_size_mb,
                        "tracemalloc_current_mb": cur_mb_g2,
                        "tracemalloc_peak_mb": peak_mb_g2,
                    },
                    "opt_value": opt2,
                    "ratio_to_opt": diag_g2.f_value / opt2,
                }
                save_json(os.path.join(args.outdir, f"stats_{name}_optbench_greedy.json"), stats_g2)

                # Threshold small
                group_t2 = f"R2_Threshold_{name}_opt"
                (chosen_t2, covered_t2, diag_t2, num_guesses2), cur_mb_t2, peak_mb_t2 = measure_peak_memory_mb(
                    lambda: run_with_job_group(
                        sc,
                        group_t2,
                        "Threshold OPT-benchmark",
                        lambda: threshold_max_coverage_lecture7(
                            sets_opt,
                            args.k_opt,
                            args.epsilon,
                            sc=sc,
                            master=args.master,
                            app_name=f"R2_Coverage_Threshold_{name}_OPT",
                            stop_spark=False,
                        ),
                    )
                )
                if sc is not None:
                    jt2, st2 = spark_jobs_stages_for_group(sc, group_t2)
                    diag_t2.spark_jobs, diag_t2.spark_stages = jt2, st2

                stats_t2 = {
                    "problem": "maximum_coverage_points",
                    "algo": "threshold_lecture7",
                    "regime": "opt_benchmark",
                    "dataset": name,
                    "n_points": n,
                    "k": args.k_opt,
                    "epsilon": args.epsilon,
                    "num_guesses": num_guesses2,
                    **inst_meta_opt,
                    "rounds_logical": diag_t2.rounds_logical,
                    "spark_jobs": diag_t2.spark_jobs,
                    "spark_stages": diag_t2.spark_stages,
                    "f_value": diag_t2.f_value,
                    "chosen_count": diag_t2.chosen_count,
                    "python_space_usage_mb": {
                        "sets_list_estimate_mb": py_sets_opt_size_mb,
                        "tracemalloc_current_mb": cur_mb_t2,
                        "tracemalloc_peak_mb": peak_mb_t2,
                    },
                    "opt_value": opt2,
                    "ratio_to_opt": diag_t2.f_value / opt2,
                }
                save_json(os.path.join(args.outdir, f"stats_{name}_optbench_threshold.json"), stats_t2)

                print(
                    f"[{name}][opt_benchmark] OPT={opt2} | "
                    f"Greedy={diag_g2.f_value} ({diag_g2.f_value/opt2:.3f}) | "
                    f"Threshold={diag_t2.f_value} ({diag_t2.f_value/opt2:.3f})"
                )

    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
