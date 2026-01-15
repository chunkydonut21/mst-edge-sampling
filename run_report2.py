# run_report2.py (WITH VISUALIZATION OUTPUT)
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
    brute_force_optimum,
    brute_force_optimum_solution,
    greedy_max_coverage,
    sample_radii,
    threshold_max_coverage_lecture7,
    build_ball_sets_bitmask_with_meta,
)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def measure_peak_memory_mb(fn):
    tracemalloc.start()
    out = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, current / (1024 * 1024), peak / (1024 * 1024)


def spark_jobs_stages_for_group(sc: SparkContext, group_id: str) -> Tuple[int, int]:
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
    if sc is not None:
        sc.setJobGroup(group_id, desc)
    try:
        return fn()
    finally:
        if sc is not None:
            sc.setJobGroup(None, None)


def mask_to_bool_list(mask: int, n: int):
    return [(mask >> i) & 1 == 1 for i in range(n)]


def plot_solution(
    X,
    centers,
    radii,
    chosen_indices,
    covered_mask: int,
    *,
    title: str,
    outpath: str,
    show_all_candidates: bool = False,
):
    """
    Saves a PNG with:
      - points colored by covered/uncovered
      - chosen circles drawn
      - optionally all candidate circles faint
    Assumes X is 2D (which is true for TwoMoons/TwoCircles/etc).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n = X.shape[0]
    covered = mask_to_bool_list(covered_mask, n)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(title)

    # points: covered vs uncovered
    x0 = X[:, 0]
    x1 = X[:, 1]
    cov_x = [x0[i] for i in range(n) if covered[i]]
    cov_y = [x1[i] for i in range(n) if covered[i]]
    unc_x = [x0[i] for i in range(n) if not covered[i]]
    unc_y = [x1[i] for i in range(n) if not covered[i]]

    if unc_x:
        ax.scatter(unc_x, unc_y, s=10, alpha=0.9, label="uncovered")
    if cov_x:
        ax.scatter(cov_x, cov_y, s=10, alpha=0.9, label="covered")

    # candidate circles (optional)
    if show_all_candidates:
        for c, r in zip(centers, radii):
            ax.add_patch(Circle((c[0], c[1]), r, fill=False, alpha=0.08, linewidth=0.8))

    # chosen circles (highlight)
    for idx in chosen_indices:
        c = centers[idx]
        r = radii[idx]
        ax.add_patch(Circle((c[0], c[1]), r, fill=False, linewidth=2.5))

    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    parser = ArgumentParser()

    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="Results/report2_coverage")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset-only", type=str, default="")

    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--k", type=int, default=20)

    parser.add_argument("--radius", type=float, default=0.7)
    parser.add_argument("--radii-mode", type=str, default="fixed", choices=["fixed", "uniform"])
    parser.add_argument("--radius-min", type=float, default=0.4)
    parser.add_argument("--radius-max", type=float, default=1.0)

    parser.add_argument("--master", type=str, default="local[*]")

    parser.add_argument("--compute-opt", action="store_true")
    parser.add_argument("--also-run-opt-benchmark", action="store_true")
    parser.add_argument("--m-opt", type=int, default=25)
    parser.add_argument("--k-opt", type=int, default=6)

    # NEW: visualization
    parser.add_argument("--plot", action="store_true", help="Write PNG visualizations to outdir")
    parser.add_argument("--plot-all-candidates", action="store_true", help="Also draw all candidate circles faintly")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sc = None
    if _HAS_PYSPARK:
        conf = SparkConf().setAppName("Report2_MaxCoverage").setMaster(args.master)
        sc = SparkContext.getOrCreate(conf=conf)

    from mstfordensegraphs import get_clustering_data
    datasets = get_clustering_data()
    names = ["TwoCircles", "TwoMoons", "Varied", "Aniso", "Blobs", "Random"]

    def build_instance_with_meta(X, m: int):
        radii = sample_radii(
            m,
            mode=args.radii_mode,
            r=args.radius,
            r_min=args.radius_min,
            r_max=args.radius_max,
            seed=args.seed,
        )
        masks, centers, radii_arr = build_ball_sets_bitmask_with_meta(
            X, m=m, radius=radii, seed=args.seed, centers_from_points=True
        )
        meta = {
            "m_sets": m,
            "radii_mode": args.radii_mode,
            "radius_fixed": args.radius,
            "radius_min": args.radius_min,
            "radius_max": args.radius_max,
        }
        return masks, centers, radii_arr, meta

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

            X = ds[0][0]
            n = int(X.shape[0])

            # -------------------------
            # MAIN regime
            # -------------------------
            sets, centers, radii_arr, inst_meta = build_instance_with_meta(X, args.m)
            py_sets_size_mb = approx_py_object_size_mb(sets)
            opt = maybe_opt(sets, args.k)

            # Greedy main
            group_g = f"R2_Greedy_{name}_main"

            def _run_greedy():
                return greedy_max_coverage(
                    sets, args.k, sc=sc, master=args.master, app_name=f"R2_Coverage_Greedy_{name}", stop_spark=False
                )

            (chosen_g, covered_g, diag_g), cur_mb_g, peak_mb_g = measure_peak_memory_mb(
                lambda: run_with_job_group(sc, group_g, "Greedy main", _run_greedy)
            )

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
                "python_space_usage_mb": {
                    "sets_list_estimate_mb": py_sets_size_mb,
                    "tracemalloc_current_mb": cur_mb_g,
                    "tracemalloc_peak_mb": peak_mb_g,
                },
                "opt_value": opt,
                "ratio_to_opt": (diag_g.f_value / opt) if opt else None,
                "chosen_indices": chosen_g,
            }
            save_json(os.path.join(args.outdir, f"stats_{name}_main_greedy.json"), stats_g)

            if args.plot:
                out_png = os.path.join(args.outdir, f"viz_{name}_main_greedy.png")
                plot_solution(
                    X,
                    centers,
                    radii_arr,
                    chosen_g,
                    covered_g,
                    title=f"{name} | MAIN | Greedy | covered={diag_g.f_value}/{n} | chosen={len(chosen_g)}",
                    outpath=out_png,
                    show_all_candidates=args.plot_all_candidates,
                )

            # Threshold main
            group_t = f"R2_Threshold_{name}_main"

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
                "python_space_usage_mb": {
                    "sets_list_estimate_mb": py_sets_size_mb,
                    "tracemalloc_current_mb": cur_mb_t,
                    "tracemalloc_peak_mb": peak_mb_t,
                },
                "opt_value": opt,
                "ratio_to_opt": (diag_t.f_value / opt) if opt else None,
                "chosen_indices": chosen_t,
            }
            save_json(os.path.join(args.outdir, f"stats_{name}_main_threshold.json"), stats_t)

            if args.plot:
                out_png = os.path.join(args.outdir, f"viz_{name}_main_threshold.png")
                plot_solution(
                    X,
                    centers,
                    radii_arr,
                    chosen_t,
                    covered_t,
                    title=f"{name} | MAIN | Threshold | covered={diag_t.f_value}/{n} | chosen={len(chosen_t)}",
                    outpath=out_png,
                    show_all_candidates=args.plot_all_candidates,
                )

            # -------------------------
            # OPT BENCH regime
            # -------------------------
            if args.also_run_opt_benchmark:
                if args.m_opt > 25 or args.k_opt > 6:
                    raise ValueError("--m-opt must be <=25 and --k-opt must be <=6 for brute force OPT.")

                sets_opt, centers_opt, radii_opt, inst_meta_opt = build_instance_with_meta(X, args.m_opt)
                py_sets_opt_size_mb = approx_py_object_size_mb(sets_opt)

                # Compute OPT solution indices + mask (for visualization)
                opt_val, opt_idx, opt_cov = brute_force_optimum_solution(sets_opt, args.k_opt)

                # Greedy optbench
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
                    "opt_value": opt_val,
                    "ratio_to_opt": diag_g2.f_value / opt_val,
                    "chosen_indices": chosen_g2,
                }
                save_json(os.path.join(args.outdir, f"stats_{name}_optbench_greedy.json"), stats_g2)

                if args.plot:
                    out_png = os.path.join(args.outdir, f"viz_{name}_optbench_greedy.png")
                    plot_solution(
                        X,
                        centers_opt,
                        radii_opt,
                        chosen_g2,
                        covered_g2,
                        title=f"{name} | OPTBENCH | Greedy | covered={diag_g2.f_value}/{n} | OPT={opt_val}",
                        outpath=out_png,
                        show_all_candidates=args.plot_all_candidates,
                    )

                # Threshold optbench
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
                    "opt_value": opt_val,
                    "ratio_to_opt": diag_t2.f_value / opt_val,
                    "chosen_indices": chosen_t2,
                }
                save_json(os.path.join(args.outdir, f"stats_{name}_optbench_threshold.json"), stats_t2)

                if args.plot:
                    out_png = os.path.join(args.outdir, f"viz_{name}_optbench_threshold.png")
                    plot_solution(
                        X,
                        centers_opt,
                        radii_opt,
                        chosen_t2,
                        covered_t2,
                        title=f"{name} | OPTBENCH | Threshold | covered={diag_t2.f_value}/{n} | OPT={opt_val}",
                        outpath=out_png,
                        show_all_candidates=args.plot_all_candidates,
                    )

                    # Also plot the actual OPT solution
                    out_png_opt = os.path.join(args.outdir, f"viz_{name}_optbench_OPT.png")
                    plot_solution(
                        X,
                        centers_opt,
                        radii_opt,
                        opt_idx,
                        opt_cov,
                        title=f"{name} | OPTBENCH | OPT (bruteforce) | covered={opt_val}/{n}",
                        outpath=out_png_opt,
                        show_all_candidates=args.plot_all_candidates,
                    )

    finally:
        if sc is not None:
            sc.stop()


if __name__ == "__main__":
    main()
 