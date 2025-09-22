#!/usr/bin/env python3
import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


JOB_RE = re.compile(r"^(?P<name>[^.]+)\.(?P<jobid>\d+)\.rank0000\.out$")


def parse_bench_poisson(text: str):
    """Return dict with keys: dofs, nnz, gpu_us, cpu_us (optional)."""
    res = {}
    # DoFs
    m = re.search(r"DoFs:\s+(?P<dofs>\d+)", text)
    if m:
        res["dofs"] = int(m.group("dofs"))
    # global nnz
    m = re.search(r"global nnz:\s+(?P<nnz>\d+)", text)
    if m:
        res["nnz"] = int(m.group("nnz"))
    # GPU per-iter time (custom GPU)
    mg = re.search(r"Custom GPU SpMV benchmark.*?per-iter:\s+(?P<us>[0-9.]+)\s+us", text, re.S)
    if mg:
        res["gpu_us"] = float(mg.group("us"))
    # CPU per-iter time (optional)
    mc = re.search(r"Hypre CPU SpMV benchmark.*?per-iter:\s+(?P<us>[0-9.]+)\s+us", text, re.S)
    if mc:
        res["cpu_us"] = float(mc.group("us"))
    return res


def parse_bench_hypre_gpu(text: str):
    """Return dict with keys: dofs, nnz, gpu_us, gawa (bool)."""
    res = {}
    m = re.search(r"DoFs:\s+(?P<dofs>\d+)", text)
    if m:
        res["dofs"] = int(m.group("dofs"))
    m = re.search(r"global nnz:\s+(?P<nnz>\d+)", text)
    if m:
        res["nnz"] = int(m.group("nnz"))
    mg = re.search(r"Hypre GPU SpMV benchmark.*?per-iter:\s+(?P<us>[0-9.]+)\s+us", text, re.S)
    if mg:
        res["gpu_us"] = float(mg.group("us"))
    # Detect compile-time GPU-aware MPI status from log
    res["gawa"] = bool(re.search(r"HYPRE_USING_GPU_AWARE_MPI:\s+yes", text))
    return res


def collect_runs(logs_dir: str):
    data = {
        ("weak", "bench_poisson"): {},
        ("strong", "bench_poisson"): {},
        ("weak", "bench_hypre_gpu_nogawa"): {},
        ("strong", "bench_hypre_gpu_nogawa"): {},
        ("weak", "bench_hypre_gpu_gawa"): {},
        ("strong", "bench_hypre_gpu_gawa"): {},
    }
    # Search recursively to include archived runs like logs/no_gawa/
    candidates = sorted(glob.glob(os.path.join(logs_dir, "**", "*.rank0000.out"), recursive=True))
    for path in candidates:
        fname = os.path.basename(path)
        m = JOB_RE.match(fname)
        # Fall back: take the portion before the first dot as job name
        name = m.group("name") if m else fname.split(".")[0]
        kind = None
        target = None
        gpus = None
        m2 = re.match(r"^(ws|ss)_(bench_poisson|bench_hypre_gpu)_G(\d+)", name)
        if not m2:
            continue
        kind = "weak" if m2.group(1) == "ws" else "strong"
        target = m2.group(2)
        gpus = int(m2.group(3))

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if target == "bench_poisson":
            res = parse_bench_poisson(text)
        else:
            res = parse_bench_hypre_gpu(text)
        status = "ok" if (res and ("gpu_us" in res)) else "no-parse"
        tag = None
        if target == "bench_hypre_gpu" and res and ("gawa" in res):
            tag = "gawa" if res["gawa"] else "nogawa"
        print(f"{fname}: kind={kind} target={target}{('/'+tag) if tag else ''} gpus={gpus} status={status}")
        if not res or "gpu_us" not in res:
            continue
        if target == "bench_hypre_gpu":
            series = "bench_hypre_gpu_gawa" if res.get("gawa") else "bench_hypre_gpu_nogawa"
            data[(kind, series)][gpus] = res
        else:
            data[(kind, target)][gpus] = res
    return data if any(data[k] for k in data) else ({}, candidates)


def plot_scaling(data, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    kinds = ["weak", "strong"]
    titles = {"weak": "Weak Scaling", "strong": "Strong Scaling"}
    for ax, kind in zip(axes, kinds):
        runs_custom = data.get((kind, "bench_poisson"), {})
        runs_hypre_noga  = data.get((kind, "bench_hypre_gpu_nogawa"), {})
        runs_hypre_gawa  = data.get((kind, "bench_hypre_gpu_gawa"), {})

        plotted_any = False
        if runs_custom:
            xs_c = sorted(runs_custom.keys())
            ys_c = [runs_custom[g]["gpu_us"] for g in xs_c]
            ax.plot(xs_c, ys_c, marker="o", label="Custom GPU (bench_poisson)")
            plotted_any = True
        if runs_hypre_noga:
            xs_h0 = sorted(runs_hypre_noga.keys())
            ys_h0 = [runs_hypre_noga[g]["gpu_us"] for g in xs_h0]
            ax.plot(xs_h0, ys_h0, marker="s", label="Hypre GPU (GPU-aware OFF)")
            plotted_any = True
        if runs_hypre_gawa:
            xs_h1 = sorted(runs_hypre_gawa.keys())
            ys_h1 = [runs_hypre_gawa[g]["gpu_us"] for g in xs_h1]
            ax.plot(xs_h1, ys_h1, marker="^", label="Hypre GPU (GPU-aware ON)")
            plotted_any = True

        # Weak-scaling: annotate x-ticks with global DoFs (2 sig figs, scientific)
        if kind == "weak":
            all_xs = sorted(set(runs_custom.keys()) | set(runs_hypre_noga.keys()) | set(runs_hypre_gawa.keys()))
            if all_xs:
                # Prefer DoFs from custom; fall back to hypre if missing
                labels = []
                for g in all_xs:
                    dofs = None
                    if g in runs_custom and "dofs" in runs_custom[g]:
                        dofs = runs_custom[g]["dofs"]
                    elif g in runs_hypre_gawa and "dofs" in runs_hypre_gawa[g]:
                        dofs = runs_hypre_gawa[g]["dofs"]
                    elif g in runs_hypre_noga and "dofs" in runs_hypre_noga[g]:
                        dofs = runs_hypre_noga[g]["dofs"]
                    if dofs is not None:
                        dofs_s = f"{dofs:.2e} DoFs"
                        labels.append(f"{g} GPUs\n{dofs_s}")
                    else:
                        labels.append(str(g))
                ax.set_xticks(all_xs)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.tick_params(axis="x", labelsize=8)

        # Build plot title, optionally appending DoFs for strong scaling
        plot_title = titles[kind]
        if kind == "strong":
            dofs_val = None
            # Prefer DoFs from custom; fall back to hypre
            for runs in (runs_custom, runs_hypre_gawa, runs_hypre_noga):
                if runs:
                    for g, res in runs.items():
                        if "dofs" in res:
                            dofs_val = res["dofs"]
                            break
                if dofs_val is not None:
                    break
            if dofs_val is not None:
                plot_title += f" (DoFs={dofs_val:.2e})"
            ax.legend()
        ax.set_title(plot_title + ("" if plotted_any else " (no data)"))
        ax.set_xlabel("GPUs")
        ax.set_ylabel("Avg SpMV time (us)")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Scaling Study", fontsize=14)
    fig.savefig(out_path, dpi=150)
    root, _ = os.path.splitext(out_path)
    fig.savefig(root + ".pdf")
    print(f"Saved plots to {out_path} and {root}.pdf")


def main():
    ap = argparse.ArgumentParser(description="Parse scaling logs and plot results.")
    ap.add_argument("--logs-dir", default="logs", help="Directory with rank0000 log files")
    ap.add_argument("--out", default="tools/scaling_study.png", help="Output plot path (PNG)")
    args = ap.parse_args()

    result = collect_runs(args.logs_dir)
    if isinstance(result, tuple):
        data, candidates = result
    else:
        data = result
        candidates = []
    if not any(data[k] for k in data):
        print("No data found in", args.logs_dir)
        if candidates:
            print("Checked files:")
            for p in candidates[:10]:
                print("  ", os.path.basename(p))
        return
    plot_scaling(data, args.out)


if __name__ == "__main__":
    main()
