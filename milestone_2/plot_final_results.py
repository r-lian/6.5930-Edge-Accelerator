"""
Plot final project experiment outputs (exp_1, exp_2, exp_3).

Usage:
  python -m milestone_2.plot_final_results
  python -m milestone_2.plot_final_results --results-dir milestone_2/results
  python -m milestone_2.plot_final_results --exp1 path/to/exp1.json --exp2 ... --exp3 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _latest(results_dir: Path, pattern: str) -> Path:
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")
    return files[-1]


def _short(label: str) -> str:
    return label.replace("high_end_embedded", "high").replace("deep_embedded", "deep")


def plot_exp1(exp1_path: Path, out_dir: Path) -> None:
    d = json.loads(exp1_path.read_text())
    cfgs = d["configs"]
    cfgs_sorted = sorted(cfgs, key=lambda c: c["total_edp"])
    top = cfgs_sorted[:15]

    labels = [_short(c["label"]) for c in top]
    edp_ms = np.array([c["total_edp"] for c in top]) * 1e3
    area = np.array([c["area_mm2"] for c in top])
    pwr = np.array([c["avg_power_mw"] for c in top])

    # Figure 1: best configs by EDP
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels[::-1], edp_ms[::-1], color="#4C78A8")
    ax.set_xlabel("EDP (x1e-3 J*s)")
    ax.set_title("Exp 1: Top Configurations by EDP (lower is better)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_top_edp_bar.png", dpi=180)
    plt.close(fig)

    # Figure 2: area vs EDP colored by power
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(area, edp_ms, c=pwr, cmap="viridis", s=90, edgecolor="black", linewidth=0.4)
    for c in top:
        ax.annotate(f"{c['num_macs']}M", (c["area_mm2"], c["total_edp"] * 1e3), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Area (mm^2)")
    ax.set_ylabel("EDP (x1e-3 J*s)")
    ax.set_title("Exp 1: Area-EDP Tradeoff (color = avg power)")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Avg Power (mW)")
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_area_edp_power_scatter.png", dpi=180)
    plt.close(fig)

    # Figure 3: per-layer EDP of top-5
    top5 = cfgs_sorted[:5]
    layer_ids = sorted({l["layer_idx"] for c in top5 for l in c["layers"]})
    x = np.arange(len(layer_ids))
    w = 0.16

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(top5):
        layer_map = {l["layer_idx"]: l["edp"] * 1e3 for l in c["layers"]}
        y = [layer_map.get(lid, 0.0) for lid in layer_ids]
        ax.bar(x + (i - 2) * w, y, width=w, label=_short(c["label"]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{lid}" for lid in layer_ids])
    ax.set_ylabel("Layer EDP (x1e-3 J*s)")
    ax.set_title("Exp 1: Per-layer EDP for Top-5 Configs")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_per_layer_top5.png", dpi=180)
    plt.close(fig)


def plot_exp2(exp2_path: Path, out_dir: Path) -> None:
    d = json.loads(exp2_path.read_text())["by_hw"]
    res = [640, 448, 320]
    res_str = [str(r) for r in res]

    # Figure 1: raw EDP/MAC vs resolution
    fig, ax = plt.subplots(figsize=(9, 6))
    for hw_name, hw in d.items():
        y = [hw["by_resolution"][r]["edp_per_mac"] for r in res_str]
        ax.plot(res, y, marker="o", linewidth=2, label=hw_name)
    ax.invert_xaxis()
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("EDP per MAC")
    ax.set_title("Exp 2: EDP/MAC vs Resolution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_edp_per_mac_vs_resolution.png", dpi=180)
    plt.close(fig)

    # Figure 2: normalized vs 640
    fig, ax = plt.subplots(figsize=(9, 6))
    for hw_name, hw in d.items():
        base = hw["by_resolution"]["640"]["edp_per_mac"]
        y = [hw["by_resolution"][r]["edp_per_mac"] / base for r in res_str]
        ax.plot(res, y, marker="o", linewidth=2, label=hw_name)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.invert_xaxis()
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Normalized EDP/MAC (vs 640)")
    ax.set_title("Exp 2: Normalized EDP/MAC Improvement")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_normalized_edp_per_mac.png", dpi=180)
    plt.close(fig)

    # Figure 3: per-layer latency speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    for hw_name, hw in d.items():
        l640 = {l["layer_idx"]: l for l in hw["by_resolution"]["640"]["layers"]}
        l320 = {l["layer_idx"]: l for l in hw["by_resolution"]["320"]["layers"]}
        ids = sorted(set(l640) & set(l320))
        speedup = [l640[i]["latency_s"] / l320[i]["latency_s"] for i in ids]
        ax.plot([f"T{i}" for i in ids], speedup, marker="o", linewidth=2, label=hw_name)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_ylabel("Latency speedup (640/320)")
    ax.set_title("Exp 2: Per-layer Latency Speedup from 320px")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_per_layer_latency_speedup.png", dpi=180)
    plt.close(fig)


def plot_exp3(exp3_path: Path, out_dir: Path) -> None:
    d = json.loads(exp3_path.read_text())
    areas = d["area_tiers_mm2"]
    powers = d["power_tiers_mw"]
    grid = d["grid"]

    M = np.full((len(areas), len(powers)), np.nan)
    labels = [["" for _ in powers] for _ in areas]
    for i, a in enumerate(areas):
        for j, p in enumerate(powers):
            cell = grid.get(f"{a:.2f}_{p}")
            if cell is None:
                continue
            b = cell["best"]
            M[i, j] = b["total_edp"] * 1e3
            labels[i][j] = f"{b['num_macs']}M\n{b['sram_kb']}KB"

    # Figure 1: heatmap
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(len(powers)))
    ax.set_xticklabels([str(p) for p in powers])
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels([str(a) for a in areas])
    ax.set_xlabel("Power cap (mW)")
    ax.set_ylabel("Area cap (mm^2)")
    ax.set_title("Exp 3: Best EDP per (Area, Power) Cell (x1e-3 J*s)")
    for i in range(len(areas)):
        for j in range(len(powers)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}\n{labels[i][j]}", ha="center", va="center", fontsize=8, color="white")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("EDP (x1e-3 J*s)")
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_budget_grid_heatmap.png", dpi=180)
    plt.close(fig)

    # Figure 2: constraint utilization
    keys = []
    au = []
    pu = []
    for k, cell in grid.items():
        if cell is None:
            continue
        keys.append(k)
        au.append(cell["best"]["area_utilization"])
        pu.append(cell["best"]["power_utilization"])

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, au, width=w, label="Area utilization")
    ax.bar(x + w / 2, pu, width=w, label="Power utilization")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("Utilization (fraction of cap)")
    ax.set_title("Exp 3: Constraint Utilization by Budget Cell")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_constraint_utilization.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("milestone_2/results"))
    parser.add_argument("--exp1", type=Path, default=None)
    parser.add_argument("--exp2", type=Path, default=None)
    parser.add_argument("--exp3", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    exp1 = args.exp1 or _latest(results_dir, "exp_1_*.json")
    exp2 = args.exp2 or _latest(results_dir, "exp_2_*.json")
    exp3 = args.exp3 or _latest(results_dir, "exp_3_*.json")
    out_dir = args.out_dir or (results_dir / "final_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot] exp1: {exp1.name}")
    print(f"[plot] exp2: {exp2.name}")
    print(f"[plot] exp3: {exp3.name}")
    print(f"[plot] out:  {out_dir}")

    plot_exp1(exp1, out_dir)
    plot_exp2(exp2, out_dir)
    plot_exp3(exp3, out_dir)

    print("[plot] done. generated files:")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
