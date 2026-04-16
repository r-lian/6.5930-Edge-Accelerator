"""
plot_results.py — Visualize co-design sweep results
=====================================================

Usage:
    python -m milestone_2.plot_results                    # most recent probe_sweep
    python -m milestone_2.plot_results results/probe_sweep_20260416_172426.json
    python -m milestone_2.plot_results --file results/budget_sensitivity_*.json
    python -m milestone_2.plot_results --list             # list available result files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; override with MPLBACKEND=TkAgg if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

PROBE_LAYER_NAMES = {1: "T1 stem", 7: "T7 L3", 10: "T10 L4", 16: "T16 FFN", 19: "T19 detH"}
BUDGET_LINES = {"tight": 0.20, "baseline": 0.38, "relaxed": 0.80}  # mm²

# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_results(tag: str = "probe_sweep") -> Path:
    files = sorted(_RESULTS_DIR.glob(f"{tag}_*.json"))
    if not files:
        raise FileNotFoundError(f"No {tag}_*.json files in {_RESULTS_DIR}")
    return files[-1]


def _short_label(label: str) -> str:
    """Compact label for axis ticks."""
    return label.strip().replace("KB-scratch", "KB-s").replace("embedded", "")


def _preset_marker(preset: str) -> str:
    return "o" if "high" in preset else "s"


def _pareto_front(xs, ys):
    """Return indices of Pareto-optimal points (minimize both x and y)."""
    pts = sorted(enumerate(zip(xs, ys)), key=lambda t: t[1][0])
    pareto = []
    min_y = float("inf")
    for idx, (x, y) in pts:
        if y < min_y:
            pareto.append(idx)
            min_y = y
    return pareto


# ── plot functions ─────────────────────────────────────────────────────────────

def plot_probe_sweep(data: list[dict], out_path: Path) -> None:
    """4-panel summary of the probe-layer sweep."""
    configs = data
    labels   = [_short_label(c["label"]) for c in configs]
    areas    = np.array([c["area_mm2"]       for c in configs])
    energies = np.array([c["total_energy_j"] * 1e3 for c in configs])  # mJ
    latencies= np.array([c["total_latency_s"] * 1e3 for c in configs])  # ms
    edps     = np.array([c["total_edp"]       for c in configs])
    powers   = np.array([c["avg_power_mw"]    for c in configs])

    # Color by num_macs
    mac_vals = sorted(set(c["num_macs"] for c in configs))
    cmap = plt.cm.tab10
    mac_color = {m: cmap(i) for i, m in enumerate(mac_vals)}
    colors = [mac_color[c["num_macs"]] for c in configs]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Co-design Probe Sweep — Architecture Configuration Comparison", fontsize=14, y=0.98)

    # ── Panel 1: EDP bar chart (primary figure-of-merit) ──────────────────────
    ax = axes[0, 0]
    order = np.argsort(edps)
    bars = ax.barh([labels[i] for i in order], edps[order] * 1e3,
                   color=[colors[i] for i in order], edgecolor="white", linewidth=0.5)
    # Annotate best
    best_idx = int(np.argmin(edps))
    ax.axvline(edps[best_idx] * 1e3, color="red", linestyle="--", linewidth=1.2, label=f"Best: {labels[best_idx]}")
    ax.set_xlabel("EDP (mJ·s × 10³)", fontsize=10)
    ax.set_title("Energy-Delay Product (lower = better)", fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.grid(True, alpha=0.4)

    # ── Panel 2: Area–EDP Pareto scatter ──────────────────────────────────────
    ax = axes[0, 1]
    for c, col in zip(configs, colors):
        marker = _preset_marker(c["system_preset"])
        ax.scatter(c["area_mm2"], c["total_edp"] * 1e3,
                   color=col, marker=marker, s=90, zorder=3)
        ax.annotate(_short_label(c["label"]).split("/")[0],
                    (c["area_mm2"], c["total_edp"] * 1e3),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    # Draw Pareto front
    pareto_idx = _pareto_front(areas.tolist(), edps.tolist())
    px = [areas[i] for i in pareto_idx]
    py = [edps[i] * 1e3 for i in pareto_idx]
    ax.step(px + [px[-1]], py + [py[-1]], where="post",
            color="red", linewidth=1.5, linestyle="--", label="Pareto front", zorder=2)

    # Budget lines
    for bname, blim in BUDGET_LINES.items():
        ax.axvline(blim, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.text(blim + 0.003, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.03,
                bname, fontsize=7, color="gray", va="top")

    legend_patches = [mpatches.Patch(color=mac_color[m], label=f"{m} MACs") for m in mac_vals]
    legend_patches.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", label="high_end BW"))
    legend_patches.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", label="deep_emb BW"))
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")
    ax.set_xlabel("Area (mm²)", fontsize=10)
    ax.set_ylabel("EDP (mJ·s × 10³)", fontsize=10)
    ax.set_title("Area–EDP Trade-off (Pareto front in red)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Per-layer breakdown for top-5 configs by EDP ─────────────────
    ax = axes[1, 0]
    top5_idx = np.argsort(edps)[:5]
    # Union of all layer indices present across the top-5 configs, sorted
    layer_ids = sorted(set(
        l["layer_idx"]
        for ci in top5_idx
        for l in configs[ci]["layers"]
    ))
    x = np.arange(len(layer_ids))
    width = 0.15
    for rank, ci in enumerate(top5_idx):
        layer_map = {l["layer_idx"]: l["edp"] * 1e3 for l in configs[ci]["layers"]}
        layer_edps = [layer_map.get(lid, 0.0) for lid in layer_ids]
        ax.bar(x + rank * width, layer_edps, width,
               label=_short_label(configs[ci]["label"]), color=colors[ci], alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([PROBE_LAYER_NAMES.get(lid, f"T{lid}") for lid in layer_ids], fontsize=9)
    ax.set_ylabel("EDP (mJ·s × 10³)", fontsize=10)
    ax.set_title("Per-Layer EDP — Top-5 Configs", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.grid(True, alpha=0.4)

    # ── Panel 4: Energy vs Latency scatter (bubble = area) ────────────────────
    ax = axes[1, 1]
    bubble_scale = 2000
    for c, col in zip(configs, colors):
        ax.scatter(c["total_latency_s"] * 1e3, c["total_energy_j"] * 1e3,
                   s=c["area_mm2"] * bubble_scale, color=col, alpha=0.7,
                   marker=_preset_marker(c["system_preset"]), edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("Latency (ms)", fontsize=10)
    ax.set_ylabel("Energy (mJ)", fontsize=10)
    ax.set_title("Energy vs Latency  (bubble ∝ area)", fontsize=11)
    legend_patches = [mpatches.Patch(color=mac_color[m], label=f"{m} MACs") for m in mac_vals]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


def plot_budget_sensitivity(data: dict, out_path: Path) -> None:
    """Bar chart comparing best config per budget tier."""
    budgets = list(data.keys())
    metrics = ["total_energy_j", "total_latency_s", "total_edp", "area_mm2"]
    metric_labels = ["Energy (mJ)", "Latency (ms)", "EDP (mJ·s × 10³)", "Area (mm²)"]
    scale = [1e3, 1e3, 1e3, 1.0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Budget Sensitivity: Best Config Per Budget Tier", fontsize=13)

    colors = plt.cm.Set2(np.linspace(0, 0.6, len(budgets)))
    for ax, metric, mlabel, sc in zip(axes, metrics, metric_labels, scale):
        vals, xlabels = [], []
        for b, bdat in data.items():
            best = bdat.get("best")
            if best:
                vals.append(best.get(metric, 0) * sc)
                xlabels.append(f"{b}\n{_short_label(best.get('label',''))}")
            else:
                vals.append(0)
                xlabels.append(f"{b}\n(none)")
        bars = ax.bar(xlabels, vals, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(mlabel, fontsize=10)
        ax.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


def plot_workload_comparison(data: dict, out_path: Path) -> None:
    """Side-by-side 640px vs 320px workload comparison."""
    res640 = data.get("640px", {})
    res320 = data.get("320px", {})
    if not res640 or not res320:
        print("[plot] workload file missing 640px or 320px keys — skipping")
        return

    metrics = [
        ("total_energy_j", "Energy (mJ)", 1e3),
        ("total_latency_s", "Latency (ms)", 1e3),
        ("total_edp", "EDP (mJ·s × 10³)", 1e3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Workload Comparison: 640px vs 320px YOLO-World", fontsize=13)

    for ax, (key, label, sc) in zip(axes, metrics):
        v640 = res640.get(key, 0) * sc
        v320 = res320.get(key, 0) * sc
        bars = ax.bar(["640px", "320px"], [v640, v320], color=["#4C72B0", "#DD8452"],
                      edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        if v640 > 0:
            improvement = (v640 - v320) / v640 * 100
            ax.set_title(f"{label}\n({improvement:.1f}% reduction)", fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


# ── dispatch ───────────────────────────────────────────────────────────────────

def _detect_type(data) -> str:
    """Guess result type from JSON structure."""
    if isinstance(data, list):
        return "probe_sweep"
    if isinstance(data, dict):
        # budget_sensitivity has budget-tier keys
        first_val = next(iter(data.values()), {})
        if isinstance(first_val, dict) and "budget_mm2" in first_val:
            return "budget_sensitivity"
        if "640px" in data or "320px" in data:
            return "workload"
    return "unknown"


def plot_file(path: Path) -> None:
    data = json.loads(path.read_text())
    kind = _detect_type(data)
    stem = path.stem
    out_dir = path.parent

    if kind == "probe_sweep":
        plot_probe_sweep(data, out_dir / f"{stem}_plot.png")
    elif kind == "budget_sensitivity":
        plot_budget_sensitivity(data, out_dir / f"{stem}_plot.png")
    elif kind == "workload":
        plot_workload_comparison(data, out_dir / f"{stem}_plot.png")
    else:
        print(f"[plot] Unrecognized result format in {path.name}; skipping.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot co-design results.")
    parser.add_argument("file", nargs="?", help="Path to a results JSON file")
    parser.add_argument("--list", action="store_true", help="List available result files")
    parser.add_argument("--all", action="store_true", help="Plot all result files in results/")
    args = parser.parse_args()

    if args.list:
        files = sorted(_RESULTS_DIR.glob("*.json"))
        if not files:
            print("No result files found.")
        for f in files:
            print(f"  {f.name}")
        return

    if args.all:
        for f in sorted(_RESULTS_DIR.glob("*.json")):
            if "_plot" not in f.stem:
                plot_file(f)
        return

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            # Try relative to repo root or results dir
            for candidate in [Path.cwd() / path, _RESULTS_DIR / path, path]:
                if candidate.exists():
                    path = candidate
                    break
    else:
        # Default: most recent probe_sweep
        path = _latest_results("probe_sweep")
        print(f"[plot] No file specified — using most recent: {path.name}")

    if not path.exists():
        print(f"[plot] File not found: {path}")
        sys.exit(1)

    plot_file(path)


if __name__ == "__main__":
    main()
