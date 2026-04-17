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
    bubble_scale = 800
    for c, col in zip(configs, colors):
        ax.scatter(c["total_latency_s"] * 1e3, c["total_energy_j"] * 1e3,
                   s=c["area_mm2"] * bubble_scale, color=col, alpha=0.35,
                   marker=_preset_marker(c["system_preset"]), edgecolors=col, linewidths=0.8, zorder=3)
    ax.set_xlabel("Latency (ms)", fontsize=10)
    ax.set_ylabel("Energy (mJ)", fontsize=10)
    ax.set_title("Energy vs Latency  (bubble ∝ area)", fontsize=11)
    legend_patches = [mpatches.Patch(color=mac_color[m], label=f"{m} MACs") for m in mac_vals]
    legend_patches.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", label="high_end BW"))
    legend_patches.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", label="deep_emb BW"))
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


def plot_budget_sensitivity(data: dict, out_path: Path) -> None:
    """4-panel budget sensitivity visualization using all configs per tier."""
    budgets = list(data.keys())
    tier_cmap = plt.cm.Set1
    tier_colors = {b: tier_cmap(i / max(len(budgets) - 1, 1)) for i, b in enumerate(budgets)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Budget Sensitivity: Area–EDP Co-design Trade-offs", fontsize=14, y=0.98)

    # ── Panel 1: Area–EDP scatter — all configs, colored by budget tier ───────
    ax = axes[0, 0]
    for tier, bdat in data.items():
        col = tier_colors[tier]
        budget_limit = bdat.get("budget_mm2", None)
        all_cfgs = bdat.get("all", [])
        best = bdat.get("best")
        if not all_cfgs and best:
            all_cfgs = [best]
        xs = [c["area_mm2"] for c in all_cfgs]
        ys = [c["total_edp"] * 1e3 for c in all_cfgs]
        ax.scatter(xs, ys, color=col, alpha=0.55, s=60, zorder=3,
                   label=f"{tier} (≤{budget_limit} mm²)" if budget_limit else tier)
        # Highlight best
        if best:
            ax.scatter(best["area_mm2"], best["total_edp"] * 1e3,
                       color=col, s=180, marker="*", edgecolors="black",
                       linewidths=0.8, zorder=5)
        # Pareto front within this tier
        if len(all_cfgs) > 1:
            px = [c["area_mm2"] for c in all_cfgs]
            py = [c["total_edp"] for c in all_cfgs]
            pidx = _pareto_front(px, py)
            ppx = [px[i] for i in pidx]
            ppy = [py[i] * 1e3 for i in pidx]
            ax.step(ppx + [ppx[-1]], ppy + [ppy[-1]], where="post",
                    color=col, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)
        # Budget limit vertical line
        if budget_limit:
            ax.axvline(budget_limit, color=col, linestyle=":", linewidth=0.9, alpha=0.6)

    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Area (mm²)", fontsize=10)
    ax.set_ylabel("EDP (mJ·s × 10³)", fontsize=10)
    ax.set_title("Area–EDP Trade-offs by Budget Tier\n(★ = best; dashed = Pareto front; dotted = limit)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: EDP comparison — all configs per tier as grouped bars ────────
    ax = axes[0, 1]
    x_pos = 0
    xticks, xticklabels = [], []
    gap = 0.4
    for tier, bdat in data.items():
        col = tier_colors[tier]
        all_cfgs = bdat.get("all", [])
        best = bdat.get("best")
        if not all_cfgs and best:
            all_cfgs = [best]
        sorted_cfgs = sorted(all_cfgs, key=lambda c: c["total_edp"])
        tier_xs = []
        for c in sorted_cfgs:
            edp_val = c["total_edp"] * 1e3
            bar_col = col if (best and c.get("label") != best.get("label")) else "none"
            edge_col = col
            hatch = "" if (best and c.get("label") == best.get("label")) else ".."
            ax.bar(x_pos, edp_val, color=col,
                   alpha=1.0 if (best and c.get("label") == best.get("label")) else 0.45,
                   edgecolor="black", linewidth=0.4, hatch=hatch if hatch else None, width=0.7)
            tier_xs.append(x_pos)
            x_pos += 1
        # Tier label centered under the group
        if tier_xs:
            xticks.append(np.mean(tier_xs))
            xticklabels.append(f"{tier}\n({len(sorted_cfgs)} cfgs)")
        x_pos += gap

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_ylabel("EDP (mJ·s × 10³)", fontsize=10)
    ax.set_title("EDP of All Configs Per Budget Tier\n(solid = best, hatched = others)", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    # Legend: one patch per tier
    patches = [mpatches.Patch(color=tier_colors[b], label=b) for b in budgets]
    ax.legend(handles=patches, fontsize=8)

    # ── Panel 3: Per-layer EDP for best config in each tier ───────────────────
    ax = axes[1, 0]
    all_layer_ids = sorted(set(
        l["layer_idx"]
        for bdat in data.values()
        for l in (bdat.get("best") or {}).get("layers", [])
    ))
    x = np.arange(len(all_layer_ids))
    n_tiers = len(budgets)
    width = 0.8 / max(n_tiers, 1)
    for ti, (tier, bdat) in enumerate(data.items()):
        best = bdat.get("best")
        if not best:
            continue
        layer_map = {l["layer_idx"]: l["edp"] * 1e3 for l in best.get("layers", [])}
        layer_edps = [layer_map.get(lid, 0.0) for lid in all_layer_ids]
        offset = (ti - n_tiers / 2 + 0.5) * width
        ax.bar(x + offset, layer_edps, width, label=f"{tier}: {_short_label(best['label'])}",
               color=tier_colors[tier], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([PROBE_LAYER_NAMES.get(lid, f"T{lid}") for lid in all_layer_ids], fontsize=9)
    ax.set_ylabel("EDP (mJ·s × 10³)", fontsize=10)
    ax.set_title("Per-Layer EDP — Best Config Per Budget Tier", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.grid(True, alpha=0.4)

    # ── Panel 4: Best-config summary metrics bar chart ────────────────────────
    ax = axes[1, 1]
    metrics = [
        ("total_energy_j", "Energy (mJ)", 1e3),
        ("total_latency_s", "Latency (ms)", 1e3),
        ("total_edp", "EDP (×10³)", 1e3),
        ("avg_power_mw", "Avg Power (mW)", 1.0),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / max(n_tiers, 1)
    for ti, (tier, bdat) in enumerate(data.items()):
        best = bdat.get("best")
        if not best:
            continue
        vals = [best.get(m, 0) * sc for m, _, sc in metrics]
        offset = (ti - n_tiers / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"{tier}: {_short_label(best['label'])}",
                      color=tier_colors[tier], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in metrics], fontsize=9)
    ax.set_title("Best-Config Summary Metrics Per Tier", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


def plot_workload_comparison(tiers: list[dict], out_path: Path) -> None:
    """
    4-panel 640px vs 320px comparison across hardware tiers.

    Each entry in `tiers` is a dict with keys: hw_config, r640, r320.
    """
    if not tiers:
        print("[plot] no workload data — skipping")
        return

    tier_cmap = plt.cm.Set1
    tier_colors = {t["hw_config"]["label"]: tier_cmap(i / max(len(tiers) - 1, 1))
                   for i, t in enumerate(tiers)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Workload Co-design: 640px vs 320px YOLO-World", fontsize=14, y=0.98)

    metrics = [
        ("total_energy_j", "Energy (mJ)", 1e3),
        ("total_latency_s", "Latency (ms)", 1e3),
        ("total_edp", "EDP (mJ·s × 10³)", 1e3),
    ]

    # ── Panel 1: Grouped bar — Energy, Latency, EDP per tier ──────────────────
    ax = axes[0, 0]
    x = np.arange(len(metrics))
    n = len(tiers)
    width = 0.8 / (n * 2)
    for ti, t in enumerate(tiers):
        hw = t["hw_config"]["label"]
        col = tier_colors[hw]
        r640, r320 = t.get("r640", {}), t.get("r320", {})
        vals640 = [r640.get(m, 0) * sc for m, _, sc in metrics]
        vals320 = [r320.get(m, 0) * sc for m, _, sc in metrics]
        base = (ti * 2 - n + 0.5) * width
        ax.bar(x + base,          vals640, width, color=col, alpha=0.9, edgecolor="white", label=f"{hw} 640px")
        ax.bar(x + base + width,  vals320, width, color=col, alpha=0.45, edgecolor="white", hatch="//", label=f"{hw} 320px")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in metrics], fontsize=9)
    ax.set_title("640px (solid) vs 320px (hatched) — all tiers", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # ── Panel 2: Speedup / energy reduction ratios ────────────────────────────
    ax = axes[0, 1]
    tier_labels = [t["hw_config"]["label"] for t in tiers]
    x = np.arange(len(tier_labels))
    width = 0.25
    ratio_metrics = [
        ("total_latency_s", "Latency speedup", "#4C72B0"),
        ("total_energy_j",  "Energy reduction", "#DD8452"),
        ("total_edp",       "EDP reduction",    "#59A14F"),
    ]
    for ri, (key, lbl, col) in enumerate(ratio_metrics):
        ratios = []
        for t in tiers:
            v640 = t.get("r640", {}).get(key, 0)
            v320 = t.get("r320", {}).get(key, 0)
            ratios.append(v640 / v320 if v320 > 0 else 0)
        offset = (ri - 1) * width
        bars = ax.bar(x + offset, ratios, width, color=col, alpha=0.85, edgecolor="white", label=lbl)
        ax.bar_label(bars, fmt="%.2fx", fontsize=7, padding=2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=8)
    ax.set_ylabel("640px / 320px ratio  (>1 = 320px faster/cheaper)", fontsize=9)
    ax.set_title("Resolution Reduction Ratios by Hardware Tier", fontsize=10)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)

    # ── Panel 3: Per-layer energy — 640px vs 320px for baseline tier ──────────
    ax = axes[1, 0]
    # Use baseline tier if present, else first
    base_tier = next((t for t in tiers if "baseline" in t["hw_config"]["label"].lower()), tiers[0])
    r640_layers = {l["layer_idx"]: l for l in base_tier.get("r640", {}).get("layers", [])}
    r320_layers = {l["layer_idx"]: l for l in base_tier.get("r320", {}).get("layers", [])}
    layer_ids = sorted(set(r640_layers) | set(r320_layers))
    x = np.arange(len(layer_ids))
    width = 0.35
    e640 = [r640_layers.get(lid, {}).get("energy_j", 0) * 1e3 for lid in layer_ids]
    e320 = [r320_layers.get(lid, {}).get("energy_j", 0) * 1e3 for lid in layer_ids]
    ax.bar(x - width/2, e640, width, color="#4C72B0", alpha=0.85, edgecolor="white", label="640px")
    ax.bar(x + width/2, e320, width, color="#DD8452", alpha=0.85, edgecolor="white", label="320px")
    ax.set_xticks(x)
    ax.set_xticklabels([PROBE_LAYER_NAMES.get(lid, f"T{lid}") for lid in layer_ids], fontsize=9)
    ax.set_ylabel("Energy (mJ)", fontsize=10)
    ax.set_title(f"Per-Layer Energy: {base_tier['hw_config']['label']}", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)

    # ── Panel 4: Per-layer latency speedup across tiers ───────────────────────
    ax = axes[1, 1]
    for ti, t in enumerate(tiers):
        hw = t["hw_config"]["label"]
        col = tier_colors[hw]
        r640_l = {l["layer_idx"]: l for l in t.get("r640", {}).get("layers", [])}
        r320_l = {l["layer_idx"]: l for l in t.get("r320", {}).get("layers", [])}
        ids = sorted(set(r640_l) & set(r320_l))
        speedups = [r640_l[lid]["latency_s"] / r320_l[lid]["latency_s"]
                    if r320_l[lid]["latency_s"] > 0 else 0 for lid in ids]
        ax.plot([PROBE_LAYER_NAMES.get(lid, f"T{lid}") for lid in ids], speedups,
                marker="o", color=col, linewidth=1.5, label=hw)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Latency 640px / 320px  (>1 = 320px faster)", fontsize=9)
    ax.set_title("Per-Layer Latency Speedup from 320px", fontsize=10)
    ax.legend(fontsize=8)
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
            return "workload_legacy"
        if "r640" in data and "r320" in data:
            return "workload"
    return "unknown"


def _collect_workload_tiers(path: Path) -> tuple[list[dict], Path]:
    """
    Workload results are written as one file per hardware tier
    (workload_tight_*.json, workload_baseline_*.json, workload_relaxed_*.json).
    Collect all sibling files with the same timestamp and combine them.
    Returns (list_of_tier_dicts, output_path).
    """
    # Extract timestamp suffix from filename (e.g. "20260417_005002")
    stem = path.stem  # e.g. "workload_baseline_20260417_005002"
    parts = stem.split("_")
    # timestamp is last two underscore-parts joined
    ts = "_".join(parts[-2:])
    siblings = sorted(path.parent.glob(f"workload_*_{ts}.json"))
    tiers = []
    for s in siblings:
        d = json.loads(s.read_text())
        if "r640" in d and "r320" in d:
            tiers.append(d)
    out_path = path.parent / f"workload_{ts}_plot.png"
    return tiers, out_path


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
        tiers, out_path = _collect_workload_tiers(path)
        plot_workload_comparison(tiers, out_path)
    elif kind == "workload_legacy":
        # old single-file format with 640px/320px keys
        plot_workload_comparison(
            [{"hw_config": {"label": stem}, "r640": data.get("640px", {}), "r320": data.get("320px", {})}],
            out_dir / f"{stem}_plot.png",
        )
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
        seen_workload_ts: set[str] = set()
        for f in sorted(_RESULTS_DIR.glob("*.json")):
            if "_plot" in f.stem:
                continue
            # For workload tier files, only plot once per timestamp
            if f.stem.startswith("workload_"):
                parts = f.stem.split("_")
                ts = "_".join(parts[-2:])
                if ts in seen_workload_ts:
                    continue
                seen_workload_ts.add(ts)
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
