# @Time    : 2026-04-27 15:24
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : exp_1.py

"""
Experiment 1 — Fixed-budget Pareto frontier (hardware allocation).

Question (proposal):
    Under a fixed area + average-power budget, what is the optimal allocation
    across MAC width, SRAM capacity, scratchpad size, and bandwidth preset
    for YOLO-World + a robotics action head?

Method:
    Sweep a hand-curated ~30-40 config grid that lifts the Ethos-U55 SKU
    constraint (MAC ∈ {32, 64, 128, 192, 256, 512}). Map all 5 probe layers
    per config on the robotics workload (cached). Filter by area+power per
    budget tier. Report the EDP-min config in each tier, plus four
    baseline-SKU markers for comparison.

Run inside Docker: (sudo on AWS instance)
    > (sudo) docker compose exec labs bash
    > python -m proposal.exp_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milestone_2 import common  # noqa: E402

# ── Design space (hand-curated from the 6×3×4×2 = 144 Cartesian) ──────────────
# Each row: (num_macs, sram_kb, scratch_kb, system_preset)
# Pruning rules used:
#   - Drop tiny-MAC + large-memory pairings (32 MAC + 512 KB SRAM is wasted area)
#   - Drop big-MAC + tiny-memory pairings (512 MAC + 16 KB scratch is bandwidth-starved)
#   - Cover both BW presets only at the moderate-MAC tiers where BW choice matters
#   - Keep ≥4 configs at the SKU-equivalent tiers (128, 256 MAC) for resolution
DESIGN_SPACE: list[tuple[int, int, int, str]] = [
    # 32 MAC — minimal compute
    ( 32, 256,  16, "deep_embedded"),
    ( 32, 256,  16, "high_end_embedded"),
    ( 32, 384,  32, "deep_embedded"),

    # 64 MAC
    ( 64, 256,  16, "deep_embedded"),
    ( 64, 384,  32, "deep_embedded"),
    ( 64, 384,  32, "high_end_embedded"),

    # 128 MAC — Ethos-U55/128 reference tier
    (128, 256,  16, "deep_embedded"),
    (128, 256,  32, "high_end_embedded"),
    (128, 384,  32, "deep_embedded"),       # ≈ Ethos-U55/128 SKU baseline
    (128, 384,  32, "high_end_embedded"),
    (128, 384,  64, "high_end_embedded"),
    (128, 512,  64, "deep_embedded"),
    (128, 512,  64, "high_end_embedded"),
    (128, 512, 128, "high_end_embedded"),

    # 192 MAC — between SKU steps (proposal expects optimum here)
    (192, 256,  32, "high_end_embedded"),
    (192, 384,  32, "deep_embedded"),
    (192, 384,  32, "high_end_embedded"),
    (192, 384,  64, "high_end_embedded"),
    (192, 512,  64, "high_end_embedded"),
    (192, 512, 128, "high_end_embedded"),

    # 256 MAC — Ethos-U55/256 reference tier
    (256, 384,  32, "deep_embedded"),
    (256, 384,  32, "high_end_embedded"),   # ≈ SKU baseline
    (256, 384,  64, "high_end_embedded"),
    (256, 512,  64, "deep_embedded"),
    (256, 512,  64, "high_end_embedded"),
    (256, 512, 128, "high_end_embedded"),

    # 512 MAC — beyond Arm SKUs (compute-heavy regime)
    (512, 384,  64, "high_end_embedded"),
    (512, 512,  64, "high_end_embedded"),
    (512, 512, 128, "high_end_embedded"),

    # Large-area configs to populate the relaxed budget tier (≤0.80 mm²).
    # SRAM > 512KB is outside Arm's published memory_mode list but reachable
    # via system_sram_size_bytes override.
    (256, 768, 128, "high_end_embedded"),
    (512, 512, 256, "high_end_embedded"),
    (512, 768, 128, "high_end_embedded"),
    (512, 768, 256, "high_end_embedded"),
]

# ── Approximate Ethos-U55 SKUs (for plot annotation / table marking) ──────────
# Arm-published locals are 16/16/24/48 KB at MAC widths 32/64/128/256 — we
# approximate (16,16,32,64). All SKUs are "high_end_embedded" preset.
SKU_MARKERS: list[tuple[str, int, int, int, str]] = [
    ("SKU-32",   32, 384, 16, "high_end_embedded"),
    ("SKU-64",   64, 384, 16, "high_end_embedded"),
    ("SKU-128", 128, 384, 32, "high_end_embedded"),
    ("SKU-256", 256, 384, 64, "high_end_embedded"),
]

BUDGETS: dict[str, dict] = {
    "tight":    {"area_mm2": 0.20, "power_mw":  50,  "label": "Tight    (≤0.20 mm² / ≤50 mW)"},
    "baseline": {"area_mm2": 0.38, "power_mw": 100,  "label": "Baseline (≤0.38 mm² / ≤100 mW)"},
    "relaxed":  {"area_mm2": 0.80, "power_mw": 500,  "label": "Relaxed  (≤0.80 mm² / ≤500 mW)"},
}

WORKLOAD_PATHS: dict[str, str] = {
    "robotics": common.WORKLOAD_ROBOTICS,  # default
    "yolo":     common.WORKLOAD_640,
    "320":      common.WORKLOAD_320,
}


def _format_label(nmacs: int, sram: int, scratch: int, preset: str) -> str:
    return f"{nmacs:>3}M/{sram:>3}KB/{scratch:>3}KB/{preset[:4]}"


def run_sweep(workload_yaml: str) -> list[common.ArchResult]:
    print("=" * 100)
    print(f"EXPERIMENT 1 — Fixed-budget Pareto sweep")
    print(f"  Workload: {Path(workload_yaml).name}")
    print(f"  Probe layers: {common.PROBE_LAYERS}")
    print(f"  Configs: {len(DESIGN_SPACE)}  +  {len(SKU_MARKERS)} SKU markers")
    print("=" * 100)

    # Combine design space and SKUs (deduped). SKUs that already appear in
    # DESIGN_SPACE just gain a "(SKU)" suffix on their label.
    combined: dict[tuple, str] = {}
    for (nmacs, sram, scratch, preset) in DESIGN_SPACE:
        combined[(nmacs, sram, scratch, preset)] = _format_label(nmacs, sram, scratch, preset)
    for (sku_name, nmacs, sram, scratch, preset) in SKU_MARKERS:
        key = (nmacs, sram, scratch, preset)
        suffix = f" [{sku_name}]"
        if key in combined and not combined[key].endswith(suffix):
            combined[key] = combined[key] + suffix
        else:
            combined[key] = _format_label(nmacs, sram, scratch, preset) + suffix

    configs = []
    skipped = 0
    for (nmacs, sram, scratch, preset), label in combined.items():
        area = common.compute_area_mm2(nmacs, sram, scratch)
        if area > 0.80 * 1.01:
            print(f"  Skip {label}  area={area:.3f} mm² > 0.80 mm² hard cap")
            skipped += 1
            continue
        configs.append((nmacs, sram, scratch, preset, label))

    print(f"  Running {len(configs)} configs ({skipped} skipped by hard area cap)\n")
    expected_calls = len(configs) * len(common.PROBE_LAYERS)
    print(f"  Expected (config × layer) cell count: {expected_calls}\n")

    results = common.run_configs(workload_yaml, common.PROBE_LAYERS, configs)
    return results


def report(results: list[common.ArchResult]) -> dict:
    """Print sweep table per budget tier and pareto front. Return JSON-serializable summary."""
    out: dict = {"workload": None, "configs": [common.arch_result_to_dict(r) for r in results]}

    for tier_name, b in BUDGETS.items():
        print("\n" + "=" * 100)
        print(f"  Budget tier: {b['label']}")
        print("=" * 100)
        common.print_sweep_table(results, budget_mm2=b["area_mm2"], power_mw=b["power_mw"])

        valid = [r for r in results
                 if r.area_mm2 <= b["area_mm2"] and r.avg_power_mw <= b["power_mw"]]
        if valid:
            best = min(valid, key=lambda r: r.total_edp)
            out[f"best_{tier_name}"] = {
                "label": best.label,
                "num_macs": best.num_macs,
                "sram_kb": best.sram_kb,
                "scratch_kb": best.scratch_kb,
                "system_preset": best.system_preset,
                "area_mm2": best.area_mm2,
                "avg_power_mw": best.avg_power_mw,
                "total_edp": best.total_edp,
            }
            print(f"\n  → BEST in tier: {best.label}")
            print(f"     EDP={best.total_edp:.3e}  area={best.area_mm2:.3f} mm²  "
                  f"power={best.avg_power_mw:.1f} mW")
        else:
            out[f"best_{tier_name}"] = None
            print("  → No configs satisfy this tier.")

    # Pareto front on (area, EDP) ignoring power for the visualization step.
    front = common.pareto_front(results, x_key=lambda r: r.area_mm2,
                                y_key=lambda r: r.total_edp)
    print("\n" + "=" * 100)
    print(f"  Pareto front  (area vs EDP, {len(front)} points)")
    print("=" * 100)
    for r in front:
        print(f"  {r.label:<48}  area={r.area_mm2:>6.3f} mm²  "
              f"power={r.avg_power_mw:>6.1f} mW  EDP={r.total_edp:.3e}")
    out["pareto_front_area_edp"] = [common.arch_result_to_dict(r) for r in front]

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="robotics",
                        help="Workload YAML to map (default: robotics)")
    parser.add_argument("--cap", type=int, default=128, metavar="N",
                        help="max_pmapping_templates_per_einsum (default 128)")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    common.configure(pmapping_cap=args.cap, workers=args.workers)
    common.load_cache()

    if args.workload == "robotics":
        common.generate_robotics_workload()
    elif args.workload == "320":
        common.generate_320px_workload()

    workload_yaml = WORKLOAD_PATHS[args.workload]
    results = run_sweep(workload_yaml)
    summary = report(results)
    summary["workload"] = args.workload
    common.save_results(f"exp_1_{args.workload}", summary)


if __name__ == "__main__":
    main()
