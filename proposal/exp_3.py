"""
Experiment 3 — Relative impact of area vs. power constraints.

Question (proposal):
    How does fixing one constraint (area or power) and sweeping the other
    affect achievable EDP? Which constraint is more binding at each tier?

Method:
    Reuse Exp 1's mapping data — no new mapper calls. For each cell of a
    3×3 grid {area ≤ 0.10 / 0.38 / 0.80 mm²} × {power ≤ 50 / 100 / 500 mW},
    pick the EDP-min config that satisfies both ceilings. Report the chosen
    arch tuple, achieved EDP, and area/power utilization per cell.

Run inside Docker (sudo on AWS instance):
    > (sudo) docker compose exec labs bash
    > python -m proposal.exp_3                   # uses default workload (robotics)
    > python -m proposal.exp_3 --workload yolo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milestone_2 import common  # noqa: E402
from proposal.exp_1 import DESIGN_SPACE, SKU_MARKERS, WORKLOAD_PATHS, _format_label  # noqa: E402

# 3 × 3 = 9 (area, power) cells.
AREA_TIERS_MM2 = [0.10, 0.38, 0.80]
POWER_TIERS_MW = [50, 100, 500]


def gather_results(workload_yaml: str) -> list[common.ArchResult]:
    """Replay the exp_1 design-space configs against the cache only.

    No mapper calls happen here — every probe-layer entry is expected to be
    cached. If anything is missing, run exp_1 first.
    """
    combined: dict[tuple, str] = {}
    for (nmacs, sram, scratch, preset) in DESIGN_SPACE:
        combined[(nmacs, sram, scratch, preset)] = _format_label(nmacs, sram, scratch, preset)
    for (sku_name, nmacs, sram, scratch, preset) in SKU_MARKERS:
        key = (nmacs, sram, scratch, preset)
        suffix = f" [{sku_name}]"
        combined[key] = (combined.get(key, _format_label(nmacs, sram, scratch, preset)) + suffix)

    configs = []
    for (nmacs, sram, scratch, preset), label in combined.items():
        area = common.compute_area_mm2(nmacs, sram, scratch)
        if area > 0.80 * 1.01:
            continue
        configs.append((nmacs, sram, scratch, preset, label))

    print(f"  Pulling {len(configs)} configs × {len(common.PROBE_LAYERS)} probe layers from cache ...")
    cache_size_before = common.cache_size()
    results = common.run_configs(workload_yaml, common.PROBE_LAYERS, configs)
    cache_size_after = common.cache_size()
    new_calls = cache_size_after - cache_size_before
    print(f"  Cache size: {cache_size_before} → {cache_size_after}  "
          f"(new mapper calls: {new_calls})")
    if new_calls > 0:
        print(f"  ⚠ {new_calls} cells were not cached — exp_1 likely hasn't covered them.")
        print("    Exp 3 results may be incomplete. Re-run after exp_1 finishes.")
    return results


def grid_table(results: list[common.ArchResult]) -> dict:
    """For each (area, power) cell, find the EDP-min config among the feasible ones."""
    print()
    print("=" * 110)
    print("EXPERIMENT 3 — 3×3 (area × power) constraint grid")
    print("=" * 110)
    print(f"  Area tiers:  {AREA_TIERS_MM2} mm²")
    print(f"  Power tiers: {POWER_TIERS_MW} mW")
    print()
    print(f"  {'area / power':>14} | " + " | ".join(f"{p:>6} mW" for p in POWER_TIERS_MW))
    print("  " + "-" * (14 + 3 + (len(POWER_TIERS_MW) * 12)))

    grid: dict[str, dict] = {}
    for a in AREA_TIERS_MM2:
        row = []
        for p in POWER_TIERS_MW:
            valid = [r for r in results if r.area_mm2 <= a and r.avg_power_mw <= p]
            if not valid:
                row.append("    —    ")
                grid[f"{a:.2f}_{p}"] = None
                continue
            best = min(valid, key=lambda r: r.total_edp)
            row.append(f"{best.total_edp:.2e}")
            grid[f"{a:.2f}_{p}"] = {
                "area_cap_mm2": a,
                "power_cap_mw": p,
                "n_feasible": len(valid),
                "best": {
                    "label": best.label,
                    "num_macs": best.num_macs,
                    "sram_kb": best.sram_kb,
                    "scratch_kb": best.scratch_kb,
                    "system_preset": best.system_preset,
                    "area_mm2": best.area_mm2,
                    "avg_power_mw": best.avg_power_mw,
                    "total_edp": best.total_edp,
                    "area_utilization": best.area_mm2 / a,
                    "power_utilization": best.avg_power_mw / p,
                },
            }
        print(f"  {f'≤{a:.2f} mm²':>14} | " + " | ".join(f"{c:>9}" for c in row))

    print()
    print("  Per-cell winners (binding constraint = the one closer to 100 % utilization):")
    print(f"  {'cell':<22}  {'config':<42}  {'EDP':>10}  {'area-util':>10}  {'pwr-util':>10}  binding")
    print("  " + "-" * 110)
    for key, cell in grid.items():
        if cell is None:
            print(f"  {key:<22}  (no feasible config)")
            continue
        b = cell["best"]
        binding = "AREA" if b["area_utilization"] > b["power_utilization"] else "POWER"
        print(f"  {key:<22}  {b['label']:<42}  {b['total_edp']:>10.3e}  "
              f"{b['area_utilization']:>9.1%}  {b['power_utilization']:>9.1%}  {binding}")
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PATHS), default="robotics",
                        help="Workload (must match what exp_1 used). Default: robotics.")
    parser.add_argument("--cap", type=int, default=128, metavar="N",
                        help="Pmapping cap if a cache miss forces a fresh mapper call (default 128)")
    parser.add_argument("--workers", type=int, default=2, metavar="N",
                        help="Parallel workers for any cache misses (default 2)")
    args = parser.parse_args()

    common.configure(pmapping_cap=args.cap, workers=args.workers)
    common.load_cache()

    workload_yaml = WORKLOAD_PATHS[args.workload]
    if args.workload == "robotics":
        common.generate_robotics_workload()
    elif args.workload == "320":
        common.generate_320px_workload()

    results = gather_results(workload_yaml)
    grid = grid_table(results)

    summary = {
        "workload": args.workload,
        "area_tiers_mm2": AREA_TIERS_MM2,
        "power_tiers_mw": POWER_TIERS_MW,
        "grid": grid,
        "all_configs": [common.arch_result_to_dict(r) for r in results],
    }
    common.save_results(f"exp_3_{args.workload}", summary)


if __name__ == "__main__":
    main()
