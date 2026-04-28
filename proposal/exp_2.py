"""
Experiment 2 — Workload reshape (input-resolution sensitivity).

Question (proposal):
    Holding hardware fixed, how far can we improve EDP-per-MAC by reshaping
    the workload — specifically, by reducing input resolution so each
    layer's working set fits more comfortably on-chip.

Method:
    For each of three fixed HW configs (tight / preliminary-winner /
    memory-rich), sweep input resolution ∈ {640, 448, 320}² and measure
    per-probe-layer EDP-per-MAC. Three line plots — one per HW config —
    showing the EDP-per-MAC improvement curve as resolution shrinks.

Tile-size axis dropped after Phase 0: the AccelForge mapper returns one
optimal mapping per call rather than the full search frontier, so a clean
post-filter by inner (Pₜ,Qₜ) factor isn't viable. See plan file for
details.

Run inside Docker (sudo on AWS instance):
    > (sudo) docker compose exec labs bash
    > python -m proposal.exp_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milestone_2 import common  # noqa: E402

# ── HW configs (hard-coded) ───────────────────────────────────────────────────
# Three points spanning the area / memory spectrum. The "winner" slot is the
# preliminary winner from the milestone-2 probe sweep — the same probe layers
# (T1, T7, T10, T16, T19) at the same per-layer mapping cap, just with a
# narrower MAC range than Exp 1. If Exp 1 (still running at the time of writing)
# eventually identifies a different EDP-min config (e.g. a 192-MAC point), swap
# the winner row and rerun this script — only the new (HW × resolution × layer)
# cells will need fresh mapping; the cache survives the change.
HW_CONFIGS: list[tuple[str, int, int, int, str]] = [
    # (label,         num_macs, sram_kb, scratch_kb, system_preset)
    ("tight",                128, 384,  32, "high_end_embedded"),
    ("winner_prelim_m2",     128, 512,  64, "high_end_embedded"),
    ("memory_rich",          256, 512, 128, "high_end_embedded"),
]

# Resolutions to sweep. Use the non-robotics resolution-tagged workloads here:
# the robotics head (T21-T23) isn't in the probe set, so its presence is
# irrelevant for Exp 2 measurements but adds noise to cache keys. Using
# yolo_world{,_320,_448}.yaml lets us reuse cached probe-layer mappings from
# milestone-2 results for the 640px points where the HW config matches.
RESOLUTIONS: list[tuple[int, str, list[int]]] = [
    (640, common.WORKLOAD_640, common.EXPECTED_MACS_640),
    (448, common.WORKLOAD_448, common.EXPECTED_MACS_448),
    (320, common.WORKLOAD_320, common.EXPECTED_MACS_320),
]


def probe_total_macs(macs_table: list[int]) -> int:
    return sum(macs_table[i] for i in common.PROBE_LAYERS)


def run_sweep() -> dict:
    print("=" * 100)
    print("EXPERIMENT 2 — Resolution sensitivity at fixed HW")
    print(f"  HW configs: {[c[0] for c in HW_CONFIGS]}")
    print(f"  Resolutions: {[r[0] for r in RESOLUTIONS]}")
    print(f"  Probe layers: {common.PROBE_LAYERS}")
    print("=" * 100)

    # Make sure the resolution workloads exist before we map.
    common.generate_320px_workload()
    common.generate_448px_workload()

    # Build the (HW × resolution) work list. Each row is one common.run_configs
    # call: {one HW config} × {one resolution workload} × probe layers.
    print()
    print(f"  HW configs: {len(HW_CONFIGS)}")
    print(f"  Resolutions per HW: {len(RESOLUTIONS)}")
    print(f"  Cells per (HW, res): {len(common.PROBE_LAYERS)} probe layers")
    print(f"  Expected total mapper cells: "
          f"{len(HW_CONFIGS) * len(RESOLUTIONS) * len(common.PROBE_LAYERS)}\n")

    out: dict = {"by_hw": {}}

    for (hw_label, nmacs, sram, scratch, preset) in HW_CONFIGS:
        area = common.compute_area_mm2(nmacs, sram, scratch)
        print("\n" + "─" * 100)
        print(f"  HW: {hw_label}  ({nmacs} MAC / {sram} KB SRAM / "
              f"{scratch} KB scratch / {preset})  area={area:.3f} mm²")
        print("─" * 100)

        hw_out: dict = {
            "label": hw_label,
            "num_macs": nmacs, "sram_kb": sram, "scratch_kb": scratch,
            "system_preset": preset, "area_mm2": area,
            "by_resolution": {},
        }

        for (res, workload_yaml, macs_table) in RESOLUTIONS:
            label = f"{hw_label}@{res}"
            configs = [(nmacs, sram, scratch, preset, label)]
            results = common.run_configs(workload_yaml, common.PROBE_LAYERS, configs)
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            r = results[0]
            probe_macs = probe_total_macs(macs_table)
            edp_per_mac = r.total_edp / probe_macs if probe_macs else float("inf")

            print(f"  {res:>3}px:  E={r.total_energy_j:.3e} J  "
                  f"L={r.total_latency_s:.3e} s  EDP={r.total_edp:.3e}  "
                  f"MACs={probe_macs:>13,}  EDP/MAC={edp_per_mac:.3e}")

            hw_out["by_resolution"][str(res)] = {
                "resolution": res,
                "workload_file": Path(workload_yaml).name,
                "probe_total_macs": probe_macs,
                "total_energy_j": r.total_energy_j,
                "total_latency_s": r.total_latency_s,
                "total_edp": r.total_edp,
                "edp_per_mac": edp_per_mac,
                "avg_power_mw": r.avg_power_mw,
                "layers": [
                    {"layer_idx": lr.layer_idx, "energy_j": lr.energy_j,
                     "latency_s": lr.latency_s, "edp": lr.edp}
                    for lr in r.layer_results
                ],
            }

        # Per-HW summary table: how does EDP-per-MAC change as resolution shrinks?
        print()
        print(f"  Resolution sensitivity ({hw_label}):")
        baseline_640 = hw_out["by_resolution"]["640"]["edp_per_mac"]
        for res_str in ("640", "448", "320"):
            v = hw_out["by_resolution"][res_str]
            ratio = v["edp_per_mac"] / baseline_640 if baseline_640 else float("inf")
            print(f"    {res_str:>3}px  EDP/MAC={v['edp_per_mac']:.3e}  "
                  f"({ratio:.3f}× baseline 640px)")
        out["by_hw"][hw_label] = hw_out

    return out


def report(out: dict) -> None:
    """Cross-HW comparison table — for each resolution, how does EDP/MAC scale
    across HW configs? Demonstrates the proposal's expected trend (super-linear
    EDP/MAC reduction on tight HW, near-linear on memory-rich)."""
    print("\n" + "=" * 100)
    print("  Cross-HW resolution-sensitivity comparison")
    print("=" * 100)
    print(f"  {'HW':<20}  {'640px':>14}  {'448px':>14}  {'320px':>14}  "
          f"{'320/640':>8}  {'448/640':>8}")
    print("  " + "-" * 96)
    for hw_label, hw in out["by_hw"].items():
        v640 = hw["by_resolution"]["640"]["edp_per_mac"]
        v448 = hw["by_resolution"]["448"]["edp_per_mac"]
        v320 = hw["by_resolution"]["320"]["edp_per_mac"]
        ratio_320 = v320 / v640 if v640 else float("nan")
        ratio_448 = v448 / v640 if v640 else float("nan")
        print(f"  {hw_label:<20}  {v640:>14.3e}  {v448:>14.3e}  "
              f"{v320:>14.3e}  {ratio_320:>8.3f}  {ratio_448:>8.3f}")
    print()
    print("  Expected trend: tight HW shows the largest 320/640 reduction "
          "(super-linear EDP/MAC gain),")
    print("                  memory-rich shows the smallest (near-linear).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=128, metavar="N",
                        help="max_pmapping_templates_per_einsum (default 128)")
    parser.add_argument("--workers", type=int, default=2, metavar="N",
                        help="Parallel workers (default 2 to avoid OOM seen in exp_1)")
    args = parser.parse_args()

    common.configure(pmapping_cap=args.cap, workers=args.workers)
    common.load_cache()

    out = run_sweep()
    report(out)
    common.save_results("exp_2", out)


if __name__ == "__main__":
    main()
