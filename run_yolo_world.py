"""
Mapper script for YOLO-World-S on your AccelForge accelerator.

Usage (from the lab_5/ directory):
    python run_yolo_world.py

Workflow:
  1. Set ARCH_NAME to your partner's architecture name (filename without .yaml).
  2. Run the sanity check first (T0 only, max_layers=1) to confirm setup works.
  3. Run the full workload and inspect per-layer energy and component breakdown.
"""

import os
import sys

# Ensure lab_5/ is on the path so scripts/utils.py is importable.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from scripts.utils import run_dnn_layers

# ── Configuration ─────────────────────────────────────────────────────────────

# Replace with your partner's architecture YAML name (no .yaml extension).
ARCH_NAME = "basic_analog"

# Optional architecture variable overrides — uncomment/edit as needed.
# Valid keys depend on your partner's arch template variables.
VARIABLE_OVERRIDES = {
    # "ARRAY_ROWS": 256,
    # "ARRAY_COLS": 256,
    # "ADC_RESOLUTION": 6,
    # "BITS_PER_CELL": 2,
}

# Human-readable names for each layer T0–T20.
LAYER_NAMES = [
    "T0  sanity_check   3x3  8x8x4 -> 8x8x8",
    "T1  backbone_stem  3x3s2  640x640x3 -> 320x320x32",
    "T2  backbone_L1    3x3s2  320x320x32 -> 160x160x64",
    "T3  backbone_C2f1  3x3  160x160x32 -> 160x160x32",
    "T4  backbone_L2    3x3s2  160x160x64 -> 80x80x128",
    "T5  backbone_C2f2a 3x3  80x80x64 -> 80x80x64",
    "T6  backbone_C2f2b 3x3  80x80x64 -> 80x80x64",
    "T7  backbone_L3    3x3s2  80x80x128 -> 40x40x256",
    "T8  backbone_C2f3a 3x3  40x40x128 -> 40x40x128",
    "T9  backbone_C2f3b 3x3  40x40x128 -> 40x40x128",
    "T10 backbone_L4    3x3s2  40x40x256 -> 20x20x512",
    "T11 backbone_C2f4  3x3  20x20x256 -> 20x20x256",
    "T12 sppf_1x1       1x1  20x20x512 -> 20x20x256",
    "T13 text_Q_proj    77x512 -> 77x512",
    "T14 text_K_proj    77x512 -> 77x512",
    "T15 text_V_proj    77x512 -> 77x512",
    "T16 text_FFN_up    77x512 -> 77x2048",
    "T17 text_FFN_down  77x2048 -> 77x512",
    "T18 repvl_P5_1x1   1x1  20x20x512 -> 20x20x512",
    "T19 det_head_P3    3x3  80x80x128 -> 80x80x128",
    "T20 det_head_P4    3x3  40x40x256 -> 40x40x256",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_results(results, start_idx=0):
    """Print a per-layer table and an aggregate component energy breakdown."""
    print(f"\n{'Idx':<5} {'Layer':<45} {'MACs':>12} {'Energy(J)':>13} {'pJ/MAC':>8}")
    print("-" * 87)

    total_energy = 0.0
    total_macs = 0
    all_components: dict[str, float] = {}

    for offset, r in enumerate(results):
        i = start_idx + offset
        name = LAYER_NAMES[i] if i < len(LAYER_NAMES) else f"T{i}"
        layer_e = sum(r.per_component_energy.values())
        pj_per_mac = r.per_compute("energy") * 1e12
        total_energy += layer_e
        total_macs += r.computes
        for comp, e in r.per_component_energy.items():
            all_components[comp] = all_components.get(comp, 0.0) + e
        print(f"{i:<5} {name:<45} {r.computes:>12,} {layer_e:>13.3e} {pj_per_mac:>8.3f}")

    print("-" * 87)
    avg_pj = (total_energy / total_macs * 1e12) if total_macs else 0.0
    print(f"{'TOTAL':<51} {total_macs:>12,} {total_energy:>13.3e} {avg_pj:>8.3f}")

    if total_energy > 0:
        print("\nComponent breakdown (summed over all layers):")
        for comp, e in sorted(all_components.items(), key=lambda x: -x[1]):
            if e > 0:
                pct = 100.0 * e / total_energy
                print(f"  {comp:<36} {e:.3e} J  ({pct:.1f}%)")
    print()
    return total_energy, total_macs


# ── Sanity check ──────────────────────────────────────────────────────────────

def run_sanity_check():
    """Map only T0 (tiny 3x3 conv, 18,432 MACs) to confirm the setup is correct."""
    print("=" * 65)
    print("SANITY CHECK  --  T0: 3x3 conv  8x8x4 -> 8x8x8  (18,432 MACs)")
    print(f"Architecture : {ARCH_NAME}")
    print("=" * 65)

    results = run_dnn_layers(
        ARCH_NAME,
        "yolo_world",
        variable_overrides=VARIABLE_OVERRIDES,
        max_layers=1,
        batch_size=1,
    )
    if not results:
        print("[ERROR] No result returned -- check ARCH_NAME and setup.")
        return None

    _print_results(results, start_idx=0)
    return results


# ── Full YOLO-World-S run ─────────────────────────────────────────────────────

def run_yolo_world(max_layers=21):
    """Map all YOLO-World-S layers (T0-T20) and report per-layer energy."""
    print("=" * 65)
    print(f"YOLO-WORLD-S  --  layers T0-T{max_layers - 1}")
    print(f"Architecture : {ARCH_NAME}")
    print("=" * 65)

    results = run_dnn_layers(
        ARCH_NAME,
        "yolo_world",
        variable_overrides=VARIABLE_OVERRIDES,
        max_layers=max_layers,
        batch_size=1,
    )
    if not results:
        print("[ERROR] No results returned -- check ARCH_NAME and setup.")
        return None

    total_e, total_macs = _print_results(results, start_idx=0)

    # Identify the most expensive layer.
    layer_energies = [sum(r.per_component_energy.values()) for r in results]
    worst = max(range(len(layer_energies)), key=lambda i: layer_energies[i])
    worst_name = LAYER_NAMES[worst] if worst < len(LAYER_NAMES) else f"T{worst}"
    print(f"Most expensive layer: {worst_name}")
    print(f"  Energy : {layer_energies[worst]:.3e} J  "
          f"({100 * layer_energies[worst] / total_e:.1f}% of total)")
    print()

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: confirm the toolchain works on a trivial input.
    run_sanity_check()

    # Step 2: map the full model.
    run_yolo_world()
