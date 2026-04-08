"""
mapper_script.py
Run the AccelForge mapper on YOLO-World-S workload layers for a custom edge accelerator.

Usage (from the lab_5/ directory):
    python mapper_script.py

    # Map only the sanity-check layer (T0):
    python mapper_script.py --sanity

    # Map a specific layer by index:
    python mapper_script.py --layer 1

    # Map layers T0-T12 (backbone only):
    python mapper_script.py --max-layers 13

Usage (from a notebook):
    from mapper_script import run_mapper, evaluate_mapping, min_edp_filter

Steps:
  1. Set ARCH_FILE to your partner's architecture YAML path.
  2. Run with --sanity first to confirm the toolchain works end-to-end on T0.
  3. Run the full model and inspect the EDP breakdown.
"""

import argparse
import os
import sys

import accelforge as af
from accelforge.mapper import Metrics

# ── Configuration ─────────────────────────────────────────────────────────────

# Update to your partner's architecture YAML when it is ready.
ARCH_FILE    = "arch/arch.yaml"
WORKLOAD_FILE = "workload/yolo_world.yaml"

# Variable overrides for parameterised architectures (e.g. array size, ADC bits).
# Keys must match the template variables defined in ARCH_FILE.
VARIABLE_OVERRIDES: dict[str, int | float] = {
    # "ARRAY_ROWS": 256,
    # "ARRAY_COLS": 256,
    # "ADC_RESOLUTION": 6,
    # "BITS_PER_CELL": 2,
}

# Human-readable label for each einsum index (T0–T20).
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


# ── Core mapper helpers ────────────────────────────────────────────────────────

def min_edp_filter(df):
    """Return the index of the row with the lowest Energy-Delay Product."""
    return (df["energy"] * df["latency"]).idxmin()


def run_mapper(workload_file, arch_file=ARCH_FILE, variable_overrides=None):
    """
    Auto-map a workload onto the architecture.

    Returns a Mappings object. Index it with [min_edp_filter(mappings.data)]
    to retrieve the minimum-EDP mapping.
    """
    spec = af.Spec.from_yaml(arch_file, workload_file,
                             variable_overrides=variable_overrides or {})
    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
    return spec.map_workload_to_arch()


def evaluate_mapping(workload_file, mapping_file,
                     arch_file=ARCH_FILE, sparse_file=None,
                     variable_overrides=None):
    """
    Evaluate a hand-crafted mapping YAML against a workload.

    Returns:
        (energy_pJ, latency_cycles) as floats.
    """
    files = [arch_file, workload_file, mapping_file]
    if sparse_file:
        files.append(sparse_file)
    spec = af.Spec.from_yaml(*files, variable_overrides=variable_overrides or {})
    result = spec.evaluate_mapping()
    return float(result.energy()), float(result.latency())


# ── Per-layer helpers ──────────────────────────────────────────────────────────

def _layer_name(idx):
    return LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"T{idx}"


def _map_layer(layer_idx, workload_file=WORKLOAD_FILE, arch_file=ARCH_FILE):
    """Map a single einsum (by index) and return (energy_pJ, latency_cycles, edp)."""
    mappings = run_mapper(workload_file, arch_file,
                          variable_overrides=VARIABLE_OVERRIDES)
    best = mappings[min_edp_filter(mappings.data)]
    energy  = float(best.energy())
    latency = float(best.latency())
    return energy, latency, energy * latency


def _print_table(rows):
    """
    rows: list of (layer_idx, energy_pJ, latency_cycles, edp)
    """
    hdr = f"{'Idx':<5} {'Layer':<45} {'Energy (pJ)':>14} {'Latency (cyc)':>14} {'EDP':>14}"
    print(hdr)
    print("-" * len(hdr))
    for (idx, e, lat, edp) in rows:
        print(f"{idx:<5} {_layer_name(idx):<45} {e:>14.4e} {lat:>14.4e} {edp:>14.4e}")
    print()


# ── High-level runs ────────────────────────────────────────────────────────────

def run_sanity_check():
    """Map only T0 (18,432-MAC sanity conv) to verify the pipeline end-to-end."""
    print("=" * 70)
    print("SANITY CHECK  --  T0: 3x3 conv  8x8x4 -> 8x8x8  (18,432 MACs)")
    print(f"Architecture : {ARCH_FILE}")
    print("=" * 70)

    # AccelForge maps the first einsum when max_layers / einsum filtering is used.
    # Adjust the call if your version of AccelForge uses a different API.
    mappings = run_mapper(WORKLOAD_FILE, arch_file=ARCH_FILE,
                          variable_overrides=VARIABLE_OVERRIDES)
    best    = mappings[min_edp_filter(mappings.data)]
    energy  = float(best.energy())
    latency = float(best.latency())
    edp     = energy * latency

    print(f"  energy  = {energy:.4e} pJ")
    print(f"  latency = {latency:.4e} cycles")
    print(f"  EDP     = {edp:.4e}")
    print()
    return energy, latency, edp


def run_full_model(max_layers=21):
    """
    Map all YOLO-World-S layers (T0–T20) one at a time and report a summary table.

    NOTE: AccelForge maps one workload file per call; here we call run_mapper
    once per layer index and collect results. Adjust if your AccelForge version
    supports batch layer mapping (e.g. via a max_layers argument).
    """
    print("=" * 70)
    print(f"YOLO-WORLD-S  --  layers T0-T{max_layers - 1}")
    print(f"Architecture : {ARCH_FILE}")
    print("=" * 70)

    rows = []
    for idx in range(max_layers):
        print(f"  Mapping {_layer_name(idx)} ...", end=" ", flush=True)
        try:
            e, lat, edp = _map_layer(idx)
            rows.append((idx, e, lat, edp))
            print(f"done  EDP={edp:.3e}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print()
    _print_table(rows)

    if rows:
        total_e   = sum(r[1] for r in rows)
        total_lat = sum(r[2] for r in rows)  # sequential latency lower bound
        worst     = max(rows, key=lambda r: r[3])
        print(f"Cumulative energy  : {total_e:.4e} pJ")
        print(f"Summed latency     : {total_lat:.4e} cycles")
        print(f"Most expensive EDP : {_layer_name(worst[0])}  EDP={worst[3]:.4e}")
    print()
    return rows


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map YOLO-World-S on your AccelForge arch.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sanity",     action="store_true",
                       help="Map T0 only (sanity check)")
    group.add_argument("--layer",      type=int, metavar="N",
                       help="Map a single layer by index (0-20)")
    group.add_argument("--max-layers", type=int, default=21, metavar="N",
                       help="Map layers T0 through T(N-1)  [default: 21]")
    args = parser.parse_args()

    if args.sanity:
        run_sanity_check()
    elif args.layer is not None:
        e, lat, edp = _map_layer(args.layer)
        print(f"T{args.layer}  {_layer_name(args.layer)}")
        print(f"  energy={e:.4e} pJ  latency={lat:.4e} cyc  EDP={edp:.4e}")
    else:
        run_sanity_check()
        run_full_model(max_layers=args.max_layers)
