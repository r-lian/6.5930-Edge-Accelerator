"""
mapper_script.py
Run the AccelForge mapper on YOLO-World-S workload layers for a custom edge accelerator.

Usage (from this milestone_1/ directory):
    python mapper_script.py

    # Map only the sanity-check layer (T0):
    python mapper_script.py --sanity

    # Map a specific layer by index:
    python mapper_script.py --layer 1

    # Map layers T0-T12 (backbone only):
    python mapper_script.py --max-layers 13

Usage (from a notebook):
    from milestone_1.mapper_script import run_mapper, evaluate_mapping, min_edp_filter
"""

import argparse
import accelforge as af
from accelforge.mapper import Metrics

# Update to your partner's architecture YAML when it is ready
ARCH_FILE = "../arch/ethos_u55.yaml"
WORKLOAD_FILE = "../workload/yolo_world.yaml"

VARIABLE_OVERRIDES: dict[str, int | float] = {
    # "ARRAY_ROWS": 256,
    # "ARRAY_COLS": 256,
    # "ADC_RESOLUTION": 6,
    # "BITS_PER_CELL": 2,
}

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


def min_edp_filter(df):
    return (df["energy"] * df["latency"]).idxmin()


def run_mapper(workload_file, arch_file=ARCH_FILE, variable_overrides=None):
    spec = af.Spec.from_yaml(arch_file, workload_file, variable_overrides=variable_overrides or {})
    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
    return spec.map_workload_to_arch()


def evaluate_mapping(workload_file, mapping_file, arch_file=ARCH_FILE, sparse_file=None, variable_overrides=None):
    files = [arch_file, workload_file, mapping_file]
    if sparse_file:
        files.append(sparse_file)
    spec = af.Spec.from_yaml(*files, variable_overrides=variable_overrides or {})
    result = spec.evaluate_mapping()
    return float(result.energy()), float(result.latency())


def _layer_name(idx):
    return LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"T{idx}"


def _map_layer(layer_idx, workload_file=WORKLOAD_FILE, arch_file=ARCH_FILE):
    mappings = run_mapper(workload_file, arch_file, variable_overrides=VARIABLE_OVERRIDES)
    best = mappings[min_edp_filter(mappings.data)]
    energy = float(best.energy())
    latency = float(best.latency())
    return energy, latency, energy * latency


def run_sanity_check():
    mappings = run_mapper(WORKLOAD_FILE, arch_file=ARCH_FILE, variable_overrides=VARIABLE_OVERRIDES)
    best = mappings[min_edp_filter(mappings.data)]
    energy = float(best.energy())
    latency = float(best.latency())
    edp = energy * latency
    print(f"energy={energy:.4e} pJ latency={latency:.4e} cyc EDP={edp:.4e}")
    return energy, latency, edp


def run_full_model(max_layers=21):
    rows = []
    for idx in range(max_layers):
        try:
            e, lat, edp = _map_layer(idx)
            rows.append((idx, e, lat, edp))
            print(f"{_layer_name(idx)} EDP={edp:.3e}")
        except Exception as exc:
            print(f"{_layer_name(idx)} FAILED: {exc}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map YOLO-World-S on your AccelForge arch")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sanity", action="store_true", help="Map T0 only")
    group.add_argument("--layer", type=int, metavar="N", help="Map a single layer by index")
    group.add_argument("--max-layers", type=int, default=21, metavar="N", help="Map layers T0..T(N-1)")
    args = parser.parse_args()

    if args.sanity:
        run_sanity_check()
    elif args.layer is not None:
        e, lat, edp = _map_layer(args.layer)
        print(f"T{args.layer} energy={e:.4e} latency={lat:.4e} EDP={edp:.4e}")
    else:
        run_sanity_check()
        run_full_model(max_layers=args.max_layers)

