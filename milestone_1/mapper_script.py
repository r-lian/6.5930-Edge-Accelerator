"""
mapper_script.py  —  Map YOLO-World-S layers onto the Ethos-U55-like architecture.

Run from the repo root (/home/workspace inside Docker):
    python -m milestone_1.mapper_script                     # sanity check + full model (T0–T20)
    python -m milestone_1.mapper_script --sanity            # T0 only (18,432 MACs)
    python -m milestone_1.mapper_script --layer 7           # single layer by index
    python -m milestone_1.mapper_script --max-layers 13     # backbone only (T0–T12)
    python -m milestone_1.mapper_script --sweep             # vary NUM_MACs + SRAM, compare EDP

Import from a notebook:
    from milestone_1.mapper_script import map_single_layer, run_sanity_check, run_workload
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from accelforge.mapper import Metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milestone_1.load_ethos_u55 import load_ethos_u55_spec

# ── Workload ──────────────────────────────────────────────────────────────────

WORKLOAD_YAML = str(_REPO_ROOT / "workload" / "yolo_world.yaml")

# BATCH_SIZE is the only Jinja variable in yolo_world.yaml.
WORKLOAD_JINJA = {"BATCH_SIZE": 1}

# ── Architecture baseline ─────────────────────────────────────────────────────
# These are passed directly to load_ethos_u55_spec as keyword arguments.
# Override any of them by passing keyword args to map_single_layer() or run_workload().

ARCH_DEFAULTS: dict = {
    "num_macs":                128,
    "system_preset":           "high_end_embedded",  # 4 GB/s SRAM, 0.5 GB/s FLASH
    "memory_mode":             "dedicated_384kb",     # 384 KB system SRAM
    "local_buffer_size_bytes": 32 * 1024,             # 32 KB scratchpad
    "clock_hz":                1.0e9,                 # 1 GHz
}

# ── Layer metadata ────────────────────────────────────────────────────────────

LAYER_NAMES = [
    "T0  sanity_check   3x3  8x8x4->8x8x8",
    "T1  backbone_stem  3x3s2  640x640x3->320x320x32",
    "T2  backbone_L1    3x3s2  320x320x32->160x160x64",
    "T3  backbone_C2f1  3x3  160x160x32->160x160x32",
    "T4  backbone_L2    3x3s2  160x160x64->80x80x128",
    "T5  backbone_C2f2a 3x3  80x80x64->80x80x64",
    "T6  backbone_C2f2b 3x3  80x80x64->80x80x64",
    "T7  backbone_L3    3x3s2  80x80x128->40x40x256",
    "T8  backbone_C2f3a 3x3  40x40x128->40x40x128",
    "T9  backbone_C2f3b 3x3  40x40x128->40x40x128",
    "T10 backbone_L4    3x3s2  40x40x256->20x20x512",
    "T11 backbone_C2f4  3x3  20x20x256->20x20x256",
    "T12 sppf_1x1       1x1  20x20x512->20x20x256",
    "T13 text_Q_proj    77x512->77x512",
    "T14 text_K_proj    77x512->77x512",
    "T15 text_V_proj    77x512->77x512",
    "T16 text_FFN_up    77x512->77x2048",
    "T17 text_FFN_down  77x2048->77x512",
    "T18 repvl_P5_1x1   1x1  20x20x512->20x20x512",
    "T19 det_head_P3    3x3  80x80x128->80x80x128",
    "T20 det_head_P4    3x3  40x40x256->40x40x256",
]

# Expected MACs per layer (used for pJ/MAC and sanity-checking).
EXPECTED_MACS = [
    18_432,          # T0  sanity
    88_473_600,      # T1  backbone stem
    471_859_200,     # T2  backbone L1
    235_929_600,     # T3  C2f_1
    471_859_200,     # T4  backbone L2
    235_929_600,     # T5  C2f_2a
    235_929_600,     # T6  C2f_2b
    471_859_200,     # T7  backbone L3
    235_929_600,     # T8  C2f_3a
    235_929_600,     # T9  C2f_3b
    471_859_200,     # T10 backbone L4
    235_929_600,     # T11 C2f_4
    52_428_800,      # T12 SPPF 1x1
    20_185_088,      # T13 text Q
    20_185_088,      # T14 text K
    20_185_088,      # T15 text V
    80_740_352,      # T16 FFN up
    80_740_352,      # T17 FFN down
    104_857_600,     # T18 repvl P5
    943_718_400,     # T19 head P3
    943_718_400,     # T20 head P4
]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class LayerResult:
    idx: int
    energy_j: float
    latency_cycles: float
    edp: float = field(init=False)
    component_energy: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.edp = self.energy_j * self.latency_cycles


# ── Core mapping function ─────────────────────────────────────────────────────

def map_single_layer(layer_idx: int, **arch_overrides) -> LayerResult:
    """
    Map one layer of yolo_world.yaml onto the Ethos-U55 architecture.

    AccelForge/Timeloop maps one einsum (loop nest) per spec call.
    We extract the target einsum into a temporary single-einsum YAML,
    then load it with load_ethos_u55_spec so all Jinja variables are filled.

    Parameters
    ----------
    layer_idx : int
        Index into the yolo_world.yaml einsums list (0 = T0 sanity check).
    **arch_overrides
        Any keyword accepted by load_ethos_u55_spec / build_ethos_u55_jinja_data,
        e.g. num_macs=256, memory_mode="dedicated_512kb".

    Returns
    -------
    LayerResult with energy (J), latency (cycles), EDP, and per-component energy.
    """
    # --- 1. Read and pre-process the workload YAML ---
    raw = Path(WORKLOAD_YAML).read_text()
    # Resolve the one workload-level Jinja variable before parsing YAML.
    raw = raw.replace("{{BATCH_SIZE}}", str(WORKLOAD_JINJA["BATCH_SIZE"]))
    doc = yaml.safe_load(raw)

    einsums = doc["workload"]["einsums"]
    if layer_idx >= len(einsums):
        raise IndexError(
            f"Layer index {layer_idx} is out of range "
            f"(workload has {len(einsums)} einsums, indices 0–{len(einsums)-1})."
        )

    # --- 2. Build a single-einsum workload document ---
    rank_sizes = doc["workload"].get("rank_sizes", {})
    # After BATCH_SIZE substitution the rank_sizes dict values are ints.
    # Drop any that are still strings (template vars we didn't resolve).
    rank_sizes = {k: v for k, v in rank_sizes.items() if isinstance(v, int)}

    single_doc = {
        "renames": doc.get("renames"),
        "workload": {
            "bits_per_value": doc["workload"]["bits_per_value"],
            "rank_sizes": rank_sizes,
            "einsums": [einsums[layer_idx]],
        },
    }

    # --- 3. Write to a temp file and map ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"yw_layer{layer_idx}_",
    ) as f:
        yaml.dump(single_doc, f, default_flow_style=False, allow_unicode=True)
        tmp_path = f.name

    try:
        overrides = {**ARCH_DEFAULTS, **arch_overrides}
        spec = load_ethos_u55_spec(
            workload_yaml=tmp_path,
            workload_jinja_parse={},   # BATCH_SIZE already resolved above
            **overrides,
        )
        spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
        mappings = spec.map_workload_to_arch()

        # Column names use "<SEP>" as separator (e.g. "Total<SEP>energy").
        # The mapper already Pareto-filters to the best mapping(s); take row 0.
        row = mappings.data.iloc[0]
        energy_j = float(row["Total<SEP>energy"])
        latency  = float(row["Total<SEP>latency"])

        # Per-component energy: find all columns matching "<component><SEP>energy"
        # that are not the "Total" rollup.
        comp_e: dict[str, float] = {}
        try:
            for col in mappings.data.columns:
                if col.endswith("<SEP>energy") and not col.startswith("Total"):
                    val = row[col]
                    if val == val and val != 0:   # skip NaN and zero
                        comp_e[col.split("<SEP>")[0]] = float(val)
        except Exception:
            pass

        return LayerResult(
            idx=layer_idx,
            energy_j=energy_j,
            latency_cycles=latency,
            component_energy=comp_e,
        )
    finally:
        os.unlink(tmp_path)


# ── Print helpers ─────────────────────────────────────────────────────────────

def _lname(idx: int) -> str:
    return LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"T{idx}"


def _print_layer_table(results: list[LayerResult]) -> None:
    hdr = (f"{'Idx':<5} {'Layer':<46} {'MACs':>12} "
           f"{'Energy (J)':>12} {'Latency(cyc)':>13} {'pJ/MAC':>8}")
    print(hdr)
    print("-" * len(hdr))

    total_e    = sum(r.energy_j for r in results)
    total_macs = 0

    for r in results:
        exp = EXPECTED_MACS[r.idx] if r.idx < len(EXPECTED_MACS) else 0
        pj  = (r.energy_j / exp * 1e12) if exp else float("nan")
        total_macs += exp
        print(
            f"{r.idx:<5} {_lname(r.idx):<46} {exp:>12,} "
            f"{r.energy_j:>12.3e} {r.latency_cycles:>13.3e} {pj:>8.3f}"
        )

    print("-" * len(hdr))
    avg_pj = (total_e / total_macs * 1e12) if total_macs else float("nan")
    print(
        f"{'TOTAL':<52} {total_macs:>12,} "
        f"{total_e:>12.3e} {'':>13} {avg_pj:>8.3f}"
    )
    print()


def _print_component_breakdown(results: list[LayerResult]) -> None:
    all_comp: dict[str, float] = {}
    total_e = 0.0
    for r in results:
        total_e += r.energy_j
        for comp, e in r.component_energy.items():
            all_comp[comp] = all_comp.get(comp, 0.0) + e

    if not all_comp:
        return
    print("Component energy breakdown (summed over all mapped layers):")
    for comp, e in sorted(all_comp.items(), key=lambda x: -x[1]):
        pct = 100.0 * e / total_e if total_e else 0.0
        print(f"  {comp:<36} {e:.3e} J  ({pct:.1f}%)")
    print()


# ── High-level experiment functions ──────────────────────────────────────────

def run_sanity_check(**arch_overrides) -> LayerResult:
    """Map T0 only (18,432-MAC tiny conv) to confirm the toolchain is working."""
    cfg = {**ARCH_DEFAULTS, **arch_overrides}
    print("=" * 70)
    print("SANITY CHECK  —  T0: 3x3 conv  8x8x4->8x8x8  (18,432 MACs)")
    print(f"  num_macs = {cfg['num_macs']}  |  "
          f"preset = {cfg['system_preset']}  |  sram = {cfg['memory_mode']}")
    print("=" * 70)

    r = map_single_layer(0, **arch_overrides)
    exp = EXPECTED_MACS[0]
    print(f"  energy    = {r.energy_j:.4e} J")
    print(f"  latency   = {r.latency_cycles:.4e} cycles")
    print(f"  EDP       = {r.edp:.4e}")
    print(f"  pJ/MAC    = {r.energy_j / exp * 1e12:.3f}")
    print()
    return r


def run_workload(max_layers: int = 21, **arch_overrides) -> list[LayerResult]:
    """
    Map layers T0 through T(max_layers-1) sequentially and print a summary table.

    Parameters
    ----------
    max_layers : int
        Number of layers to map (default 21 = T0–T20).
    **arch_overrides
        Architecture parameter overrides (num_macs, memory_mode, etc.).
    """
    cfg = {**ARCH_DEFAULTS, **arch_overrides}
    print("=" * 70)
    print(f"YOLO-WORLD-S  —  mapping T0–T{max_layers - 1}  ({max_layers} layers)")
    print(f"  num_macs = {cfg['num_macs']}  |  "
          f"preset = {cfg['system_preset']}  |  sram = {cfg['memory_mode']}")
    print("=" * 70)

    results: list[LayerResult] = []
    for idx in range(max_layers):
        print(f"  [{idx:>2}/{max_layers-1}] {_lname(idx)} ...", end=" ", flush=True)
        try:
            r = map_single_layer(idx, **arch_overrides)
            results.append(r)
            print(f"energy={r.energy_j:.3e} J  lat={r.latency_cycles:.3e} cyc")
        except Exception as exc:
            print(f"FAILED — {exc}")

    if results:
        print()
        _print_layer_table(results)
        _print_component_breakdown(results)

        worst_edp = max(results, key=lambda r: r.edp)
        worst_e   = max(results, key=lambda r: r.energy_j)
        worst_lat = max(results, key=lambda r: r.latency_cycles)
        total_e   = sum(r.energy_j for r in results)
        print(f"Most expensive (EDP)    : {_lname(worst_edp.idx)}  EDP={worst_edp.edp:.4e}")
        print(f"Most expensive (energy) : {_lname(worst_e.idx)}  "
              f"{worst_e.energy_j:.3e} J  ({100*worst_e.energy_j/total_e:.1f}% of total)")
        print(f"Slowest (latency)       : {_lname(worst_lat.idx)}  "
              f"{worst_lat.latency_cycles:.3e} cycles")
        print()

    return results


def run_architecture_sweep(probe_layer: int = 1) -> None:
    """
    Sweep NUM_MACs and SRAM size on a single representative layer.

    This generates the data for the 'bottleneck / improvement' section of
    your milestone writeup. T1 (backbone stem, 88 M MACs) is a good probe
    because it stresses both compute throughput and SRAM bandwidth.

    Parameters
    ----------
    probe_layer : int
        Layer index to sweep over (default 1 = T1 backbone stem).
    """
    sweep_configs = [
        {"num_macs":  32, "memory_mode": "dedicated_384kb"},
        {"num_macs":  64, "memory_mode": "dedicated_384kb"},
        {"num_macs": 128, "memory_mode": "dedicated_384kb"},   # baseline
        {"num_macs": 256, "memory_mode": "dedicated_384kb"},
        {"num_macs": 128, "memory_mode": "dedicated_512kb"},
        {"num_macs": 256, "memory_mode": "dedicated_512kb"},
    ]

    print("=" * 70)
    print(f"ARCHITECTURE SWEEP  —  probe: {_lname(probe_layer)}")
    print("=" * 70)

    rows = []
    for cfg in sweep_configs:
        label = f"{cfg['num_macs']:>3} MACs / {cfg['memory_mode']}"
        print(f"  {label} ...", end=" ", flush=True)
        try:
            r = map_single_layer(probe_layer, **cfg)
            rows.append((label, r.energy_j, r.latency_cycles, r.edp))
            print(f"EDP={r.edp:.3e}")
        except Exception as exc:
            print(f"FAILED — {exc}")
            rows.append((label, float("nan"), float("nan"), float("nan")))

    print()
    # Find baseline EDP for relative comparison.
    baseline = next((r[3] for r in rows if "128" in r[0] and "384kb" in r[0]), None)

    hdr = f"{'Config':<36} {'Energy (J)':>12} {'Latency(cyc)':>13} {'EDP':>14} {'vs baseline':>12}"
    print(hdr)
    print("-" * len(hdr))
    for label, e, lat, edp in rows:
        rel = f"{edp / baseline:>10.2f}x" if (baseline and edp == edp) else "        n/a"
        marker = "  ← baseline" if "128" in label and "384kb" in label else ""
        print(f"{label:<36} {e:>12.4e} {lat:>13.4e} {edp:>14.4e} {rel}{marker}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map YOLO-World-S layers onto the Ethos-U55-like architecture."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sanity",      action="store_true",
                       help="Map T0 only (sanity check)")
    group.add_argument("--layer",       type=int, metavar="N",
                       help="Map a single layer by index (0–20)")
    group.add_argument("--sweep",       action="store_true",
                       help="Sweep architecture parameters on T1 (backbone stem)")
    group.add_argument("--max-layers",  type=int, default=21, metavar="N",
                       help="Map T0 through T(N-1)  [default: 21]")
    args = parser.parse_args()

    if args.sanity:
        run_sanity_check()
    elif args.layer is not None:
        r = map_single_layer(args.layer)
        exp = EXPECTED_MACS[args.layer] if args.layer < len(EXPECTED_MACS) else 0
        print(f"\n{_lname(args.layer)}")
        print(f"  energy    = {r.energy_j:.4e} J")
        print(f"  latency   = {r.latency_cycles:.4e} cycles")
        print(f"  EDP       = {r.edp:.4e}")
        if exp:
            print(f"  pJ/MAC    = {r.energy_j / exp * 1e12:.3f}")
    elif args.sweep:
        run_architecture_sweep()
    else:
        run_sanity_check()
        run_workload(max_layers=args.max_layers)
