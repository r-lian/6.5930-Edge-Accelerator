"""
codesign.py  —  Milestone 2: Accelerator-workload co-design
=============================================================

Answers the four milestone questions:

  Q1. Under a fixed edge budget (area, power, throughput), what is the
      optimal allocation of resources across the accelerator?

  Q2. What parameters are explored and why?

  Q3. How does the optimal allocation change if the budget changes?

  Q4. How does changing the workload (lower resolution) improve efficiency?

Run from the repo root inside Docker:
    docker compose up -d
    docker compose exec labs bash
    
    then...

    python3 -m milestone_2.codesign --help
    python3 -m milestone_2.codesign --probe          # fast: map 5 representative probe layers (~2h)
    python3 -m milestone_2.codesign --full             # slow: map all 21 layers per config
    python3 -m milestone_2.codesign --workload        # compare 640px vs 320px YOLO-World
    python3 -m milestone_2.codesign --budget          # re-run sweep at tight / loose budgets

Suggested to run all at once: 
    python3 -m milestone_2.codesign --probe --budget --workload 
    
Make sure to caffeinate (outside of docker) so your computer doesn't sleep!

We use --probe to get faster results for design choices in ~30 min

--full is available to run overnight if you want whole-model numbers.
    NOTE: --full has led to OOM errors that we haven't fully debugged
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from accelforge.mapper import Metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milestone_1.load_ethos_u55 import (
    build_ethos_u55_jinja_data,
    load_ethos_u55_spec,
    DEFAULTS,
    MEMORY_MODES,
)

# ── Mapping cache ─────────────────────────────────────────────────────────────
# Persists layer mapping results to disk so reruns skip already-computed layers.

_CACHE_FILE = _REPO_ROOT / "milestone_2" / "mapping_cache.json"
_RESULTS_DIR = _REPO_ROOT / "milestone_2" / "results"

_cache: dict[str, dict] = {}

# ── Mapper settings ───────────────────────────────────────────────────────────
# Set from CLI args in __main__; defaults match prior behavior but cap at 128.
_PMAPPING_CAP: int = 128   # --cap N
_WORKERS: int = 4          # --workers N  (1 = serial)


def _cache_key(workload_yaml: str, layer_idx: int, num_macs: int,
               sram_kb: int, scratch_kb: int, system_preset: str) -> str:
    return f"{Path(workload_yaml).name}:T{layer_idx}:{num_macs}macs:{sram_kb}kb:{scratch_kb}scratch:{system_preset}"


def _load_cache() -> None:
    global _cache
    if _CACHE_FILE.exists():
        _cache = json.loads(_CACHE_FILE.read_text())
        print(f"[cache] Loaded {len(_cache)} cached mappings from {_CACHE_FILE.name}")
    else:
        _cache = {}


def _write_cache() -> None:
    _CACHE_FILE.write_text(json.dumps(_cache, indent=2))


def save_results(tag: str, data: object) -> Path:
    """Serialize experiment results to milestone_2/results/<tag>_<timestamp>.json."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _RESULTS_DIR / f"{tag}_{ts}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"[results] Saved to {out}")
    return out


# ── Workload files ────────────────────────────────────────────────────────────

WORKLOAD_640 = str(_REPO_ROOT / "workload" / "yolo_world.yaml")
WORKLOAD_320 = str(_REPO_ROOT / "workload" / "yolo_world_320.yaml")
WORKLOAD_JINJA = {"BATCH_SIZE": 1}

# Layer indices used as "probe" layers — one representative from each part of
# the network. Probing a subset makes sweeps feasible in ~20–30 min each.
#
#   T1  = backbone stem  (large spatial, 3-channel input)   — bandwidth-bound
#   T7  = backbone L3    (mid-size, 128 channels)           — balanced
#   T10 = backbone L4    (small spatial, deep channels)     — compute-bound
#   T16 = FFN up         (text encoder, no spatial dims)    — pure matmul
#   T19 = det head P3    (largest energy layer, 128ch 80²)
PROBE_LAYERS = [1, 7, 10, 16, 19]

# ── Budget definitions ────────────────────────────────────────────────────────
# Area is computed analytically from architecture parameters.
# Power is approximated as: dynamic_energy × throughput + leak_power.
# We define three budget tiers and find the best config within each.

BUDGETS = {
    "tight":    {"area_mm2": 0.20, "power_mw": 50,  "label": "Tight    (≤0.20 mm² / ≤50 mW)"},
    "baseline": {"area_mm2": 0.38, "power_mw": 100, "label": "Baseline (≤0.38 mm² / ≤100 mW) — ~Ethos-U55/128 area"},
    "relaxed":  {"area_mm2": 0.80, "power_mw": 500, "label": "Relaxed  (≤0.80 mm² / ≤500 mW)"},
}

# ── Architecture design space ─────────────────────────────────────────────────
# These are the axes we sweep. Each axis is motivated below.
#
#  num_macs        — compute throughput. Doubles with each step.
#                    Valid values: 32, 64, 128, 256.
#  sram_kb         — on-chip capacity. Larger SRAM = more tile reuse = less FLASH I/O.
#                    Uses system_sram_size_bytes (overrides memory_mode).
#  scratch_kb      — scratchpad (register file) near MAC. Bigger = more unroll.
#  system_preset   — SRAM + FLASH bandwidth. "deep_embedded" is power-efficient;
#                    "high_end_embedded" has 2.5× higher SRAM bandwidth.

DESIGN_SPACE = [
    # (num_macs, sram_kb, scratch_kb, system_preset)
    ( 32, 256,  16, "deep_embedded"),
    ( 32, 384,  32, "deep_embedded"),
    ( 64, 256,  16, "deep_embedded"),
    ( 64, 384,  32, "deep_embedded"),
    ( 64, 384,  32, "high_end_embedded"),
    (128, 256,  16, "deep_embedded"),
    (128, 384,  32, "deep_embedded"),       # reference: Ethos-U55/128, 384KB, low-BW preset
    (128, 384,  32, "high_end_embedded"),
    (128, 512,  64, "high_end_embedded"),
    (256, 384,  32, "high_end_embedded"),
    (256, 512,  64, "high_end_embedded"),
    (256, 512, 128, "high_end_embedded"),
]


# ── Area model ────────────────────────────────────────────────────────────────

def compute_area_mm2(num_macs: int, sram_kb: int, scratch_kb: int) -> float:
    """
    Estimate total chip area in mm² using AccelForge's area model.

    Components:
      - NPU core: fixed control logic + num_macs × mac_area
      - System SRAM: sram_area_per_bit × bits
      - Scratchpad: sram_area_per_bit × bits
    """
    jinja = build_ethos_u55_jinja_data(
        num_macs=num_macs,
        system_sram_size_bytes=sram_kb * 1024,
        local_buffer_size_bytes=scratch_kb * 1024,
    )
    return (
        jinja["NPU_CORE_AREA_M2"]
        + jinja["SYSTEM_SRAM_AREA_M2"]
        + jinja["LOCAL_BUFFER_AREA_M2"]
    ) * 1e6   # m² → mm²


# ── Mapping helpers ───────────────────────────────────────────────────────────

@dataclass
class LayerResult:
    layer_idx: int
    energy_j: float
    latency_s: float
    edp: float = field(init=False)

    def __post_init__(self):
        self.edp = self.energy_j * self.latency_s


@dataclass
class ArchResult:
    label: str
    num_macs: int
    sram_kb: int
    scratch_kb: int
    system_preset: str
    area_mm2: float
    layer_results: list[LayerResult] = field(default_factory=list)

    @property
    def total_energy_j(self) -> float:
        return sum(r.energy_j for r in self.layer_results)

    @property
    def total_latency_s(self) -> float:
        return sum(r.latency_s for r in self.layer_results)

    @property
    def total_edp(self) -> float:
        return self.total_energy_j * self.total_latency_s

    @property
    def avg_power_mw(self) -> float:
        """Average dynamic power across mapped layers: total_energy / total_latency in mW."""
        lat = self.total_latency_s
        return (self.total_energy_j / lat * 1000) if lat > 0 else float("inf")


def _do_mapping(
    workload_yaml: str,
    layer_idx: int,
    num_macs: int,
    sram_kb: int,
    scratch_kb: int,
    system_preset: str,
    pmapping_cap: int,
) -> LayerResult:
    """Execute AccelForge mapping for one layer. No cache interaction."""
    raw = Path(workload_yaml).read_text().replace("{{BATCH_SIZE}}", "1")
    doc = yaml.safe_load(raw)
    einsums = doc["workload"]["einsums"]

    rank_sizes = {}
    for k, v in doc["workload"].get("rank_sizes", {}).items():
        try:
            rank_sizes[k] = int(v)
        except (TypeError, ValueError):
            pass
    single_doc = {
        "renames": doc.get("renames"),
        "workload": {
            "bits_per_value": doc["workload"]["bits_per_value"],
            "rank_sizes": rank_sizes,
            "einsums": [einsums[layer_idx]],
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"cd_l{layer_idx}_"
    ) as f:
        yaml.dump(single_doc, f, default_flow_style=False, allow_unicode=True)
        tmp = f.name

    try:
        spec = load_ethos_u55_spec(
            workload_yaml=tmp,
            workload_jinja_parse={},
            num_macs=num_macs,
            system_sram_size_bytes=sram_kb * 1024,
            local_buffer_size_bytes=scratch_kb * 1024,
            system_preset=system_preset,
        )
        spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
        spec.mapper.max_pmapping_templates_per_einsum = pmapping_cap
        mappings = spec.map_workload_to_arch()
        row = mappings.data.iloc[0]
        return LayerResult(
            layer_idx=layer_idx,
            energy_j=float(row["Total<SEP>energy"]),
            latency_s=float(row["Total<SEP>latency"]),
        )
    finally:
        os.unlink(tmp)


def _map_layer(
    workload_yaml: str,
    layer_idx: int,
    num_macs: int,
    sram_kb: int,
    scratch_kb: int,
    system_preset: str,
) -> LayerResult:
    """Map one layer using cache. Calls _do_mapping on cache miss."""
    key = _cache_key(workload_yaml, layer_idx, num_macs, sram_kb, scratch_kb, system_preset)
    if key in _cache:
        c = _cache[key]
        return LayerResult(layer_idx=layer_idx, energy_j=c["energy_j"], latency_s=c["latency_s"])
    result = _do_mapping(workload_yaml, layer_idx, num_macs, sram_kb, scratch_kb, system_preset, _PMAPPING_CAP)
    _cache[key] = {"energy_j": result.energy_j, "latency_s": result.latency_s, "edp": result.edp}
    _write_cache()
    return result


def _map_config(
    workload_yaml: str,
    layers: list[int],
    num_macs: int,
    sram_kb: int,
    scratch_kb: int,
    system_preset: str,
    label: str,
) -> ArchResult:
    area = compute_area_mm2(num_macs, sram_kb, scratch_kb)
    result = ArchResult(
        label=label,
        num_macs=num_macs,
        sram_kb=sram_kb,
        scratch_kb=scratch_kb,
        system_preset=system_preset,
        area_mm2=area,
    )
    for idx in layers:
        print(f"    layer {idx} ...", end=" ", flush=True)
        try:
            lr = _map_layer(workload_yaml, idx, num_macs, sram_kb, scratch_kb, system_preset)
            result.layer_results.append(lr)
            print(f"E={lr.energy_j:.3e} J  lat={lr.latency_s:.3e} s")
        except Exception as exc:
            print(f"FAILED: {exc}")
    return result


def _map_config_worker(
    workload_yaml: str,
    layers: list,
    num_macs: int,
    sram_kb: int,
    scratch_kb: int,
    system_preset: str,
    label: str,
    pmapping_cap: int,
    cache_snapshot: dict,
) -> tuple:
    """ProcessPoolExecutor worker. Maps all layers for one config.
    Reads cache_snapshot for hits; returns (ArchResult, new_cache_entries)."""
    area = compute_area_mm2(num_macs, sram_kb, scratch_kb)
    result = ArchResult(
        label=label, num_macs=num_macs, sram_kb=sram_kb,
        scratch_kb=scratch_kb, system_preset=system_preset, area_mm2=area,
    )
    new_entries: dict = {}
    for idx in layers:
        key = _cache_key(workload_yaml, idx, num_macs, sram_kb, scratch_kb, system_preset)
        if key in cache_snapshot:
            c = cache_snapshot[key]
            lr = LayerResult(layer_idx=idx, energy_j=c["energy_j"], latency_s=c["latency_s"])
            print(f"  [{label}] T{idx} (cached) E={lr.energy_j:.3e} J  lat={lr.latency_s:.3e} s", flush=True)
        else:
            print(f"  [{label}] T{idx} mapping ...", flush=True)
            try:
                lr = _do_mapping(workload_yaml, idx, num_macs, sram_kb, scratch_kb, system_preset, pmapping_cap)
                new_entries[key] = {"energy_j": lr.energy_j, "latency_s": lr.latency_s, "edp": lr.edp}
                print(f"  [{label}] T{idx} done   E={lr.energy_j:.3e} J  lat={lr.latency_s:.3e} s", flush=True)
            except Exception as exc:
                print(f"  [{label}] T{idx} FAILED: {exc}", flush=True)
                continue
        result.layer_results.append(lr)
    return result, new_entries


def _run_configs_parallel(layers: list, configs: list) -> list:
    """Map a list of configs, optionally in parallel.

    Each config is a tuple: (workload_yaml, nmacs, sram_kb, scratch_kb, system_preset, label).
    Falls back to serial _map_config when _WORKERS == 1.
    Returns list[ArchResult] (order not guaranteed when parallel).
    """
    if _WORKERS == 1:
        results = []
        for (workload_yaml, nmacs, sram, scratch, preset, label) in configs:
            area = compute_area_mm2(nmacs, sram, scratch)
            print(f"\n  Config: {label}  area={area:.3f} mm²")
            results.append(_map_config(workload_yaml, layers, nmacs, sram, scratch, preset, label))
        return results

    print(f"\n  Dispatching {len(configs)} configs across {_WORKERS} workers ...")
    cache_snap = dict(_cache)
    results = []
    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        future_to_label = {
            pool.submit(
                _map_config_worker,
                workload_yaml, layers, nmacs, sram, scratch, preset, label, _PMAPPING_CAP, cache_snap,
            ): label
            for (workload_yaml, nmacs, sram, scratch, preset, label) in configs
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result, new_entries = future.result()
                _cache.update(new_entries)
                if new_entries:
                    _write_cache()
                results.append(result)
                print(f"  ✓ {label}  E={result.total_energy_j:.3e} J  lat={result.total_latency_s:.3e} s")
            except Exception as exc:
                print(f"  ✗ {label}  FAILED: {exc}")
    return results


# ── Serialization helpers ─────────────────────────────────────────────────────

def _arch_result_to_dict(r: ArchResult) -> dict:
    return {
        "label": r.label,
        "num_macs": r.num_macs,
        "sram_kb": r.sram_kb,
        "scratch_kb": r.scratch_kb,
        "system_preset": r.system_preset,
        "area_mm2": r.area_mm2,
        "total_energy_j": r.total_energy_j,
        "total_latency_s": r.total_latency_s,
        "total_edp": r.total_edp,
        "avg_power_mw": r.avg_power_mw,
        "layers": [
            {"layer_idx": lr.layer_idx, "energy_j": lr.energy_j,
             "latency_s": lr.latency_s, "edp": lr.edp}
            for lr in r.layer_results
        ],
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def _sep():
    print("-" * 100)


def _print_sweep_table(
    results: list[ArchResult],
    budget_mm2: Optional[float] = None,
    power_mw: Optional[float] = None,
):
    """Print a summary table sorted by EDP. Configs violating either budget are excluded."""
    hdr = (f"  {'Config':<42} {'Area(mm²)':>9} {'Pwr(mW)':>8} {'Energy(J)':>11} "
           f"{'Latency(s)':>11} {'EDP':>13} {'vs best':>8}")
    print(hdr)
    _sep()

    def _over(r: ArchResult) -> str | None:
        reasons = []
        if budget_mm2 is not None and r.area_mm2 > budget_mm2:
            reasons.append(f"area {r.area_mm2:.3f} > {budget_mm2:.2f} mm²")
        if power_mw is not None and r.avg_power_mw > power_mw:
            reasons.append(f"power {r.avg_power_mw:.1f} > {power_mw:.0f} mW")
        return ", ".join(reasons) if reasons else None

    valid = [r for r in results if _over(r) is None]
    over  = [(r, _over(r)) for r in results if _over(r) is not None]

    sorted_valid = sorted(valid, key=lambda r: r.total_edp)
    best_edp = sorted_valid[0].total_edp if sorted_valid else None

    for r in sorted_valid:
        rel = f"{r.total_edp / best_edp:.2f}x" if best_edp else "   —"
        marker = "  ★ BEST" if r == sorted_valid[0] else ""
        print(
            f"  {r.label:<42} {r.area_mm2:>9.3f} {r.avg_power_mw:>8.1f} {r.total_energy_j:>11.3e} "
            f"{r.total_latency_s:>11.3e} {r.total_edp:>13.3e} {rel:>8}{marker}"
        )

    if over:
        budget_str = " / ".join(filter(None, [
            f"{budget_mm2:.2f} mm²" if budget_mm2 else None,
            f"{power_mw:.0f} mW" if power_mw else None,
        ]))
        print(f"  [Over budget ({budget_str}) — excluded:]")
        for r, reason in sorted(over, key=lambda x: x[0].area_mm2):
            print(f"  {r.label:<42} {r.area_mm2:>9.3f} {r.avg_power_mw:>8.1f}  ({reason})")
    print()


def _print_layer_breakdown(results: list[ArchResult], layer_names: list[str]):
    """Per-layer comparison across configs (for the probe-layer experiment)."""
    if not results or not results[0].layer_results:
        return

    layer_indices = [lr.layer_idx for lr in results[0].layer_results]
    print(f"  {'Layer':<38}", end="")
    for r in results[:4]:   # cap at 4 configs for readability
        print(f"  {r.label[:16]:>16}", end="")
    print()
    _sep()

    for li in layer_indices:
        name = layer_names[li] if li < len(layer_names) else f"T{li}"
        print(f"  {name:<38}", end="")
        for r in results[:4]:
            lr = next((x for x in r.layer_results if x.layer_idx == li), None)
            val = f"{lr.energy_j:.2e}" if lr else "   —"
            print(f"  {val:>16}", end="")
        print()
    print()


# ── Experiment functions ──────────────────────────────────────────────────────

LAYER_NAMES = [
    "T0  sanity_check   3x3  8x8x4->8x8x8",
    "T1  backbone_stem  3x3s2  640²x3->320²x32",
    "T2  backbone_L1    3x3s2  320²x32->160²x64",
    "T3  backbone_C2f1  3x3  160²x32->160²x32",
    "T4  backbone_L2    3x3s2  160²x64->80²x128",
    "T5  backbone_C2f2a 3x3  80²x64->80²x64",
    "T6  backbone_C2f2b 3x3  80²x64->80²x64",
    "T7  backbone_L3    3x3s2  80²x128->40²x256",
    "T8  backbone_C2f3a 3x3  40²x128->40²x128",
    "T9  backbone_C2f3b 3x3  40²x128->40²x128",
    "T10 backbone_L4    3x3s2  40²x256->20²x512",
    "T11 backbone_C2f4  3x3  20²x256->20²x256",
    "T12 sppf_1x1       1x1  20²x512->20²x256",
    "T13 text_Q_proj    77x512->77x512",
    "T14 text_K_proj    77x512->77x512",
    "T15 text_V_proj    77x512->77x512",
    "T16 text_FFN_up    77x512->77x2048",
    "T17 text_FFN_down  77x2048->77x512",
    "T18 repvl_P5_1x1   1x1  20²x512->20²x512",
    "T19 det_head_P3    3x3  80²x128->80²x128",
    "T20 det_head_P4    3x3  40²x256->40²x256",
]

LAYER_NAMES_320 = [
    "T0  sanity_check   (unchanged)",
    "T1  backbone_stem  3x3s2  320²x3->160²x32",
    "T2  backbone_L1    3x3s2  160²x32->80²x64",
    "T3  backbone_C2f1  3x3  80²x32->80²x32",
    "T4  backbone_L2    3x3s2  80²x64->40²x128",
    "T5  backbone_C2f2a 3x3  40²x64->40²x64",
    "T6  backbone_C2f2b 3x3  40²x64->40²x64",
    "T7  backbone_L3    3x3s2  40²x128->20²x256",
    "T8  backbone_C2f3a 3x3  20²x128->20²x128",
    "T9  backbone_C2f3b 3x3  20²x128->20²x128",
    "T10 backbone_L4    3x3s2  20²x256->10²x512",
    "T11 backbone_C2f4  3x3  10²x256->10²x256",
    "T12 sppf_1x1       1x1  10²x512->10²x256",
    "T13-T18 (text + fusion, unchanged)",
    "", "", "", "", "",
    "T19 det_head_P3    3x3  40²x128->40²x128",
    "T20 det_head_P4    3x3  20²x256->20²x256",
]

# Expected MACs for each layer — 640px version
EXPECTED_MACS_640 = [
    18_432, 88_473_600, 471_859_200, 235_929_600, 471_859_200,
    235_929_600, 235_929_600, 471_859_200, 235_929_600, 235_929_600,
    471_859_200, 235_929_600, 52_428_800, 20_185_088, 20_185_088,
    20_185_088, 80_740_352, 80_740_352, 104_857_600, 943_718_400, 943_718_400,
]

# Expected MACs for each layer — 320px version (spatial layers ÷4, text unchanged)
EXPECTED_MACS_320 = [
    18_432,
    88_473_600 // 4,    # T1
    471_859_200 // 4,   # T2
    235_929_600 // 4,   # T3
    471_859_200 // 4,   # T4
    235_929_600 // 4,   # T5
    235_929_600 // 4,   # T6
    471_859_200 // 4,   # T7
    235_929_600 // 4,   # T8
    235_929_600 // 4,   # T9
    471_859_200 // 4,   # T10
    235_929_600 // 4,   # T11
    52_428_800 // 4,    # T12
    20_185_088,         # T13 text (unchanged)
    20_185_088,         # T14
    20_185_088,         # T15
    80_740_352,         # T16
    80_740_352,         # T17
    104_857_600 // 4,   # T18 fusion (spatial)
    943_718_400 // 4,   # T19
    943_718_400 // 4,   # T20
]


def generate_320px_workload():
    """
    Generate yolo_world_320.yaml by halving the spatial dimensions of all
    spatial layers in yolo_world.yaml. Text encoder layers are unchanged.

    Spatial layers: T1-T12, T18, T19, T20.
    Text layers:    T13-T17 (77-token sequence, no spatial dims — unchanged).
    """
    if Path(WORKLOAD_320).exists():
        print(f"  {WORKLOAD_320} already exists, skipping generation.")
        return

    raw = Path(WORKLOAD_640).read_text()
    doc = yaml.safe_load(raw.replace("{{BATCH_SIZE}}", "1"))

    TEXT_LAYERS = {13, 14, 15, 16, 17}  # no spatial dims, leave unchanged

    new_einsums = []
    for i, e in enumerate(doc["workload"]["einsums"]):
        if i in TEXT_LAYERS:
            new_einsums.append(e)
            continue

        # Halve all spatial bounds (p, q dimensions only).
        new_shape = []
        for constraint in e.get("iteration_space_shape", []):
            # constraints look like "0 <= p1 < 320"
            # Use rsplit on " < " to isolate the upper bound without
            # accidentally splitting on the "<" inside "<=".
            s = str(constraint)
            if "<= " in s and " < " in s:
                lhs, upper = s.rsplit(" < ", 1)
                dim_name = lhs.split("<= ", 1)[-1].strip()
                bound = int(upper.strip())
                # Halve spatial dimensions (p and q), leave others untouched
                if dim_name.startswith(("p", "q")):
                    bound = max(1, bound // 2)
                new_shape.append(f"0 <= {dim_name} < {bound}")
            else:
                new_shape.append(constraint)
        new_e = dict(e)
        new_e["iteration_space_shape"] = new_shape
        new_einsums.append(new_e)

    doc["workload"]["einsums"] = new_einsums

    # Leave N: 1 as a literal integer — _do_mapping does its own {{BATCH_SIZE}}
    # replacement. Re-inserting the Jinja template as a quoted YAML string would
    # cause yaml to parse it as str instead of int, silently dropping N from
    # rank_sizes and producing an unbounded batch dimension.
    out_str = yaml.dump(
        {"renames": doc.get("renames"), "workload": doc["workload"]},
        default_flow_style=False, allow_unicode=True,
    )

    Path(WORKLOAD_320).write_text(
        "# YOLO-World-S  320×320 input  (spatial dims halved from yolo_world.yaml)\n"
        "# Text encoder layers T13-T17 are unchanged.\n\n"
        + out_str
    )
    print(f"  Generated {WORKLOAD_320}")


def experiment_probe_sweep(
    budget_mm2: Optional[float] = None,
    power_mw: Optional[float] = None,
):
    """
    Q1 + Q2: Sweep all architecture configs on the 5 probe layers.
    budget_mm2 is applied pre-mapping (skips configs by area).
    power_mw is applied post-mapping (excludes configs from the results table).
    """
    bparts = []
    if budget_mm2: bparts.append(f"≤{budget_mm2:.2f} mm²")
    if power_mw:   bparts.append(f"≤{power_mw:.0f} mW")
    bstr = f" (budget: {', '.join(bparts)})" if bparts else ""
    print("=" * 100)
    print(f"EXPERIMENT 1 — Design space sweep on probe layers{bstr}")
    print(f"  Layers: {[LAYER_NAMES[i] for i in PROBE_LAYERS]}")
    print("=" * 100)

    configs_to_run = []
    for (nmacs, sram, scratch, preset) in DESIGN_SPACE:
        label = f"{nmacs:>3}MACs/{sram:>3}KB/{scratch:>2}KB-scratch/{preset[:4]}"
        area  = compute_area_mm2(nmacs, sram, scratch)
        if budget_mm2 and area > budget_mm2 * 1.01:
            print(f"  Skip {label}  (area={area:.3f} mm² > area budget)")
            continue
        configs_to_run.append((WORKLOAD_640, nmacs, sram, scratch, preset, label))

    all_results = _run_configs_parallel(PROBE_LAYERS, configs_to_run)

    print("\n--- PROBE LAYER SWEEP RESULTS ---")
    _print_sweep_table(all_results, budget_mm2, power_mw)
    _print_layer_breakdown(all_results, LAYER_NAMES)

    if not budget_mm2 and not power_mw:  # don't double-save from experiment_budget_sensitivity
        save_results("probe_sweep", [_arch_result_to_dict(r) for r in all_results])
    return all_results


def experiment_budget_sensitivity():
    """
    Q3: Run the probe sweep at three budget tiers and compare the optimal
    config at each tier. Shows how the optimal allocation shifts as budget changes.
    """
    print("=" * 100)
    print("EXPERIMENT 2 — Budget sensitivity analysis")
    print("  How does the optimal config change as area budget tightens / relaxes?")
    print("=" * 100)

    all_budget_results = {}
    for name, bdef in BUDGETS.items():
        budget_a = bdef["area_mm2"]
        budget_p = bdef["power_mw"]
        print(f"\n  ── Budget tier: {bdef['label']} ──")
        results = experiment_probe_sweep(budget_mm2=budget_a, power_mw=budget_p)
        # valid = configs that pass BOTH area and power constraints
        valid = [r for r in results
                 if r.area_mm2 <= budget_a and r.avg_power_mw <= budget_p]
        if valid:
            best = min(valid, key=lambda r: r.total_edp)
            print(f"  → Best within budget: {best.label}  "
                  f"EDP={best.total_edp:.3e}  area={best.area_mm2:.3f} mm²  "
                  f"power={best.avg_power_mw:.1f} mW")
            all_budget_results[name] = {
                "budget_mm2": budget_a,
                "budget_power_mw": budget_p,
                "best": _arch_result_to_dict(best),
                "all": [_arch_result_to_dict(r) for r in valid],
            }
        else:
            print(f"  → No configs fit within budget.")
    save_results("budget_sensitivity", all_budget_results)


def _print_workload_comparison(
    r640: ArchResult,
    r320: ArchResult,
    layers_to_map: list[int],
):
    """Print per-layer 640px vs 320px comparison table with latency and energy."""
    print("\n  Per-layer comparison (640px vs 320px):")
    hdr = (f"  {'Layer':<38} {'MACs-640':>10} {'MACs-320':>10}"
           f" {'Lat-640(s)':>11} {'Lat-320(s)':>11} {'lat-spdup':>9}"
           f" {'E-640(J)':>11} {'E-320(J)':>11} {'e-spdup':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for li in layers_to_map:
        name  = LAYER_NAMES[li] if li < len(LAYER_NAMES) else f"T{li}"
        lr640 = next((x for x in r640.layer_results if x.layer_idx == li), None)
        lr320 = next((x for x in r320.layer_results if x.layer_idx == li), None)
        m640  = EXPECTED_MACS_640[li] if li < len(EXPECTED_MACS_640) else 0
        m320  = EXPECTED_MACS_320[li] if li < len(EXPECTED_MACS_320) else 0
        l640  = f"{lr640.latency_s:.3e}"  if lr640 else "  —"
        l320  = f"{lr320.latency_s:.3e}"  if lr320 else "  —"
        lsp   = f"{lr640.latency_s / lr320.latency_s:.2f}x" if (lr640 and lr320) else "  —"
        e640  = f"{lr640.energy_j:.3e}"   if lr640 else "  —"
        e320  = f"{lr320.energy_j:.3e}"   if lr320 else "  —"
        esp   = f"{lr640.energy_j / lr320.energy_j:.2f}x" if (lr640 and lr320) else "  —"
        print(f"  {name:<38} {m640:>10,} {m320:>10,}"
              f" {l640:>11} {l320:>11} {lsp:>9}"
              f" {e640:>11} {e320:>11} {esp:>8}")

    total_e640 = r640.total_energy_j
    total_e320 = r320.total_energy_j
    total_l640 = r640.total_latency_s
    total_l320 = r320.total_latency_s
    lsp_total = f"{total_l640/total_l320:.2f}x" if total_l320 > 0 else "  —"
    esp_total = f"{total_e640/total_e320:.2f}x" if total_e320 > 0 else "  —"
    lat_pct   = f"{100*(1 - total_l320/total_l640):.1f}%" if total_l640 > 0 and total_l320 > 0 else "  —"
    e_pct     = f"{100*(1 - total_e320/total_e640):.1f}%" if total_e640 > 0 and total_e320 > 0 else "  —"
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL (probe layers)':<38} {'':>10} {'':>10}"
          f" {total_l640:>11.3e} {total_l320:>11.3e}"
          f" {lsp_total:>9}"
          f" {total_e640:>11.3e} {total_e320:>11.3e}"
          f" {esp_total:>8}")
    print(f"\n  Latency reduction: {lat_pct}"
          f"   Energy reduction: {e_pct}")
    print(f"  Note: text encoder layers (T13-T17) are unchanged between resolutions.")
    print(f"  As spatial layers shrink, the text encoder becomes a larger fraction of total cost.")
    print()


# Representative hardware configs for workload comparison across budget tiers.
# Tight = smallest feasible config; Baseline = Ethos-U55/128; Relaxed = largest config.
WORKLOAD_COMPARISON_CONFIGS = [
    # (label,              nmacs, sram_kb, scratch_kb, preset)
    ("Tight   32M/256KB",     32,     256,         16, "deep_embedded"),
    ("Baseline 128M/384KB",  128,     384,         32, "high_end_embedded"),
    ("Relaxed 256M/512KB",   256,     512,        128, "high_end_embedded"),
]


def experiment_workload_comparison():
    """
    Q4: Compare 640px vs 320px YOLO-World-S across tight / baseline / relaxed
    hardware configs. Shows how much workload resolution reduction helps when
    hardware is most and least constrained.
    """
    print("=" * 100)
    print("EXPERIMENT 3 — Workload co-design: 640px vs 320px input resolution")
    print("  Runs on tight / baseline / relaxed hardware configs (Goal: most + least permissive)")
    print("=" * 100)

    generate_320px_workload()
    layers_to_map = PROBE_LAYERS

    # Build flat list of all (hw_config × resolution) combos and map in parallel.
    all_combos = []
    for hw_label, nmacs, sram, scratch, preset in WORKLOAD_COMPARISON_CONFIGS:
        area = compute_area_mm2(nmacs, sram, scratch)
        print(f"  Queuing: {hw_label}  ({nmacs}MACs/{sram}KB/{scratch}KB-scratch)  area={area:.3f} mm²")
        all_combos.append((WORKLOAD_640, nmacs, sram, scratch, preset, f"{hw_label} 640px"))
        all_combos.append((WORKLOAD_320, nmacs, sram, scratch, preset, f"{hw_label} 320px"))

    all_r = _run_configs_parallel(layers_to_map, all_combos)
    r_by_label = {r.label: r for r in all_r}

    for hw_label, nmacs, sram, scratch, preset in WORKLOAD_COMPARISON_CONFIGS:
        area = compute_area_mm2(nmacs, sram, scratch)
        print(f"\n{'─'*100}")
        print(f"  Config: {hw_label}  ({nmacs} MACs / {sram} KB SRAM / {scratch} KB scratch"
              f" / {preset})  area={area:.3f} mm²")
        print(f"{'─'*100}")
        r640 = r_by_label.get(f"{hw_label} 640px")
        r320 = r_by_label.get(f"{hw_label} 320px")
        if r640 and r320:
            _print_workload_comparison(r640, r320, layers_to_map)
            save_results(f"workload_{hw_label.split()[0].lower()}", {
                "hw_config": {"label": hw_label, "num_macs": nmacs, "sram_kb": sram,
                              "scratch_kb": scratch, "system_preset": preset, "area_mm2": area},
                "r640": _arch_result_to_dict(r640),
                "r320": _arch_result_to_dict(r320),
            })
        else:
            print(f"  Incomplete results for {hw_label} — skipping comparison.")


def experiment_full_model(workload_yaml: str = WORKLOAD_640):
    """
    Map all 21 layers on the baseline architecture (use for overnight runs).
    """
    layers = list(range(21))
    nmacs, sram, scratch, preset = 128, 384, 32, "high_end_embedded"
    label = f"128MACs/384KB/32KB-scratch/high (baseline)"
    wname = "640px" if workload_yaml == WORKLOAD_640 else "320px"

    print("=" * 100)
    print(f"FULL MODEL RUN — {wname} — all 21 layers — {label}")
    print("=" * 100)

    r = _map_config(workload_yaml, layers, nmacs, sram, scratch, preset, label)

    macs_list = EXPECTED_MACS_640 if workload_yaml == WORKLOAD_640 else EXPECTED_MACS_320
    names     = LAYER_NAMES if workload_yaml == WORKLOAD_640 else LAYER_NAMES_320

    hdr = f"  {'Idx':<4} {'Layer':<44} {'MACs':>12} {'Energy(J)':>11} {'Lat(s)':>11} {'pJ/MAC':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    total_e = 0.0
    total_m = 0
    for lr in r.layer_results:
        name = names[lr.layer_idx] if lr.layer_idx < len(names) else f"T{lr.layer_idx}"
        m    = macs_list[lr.layer_idx] if lr.layer_idx < len(macs_list) else 0
        pj   = lr.energy_j / m * 1e12 if m else float("nan")
        total_e += lr.energy_j
        total_m += m
        print(f"  {lr.layer_idx:<4} {name:<44} {m:>12,} {lr.energy_j:>11.3e} {lr.latency_s:>11.3e} {pj:>8.2f}")

    print("  " + "-" * (len(hdr) - 2))
    avg_pj = total_e / total_m * 1e12 if total_m else float("nan")
    print(f"  {'TOTAL':<49} {total_m:>12,} {total_e:>11.3e} {'':>11} {avg_pj:>8.2f}")
    print()
    save_results(f"full_model_{wname}", _arch_result_to_dict(r))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Milestone 2: accelerator-workload co-design experiments."
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Sweep all arch configs on 5 probe layers (fastest, ~2h)"
    )
    parser.add_argument(
        "--budget", action="store_true",
        help="Re-run probe sweep at tight / baseline / relaxed area budgets"
    )
    parser.add_argument(
        "--workload", action="store_true",
        help="Compare 640px vs 320px YOLO-World on tight / baseline / relaxed arch configs"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Map all 21 layers on baseline arch (very slow, use overnight)"
    )
    parser.add_argument(
        "--full-320", action="store_true",
        help="Map all 21 layers of 320px workload on baseline arch"
    )
    parser.add_argument(
        "--cap", type=int, default=128, metavar="N",
        help="Max pmapping templates per einsum (default: 128; lower=faster, higher=more optimal)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Parallel worker processes (default: 4; use 1 for serial)"
    )
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()

    _PMAPPING_CAP = args.cap
    _WORKERS = args.workers

    _load_cache()
    if args.probe:
        experiment_probe_sweep()
    if args.budget:
        experiment_budget_sensitivity()
    if args.workload:
        experiment_workload_comparison()
    if args.full:
        experiment_full_model(WORKLOAD_640)
    if args.full_320:
        generate_320px_workload()
        experiment_full_model(WORKLOAD_320)
