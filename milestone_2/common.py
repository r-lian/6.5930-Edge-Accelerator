# @Time    : 2026-04-27 15:23
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : common.py

"""
common.py — shared infrastructure for proposal/exp_*.py scripts.

Extracted from milestone_2/codesign.py so the three final-project experiment
scripts can stay thin (each one only owns its design space + analysis logic).

Provides:
  - Cache (persistent disk-backed mapping cache shared with codesign.py)
  - Mapping core: _do_mapping, _map_layer, _map_config, parallel runner
  - Result types: LayerResult, ArchResult
  - Area model: compute_area_mm2
  - Workload generators (320px, 448px, robotics) and metadata
  - Pareto-front utility for Exp 1
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milestone_1.load_ethos_u55 import (  # noqa: E402
    build_ethos_u55_jinja_data,
    load_ethos_u55_spec,
)

# ── Cache ─────────────────────────────────────────────────────────────────────

_CACHE_FILE = _REPO_ROOT / "milestone_2" / "mapping_cache.json"
_RESULTS_DIR = _REPO_ROOT / "milestone_2" / "results"
_cache: dict[str, dict] = {}

# Mapper settings (set by experiment scripts via configure()).
_PMAPPING_CAP: int = 128
_WORKERS: int = 4


def configure(pmapping_cap: int = 128, workers: int = 4) -> None:
    """Set parallelism + pmapping cap for the current process."""
    global _PMAPPING_CAP, _WORKERS
    _PMAPPING_CAP = pmapping_cap
    _WORKERS = workers


def _cache_key(workload_yaml: str, layer_idx: int, num_macs: int,
               sram_kb: int, scratch_kb: int, system_preset: str) -> str:
    return f"{Path(workload_yaml).name}:T{layer_idx}:{num_macs}macs:{sram_kb}kb:{scratch_kb}scratch:{system_preset}"


def load_cache() -> None:
    global _cache
    if _CACHE_FILE.exists():
        _cache = json.loads(_CACHE_FILE.read_text())
        print(f"[cache] Loaded {len(_cache)} cached mappings from {_CACHE_FILE.name}")
    else:
        _cache = {}


def _write_cache() -> None:
    _CACHE_FILE.write_text(json.dumps(_cache, indent=2))


def save_results(tag: str, data: object) -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _RESULTS_DIR / f"{tag}_{ts}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"[results] Saved to {out}")
    return out


# ── Workload paths and generators ─────────────────────────────────────────────

WORKLOAD_640 = str(_REPO_ROOT / "workload" / "yolo_world.yaml")
WORKLOAD_320 = str(_REPO_ROOT / "workload" / "yolo_world_320.yaml")
WORKLOAD_448 = str(_REPO_ROOT / "workload" / "yolo_world_448.yaml")
WORKLOAD_ROBOTICS = str(_REPO_ROOT / "workload" / "yolo_world_robotics.yaml")
WORKLOAD_JINJA = {"BATCH_SIZE": 1}

PROBE_LAYERS = [1, 7, 10, 16, 19]

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

# Robotics-head einsums appended after T20.
ROBOTICS_HEAD_LAYER_NAMES = [
    "T21 head_pool_proj  matmul  1x512->1x512",
    "T22 head_hidden     matmul  1x512->1x256",
    "T23 head_action     matmul  1x256->1x64",
]
LAYER_NAMES_ROBOTICS = LAYER_NAMES + ROBOTICS_HEAD_LAYER_NAMES

EXPECTED_MACS_640 = [
    18_432, 88_473_600, 471_859_200, 235_929_600, 471_859_200,
    235_929_600, 235_929_600, 471_859_200, 235_929_600, 235_929_600,
    471_859_200, 235_929_600, 52_428_800, 20_185_088, 20_185_088,
    20_185_088, 80_740_352, 80_740_352, 104_857_600, 943_718_400, 943_718_400,
]

EXPECTED_MACS_320 = [
    18_432,
    88_473_600 // 4, 471_859_200 // 4, 235_929_600 // 4, 471_859_200 // 4,
    235_929_600 // 4, 235_929_600 // 4, 471_859_200 // 4, 235_929_600 // 4,
    235_929_600 // 4, 471_859_200 // 4, 235_929_600 // 4, 52_428_800 // 4,
    20_185_088, 20_185_088, 20_185_088, 80_740_352, 80_740_352,
    104_857_600 // 4, 943_718_400 // 4, 943_718_400 // 4,
]

# 448px = (448/640)² × 640px MACs ≈ 0.49× spatial. Spatial layers (T1-T12, T18-T20)
# scale; text layers (T13-T17) and sanity (T0) are unchanged.
def _scaled_448_macs() -> list[int]:
    # We compute (P/640)² × original = (448/640)² ≈ 0.49 for spatial layers.
    # Stem T1 input=640→output=320 with stride-2; at 448 input → output=224. P scales 224/320 = 0.7
    # → MACs scale (0.7)² = 0.49.
    factor_num, factor_den = 448 * 448, 640 * 640  # = 0.49
    spatial = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20}
    out: list[int] = []
    for i, m in enumerate(EXPECTED_MACS_640):
        if i in spatial:
            out.append(m * factor_num // factor_den)
        else:
            out.append(m)
    return out

EXPECTED_MACS_448 = _scaled_448_macs()

# Robotics head MACs (matmul of [K,M] with batch=1).
EXPECTED_MACS_ROBOTICS_HEAD = [
    1 * 512 * 512,  # T21
    1 * 512 * 256,  # T22
    1 * 256 * 64,   # T23
]
EXPECTED_MACS_ROBOTICS = EXPECTED_MACS_640 + EXPECTED_MACS_ROBOTICS_HEAD


def generate_320px_workload() -> None:
    """Generate yolo_world_320.yaml by halving spatial dims of T1-T12,T18-T20."""
    if Path(WORKLOAD_320).exists():
        return
    _generate_resolution_workload(WORKLOAD_320, src_resolution=640, dst_resolution=320)


def generate_448px_workload() -> None:
    """Generate yolo_world_448.yaml by scaling spatial dims of T1-T12,T18-T20 from 640 to 448."""
    if Path(WORKLOAD_448).exists():
        return
    _generate_resolution_workload(WORKLOAD_448, src_resolution=640, dst_resolution=448)


def _generate_resolution_workload(out_path: str, src_resolution: int, dst_resolution: int) -> None:
    """Create a new workload YAML by rescaling spatial p/q bounds.

    Spatial layers are T1-T12 and T18-T20 (text encoder T13-T17 and sanity T0
    are unchanged). For each spatial bound `0 <= dim < N`, rescale:
        N_new = max(1, N * dst_resolution // src_resolution)
    Only ranks named starting with 'p' or 'q' are rescaled.
    """
    raw = Path(WORKLOAD_640).read_text()
    doc = yaml.safe_load(raw.replace("{{BATCH_SIZE}}", "1"))

    TEXT_LAYERS = {13, 14, 15, 16, 17}
    new_einsums = []
    for i, e in enumerate(doc["workload"]["einsums"]):
        if i in TEXT_LAYERS or i == 0:
            new_einsums.append(e)
            continue
        new_shape = []
        for constraint in e.get("iteration_space_shape", []):
            s = str(constraint)
            if "<= " in s and " < " in s:
                lhs, upper = s.rsplit(" < ", 1)
                dim_name = lhs.split("<= ", 1)[-1].strip()
                bound = int(upper.strip())
                if dim_name.startswith(("p", "q")):
                    bound = max(1, bound * dst_resolution // src_resolution)
                new_shape.append(f"0 <= {dim_name} < {bound}")
            else:
                new_shape.append(constraint)
        new_e = dict(e)
        new_e["iteration_space_shape"] = new_shape
        new_einsums.append(new_e)

    doc["workload"]["einsums"] = new_einsums
    out_str = yaml.dump(
        {"renames": doc.get("renames"), "workload": doc["workload"]},
        default_flow_style=False, allow_unicode=True,
    )
    Path(out_path).write_text(
        f"# YOLO-World-S {dst_resolution}x{dst_resolution} input "
        f"(spatial dims rescaled from {src_resolution}x{src_resolution}).\n"
        f"# Text encoder layers T13-T17 and sanity T0 are unchanged.\n\n"
        + out_str
    )
    print(f"  Generated {out_path}")


def generate_robotics_workload() -> None:
    """Generate yolo_world_robotics.yaml = yolo_world.yaml + T21-T23 head einsums.

    Action-head MLP appended after T20:
      T21: pooled visual features [1, 512] → [1, 512]   (262 144 MACs)
      T22: hidden                 [1, 512] → [1, 256]   (131 072 MACs)
      T23: action head            [1, 256] → [1, 64]    ( 16 384 MACs)
    Total head ≈ 410 K MACs (< 0.02 % of full-model MACs).
    """
    if Path(WORKLOAD_ROBOTICS).exists():
        return
    raw = Path(WORKLOAD_640).read_text()
    doc = yaml.safe_load(raw.replace("{{BATCH_SIZE}}", "1"))

    # Build T21-T23 in the same matmul style as T13-T17 (sequence-position p, batch n,
    # output channel m, input channel c). Sequence position is fixed to 1 (single
    # pooled vector — no spatial / sequence dim).
    head_einsums = []
    for tnum, (k_in, m_out) in enumerate(((512, 512), (512, 256), (256, 64)), start=21):
        head_einsums.append({
            "einsum": f"T{tnum}[p{tnum}, n, m{tnum}] = I{tnum}[n, c{tnum}, p{tnum}] * W{tnum}[c{tnum}, m{tnum}]",
            "iteration_space_shape": [
                f"0 <= p{tnum} < 1",
                f"0 <= c{tnum} < {k_in}",
                f"0 <= m{tnum} < {m_out}",
            ],
            "renames": {"output": f"T{tnum}", "input": f"I{tnum}", "weight": f"W{tnum}"},
        })
    doc["workload"]["einsums"] = list(doc["workload"]["einsums"]) + head_einsums

    out_str = yaml.dump(
        {"renames": doc.get("renames"), "workload": doc["workload"]},
        default_flow_style=False, allow_unicode=True,
    )
    Path(WORKLOAD_ROBOTICS).write_text(
        "# YOLO-World-S + robotics head (T21-T23 action-decoder MLP).\n"
        "# Head appended verbatim to the 21-layer perception stack.\n\n"
        + out_str
    )
    print(f"  Generated {WORKLOAD_ROBOTICS}")


# ── Result types ──────────────────────────────────────────────────────────────

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
        lat = self.total_latency_s
        return (self.total_energy_j / lat * 1000) if lat > 0 else float("inf")


def arch_result_to_dict(r: ArchResult) -> dict:
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


# ── Area model ────────────────────────────────────────────────────────────────

def compute_area_mm2(num_macs: int, sram_kb: int, scratch_kb: int) -> float:
    jinja = build_ethos_u55_jinja_data(
        num_macs=num_macs,
        system_sram_size_bytes=sram_kb * 1024,
        local_buffer_size_bytes=scratch_kb * 1024,
    )
    return (
        jinja["NPU_CORE_AREA_M2"]
        + jinja["SYSTEM_SRAM_AREA_M2"]
        + jinja["LOCAL_BUFFER_AREA_M2"]
    ) * 1e6


# ── Mapping core ──────────────────────────────────────────────────────────────

def _do_mapping(
    workload_yaml: str,
    layer_idx: int,
    num_macs: int,
    sram_kb: int,
    scratch_kb: int,
    system_preset: str,
    pmapping_cap: int,
) -> LayerResult:
    """Execute one mapping for one layer. No cache interaction."""
    from accelforge.mapper import Metrics  # noqa: E402

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
        mode="w", suffix=".yaml", delete=False, prefix=f"cm_l{layer_idx}_"
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


def _map_layer(workload_yaml, layer_idx, num_macs, sram_kb, scratch_kb, system_preset) -> LayerResult:
    key = _cache_key(workload_yaml, layer_idx, num_macs, sram_kb, scratch_kb, system_preset)
    if key in _cache:
        c = _cache[key]
        return LayerResult(layer_idx=layer_idx, energy_j=c["energy_j"], latency_s=c["latency_s"])
    result = _do_mapping(workload_yaml, layer_idx, num_macs, sram_kb, scratch_kb, system_preset, _PMAPPING_CAP)
    _cache[key] = {"energy_j": result.energy_j, "latency_s": result.latency_s, "edp": result.edp}
    _write_cache()
    return result


def _map_config(workload_yaml, layers, num_macs, sram_kb, scratch_kb, system_preset, label) -> ArchResult:
    area = compute_area_mm2(num_macs, sram_kb, scratch_kb)
    result = ArchResult(label=label, num_macs=num_macs, sram_kb=sram_kb,
                        scratch_kb=scratch_kb, system_preset=system_preset, area_mm2=area)
    for idx in layers:
        print(f"    layer {idx} ...", end=" ", flush=True)
        try:
            lr = _map_layer(workload_yaml, idx, num_macs, sram_kb, scratch_kb, system_preset)
            result.layer_results.append(lr)
            print(f"E={lr.energy_j:.3e} J  lat={lr.latency_s:.3e} s")
        except Exception as exc:
            print(f"FAILED: {exc}")
    return result


def _map_config_worker(workload_yaml, layers, num_macs, sram_kb, scratch_kb, system_preset,
                       label, pmapping_cap, cache_snapshot) -> tuple:
    area = compute_area_mm2(num_macs, sram_kb, scratch_kb)
    result = ArchResult(label=label, num_macs=num_macs, sram_kb=sram_kb,
                        scratch_kb=scratch_kb, system_preset=system_preset, area_mm2=area)
    new_entries: dict = {}
    for idx in layers:
        key = _cache_key(workload_yaml, idx, num_macs, sram_kb, scratch_kb, system_preset)
        if key in cache_snapshot:
            c = cache_snapshot[key]
            lr = LayerResult(layer_idx=idx, energy_j=c["energy_j"], latency_s=c["latency_s"])
            print(f"  [{label}] T{idx} (cached) E={lr.energy_j:.3e}  lat={lr.latency_s:.3e}", flush=True)
        else:
            print(f"  [{label}] T{idx} mapping ...", flush=True)
            try:
                lr = _do_mapping(workload_yaml, idx, num_macs, sram_kb, scratch_kb, system_preset, pmapping_cap)
                new_entries[key] = {"energy_j": lr.energy_j, "latency_s": lr.latency_s, "edp": lr.edp}
                print(f"  [{label}] T{idx} done   E={lr.energy_j:.3e}  lat={lr.latency_s:.3e}", flush=True)
            except Exception as exc:
                print(f"  [{label}] T{idx} FAILED: {exc}", flush=True)
                continue
        result.layer_results.append(lr)
    return result, new_entries


def run_configs(workload_yaml: str, layers: list[int], configs: list) -> list[ArchResult]:
    """Map a list of (nmacs, sram_kb, scratch_kb, preset, label) configs.

    Honors module-level _WORKERS / _PMAPPING_CAP set via configure().
    """
    if _WORKERS == 1:
        results = []
        for (nmacs, sram, scratch, preset, label) in configs:
            area = compute_area_mm2(nmacs, sram, scratch)
            print(f"\n  Config: {label}  area={area:.3f} mm²")
            results.append(_map_config(workload_yaml, layers, nmacs, sram, scratch, preset, label))
        return results

    print(f"\n  Dispatching {len(configs)} configs across {_WORKERS} workers ...")
    cache_snap = dict(_cache)
    results: list[ArchResult] = []
    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        future_to_label = {
            pool.submit(
                _map_config_worker, workload_yaml, layers, nmacs, sram, scratch, preset,
                label, _PMAPPING_CAP, cache_snap,
            ): label
            for (nmacs, sram, scratch, preset, label) in configs
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result, new_entries = future.result()
                _cache.update(new_entries)
                if new_entries:
                    _write_cache()
                results.append(result)
                print(f"  ✓ {label}  E={result.total_energy_j:.3e}  lat={result.total_latency_s:.3e}")
            except Exception as exc:
                print(f"  ✗ {label}  FAILED: {exc}")
    return results


# ── Pareto frontier ───────────────────────────────────────────────────────────

def pareto_front(
    points: Iterable,
    x_key: Callable,
    y_key: Callable,
) -> list:
    """Return the subset that's Pareto-optimal in (x, y) — minimize both axes.

    A point is dominated if some other point has x' <= x AND y' <= y AND at
    least one strict inequality.
    """
    pts = list(points)
    front = []
    for p in pts:
        x, y = x_key(p), y_key(p)
        dominated = False
        for q in pts:
            if q is p:
                continue
            qx, qy = x_key(q), y_key(q)
            if qx <= x and qy <= y and (qx < x or qy < y):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=x_key)


# ── Print helpers ─────────────────────────────────────────────────────────────

def sep(width: int = 100) -> None:
    print("-" * width)


def print_sweep_table(
    results: list[ArchResult],
    budget_mm2: Optional[float] = None,
    power_mw: Optional[float] = None,
) -> None:
    hdr = (f"  {'Config':<42} {'Area(mm²)':>9} {'Pwr(mW)':>8} {'Energy(J)':>11} "
           f"{'Latency(s)':>11} {'EDP':>13} {'vs best':>8}")
    print(hdr)
    sep()

    def _over(r: ArchResult) -> Optional[str]:
        reasons = []
        if budget_mm2 is not None and r.area_mm2 > budget_mm2:
            reasons.append(f"area {r.area_mm2:.3f} > {budget_mm2:.2f} mm²")
        if power_mw is not None and r.avg_power_mw > power_mw:
            reasons.append(f"power {r.avg_power_mw:.1f} > {power_mw:.0f} mW")
        return ", ".join(reasons) if reasons else None

    valid = [r for r in results if _over(r) is None]
    over = [(r, _over(r)) for r in results if _over(r) is not None]
    sorted_valid = sorted(valid, key=lambda r: r.total_edp)
    best_edp = sorted_valid[0].total_edp if sorted_valid else None

    for r in sorted_valid:
        rel = f"{r.total_edp / best_edp:.2f}x" if best_edp else "   —"
        marker = "  ★ BEST" if r is sorted_valid[0] else ""
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


# Internal cache accessors for scripts that need to confirm cache reuse.
def cache_size() -> int:
    return len(_cache)
