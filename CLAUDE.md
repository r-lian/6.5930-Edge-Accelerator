# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIT 6.5930 Final Project (Spring 2026) by Hector Astrom and Richard Lian. Maps YOLO-World-S (object detection CNN) onto an Ethos-U55-like edge NPU using AccelForge (a Timeloop wrapper), exploring accelerator-workload co-design tradeoffs between hardware resources (area, MACs, SRAM) and workload performance (energy, latency, EDP).

## Running Scripts

AccelForge must be available — use Docker for a guaranteed environment:

```bash
# Start Docker container (Jupyter at http://localhost:8888)
docker compose up
docker compose exec labs bash
```

Once inside the container (or if AccelForge is locally installed):

```bash
# Quick sanity check — map T0 only (<1 min)
python -m milestone_1.mapper_script --sanity

# Map a single layer
python -m milestone_1.mapper_script --layer 7

# Map all 21 YOLO-World layers (~10-15 min)
python -m milestone_1.mapper_script --max-layers 21

# Sweep num_macs + SRAM on T1 to compare EDP
python -m milestone_1.mapper_script --sweep

# Co-design: probe 5 representative layers (~30 min)
python -m milestone_2.codesign --probe

# Co-design: all 21 layers (overnight)
python -m milestone_2.codesign --full

# Co-design: compare 640px vs 320px YOLO resolution
python -m milestone_2.codesign --workload

# Co-design: sweep tight/baseline/relaxed area budgets
python -m milestone_2.codesign --budget
```

## Architecture

### Key Abstractions

**`milestone_1/load_ethos_u55.py`** — Central API. `load_ethos_u55_spec(workload_yaml, mapping_yaml=None, arch_yaml=None, **arch_overrides)` builds an AccelForge `Spec` by filling Jinja2 templates in `arch/ethos_u55.yaml` with energy/area/bandwidth values. All hardware parameter sweeps go through this function.

**`arch/ethos_u55.yaml`** — Jinja2-templated hardware spec. Three-level memory hierarchy:
- `FLASH` (off-chip, slow) → `SRAM` (on-chip shared, 256–512 KB) → `Scratchpad` (local, 16–128 KB)
- Configurable MAC array (32/64/128/256 MACs @ 1 GHz)

**`workload/yolo_world.yaml`** — 21 einsum layers (T0–T20):
- T0: sanity check (tiny conv)
- T1–T12: YOLOv8 backbone (~3.5B MACs total)
- T13–T17: CLIP text encoder (matmuls)
- T18: cross-modal fusion (RepVL-PAN)
- T19–T20: detection head convs

### Hardware Parameters (arch override kwargs)

| Parameter | Default | Options |
|---|---|---|
| `num_macs` | 128 | 32, 64, 128, 256 |
| `memory_mode` | `"dedicated_384kb"` | `"dedicated_512kb"` |
| `system_sram_size_bytes` | — | e.g. `256*1024` |
| `local_buffer_size_bytes` | — | scratchpad override |
| `system_preset` | `"high_end_embedded"` | `"deep_embedded"` (1.6 GB/s vs 4 GB/s) |
| `clock_hz` | 1e9 | any frequency |

### Energy Model Constants (in `load_ethos_u55.py`)

- MAC: 0.08 pJ/op (8-bit, 16 nm)
- SRAM: 1.8 pJ read / 2.2 pJ write
- Scratchpad: 0.5 pJ read / 0.6 pJ write
- FLASH: 15 pJ read / 20 pJ write

### Co-design Design Space (`milestone_2/codesign.py`)

Sweeps 9 configurations × 3 area budgets (0.2 / 0.38 / 0.8 mm²). Probe layers: T1, T7, T10, T16, T19 (one from each network stage). Reports per-layer energy, latency, EDP, and Pareto-optimal area–EDP curves.

## Python Environment

Python 3.13 (`.python-version`). No external pip dependencies beyond what AccelForge provides via Docker. If running locally, ensure `accelforge` / `timeloop` binaries are on `PATH`.
