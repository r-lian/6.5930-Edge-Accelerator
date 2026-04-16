# Co-design Search Space

## Design axes (all experiments)

| Axis | Values |
|---|---|
| `num_macs` | 32, 64, 128, 256 |
| `sram_kb` | 256, 384, 512 |
| `scratch_kb` | 16, 32, 64, 128 |
| `system_preset` | `deep_embedded` (1.6 GB/s), `high_end_embedded` (4 GB/s) |

Configs are not a full grid — they're 12 hand-picked points that cover meaningful (MACs, memory, bandwidth) combinations and exclude clearly dominated configs:

| num_macs | sram_kb | scratch_kb | system_preset |
|---|---|---|---|
| 32 | 256 | 16 | deep |
| 32 | 384 | 32 | deep |
| 64 | 256 | 16 | deep |
| 64 | 384 | 32 | deep |
| 64 | 384 | 32 | high |
| 128 | 256 | 16 | deep |
| 128 | 384 | 32 | deep ← reference (Ethos-U55/128) |
| 128 | 384 | 32 | high |
| 128 | 512 | 64 | high |
| 256 | 384 | 32 | high |
| 256 | 512 | 64 | high |
| 256 | 512 | 128 | high |

## Per-flag scope

### `--probe`
All 12 configs × **5 probe layers**: T1 (backbone stem), T7 (backbone L3), T10 (backbone L4), T16 (FFN), T19 (det head).  
No budget filter. ~2hrs and no OOM erros

### `--full`
All 12 configs × **all 21 layers** (T0–T20).  
No budget filter. Overnight; has hit OOM on some layers.

### `--budget`
All 12 configs × 5 probe layers, run **3 times** — once per budget tier:

| Tier | Area limit | Power limit |
|---|---|---|
| tight | ≤ 0.20 mm² | ≤ 50 mW |
| baseline | ≤ 0.38 mm² | ≤ 100 mW |
| relaxed | ≤ 0.80 mm² | ≤ 500 mW |

Configs that exceed the area limit are skipped before mapping; power-violating configs are excluded from the reported best.

### `--workload`
The **reference config** (128 MACs / 384 KB / 32 KB-scratch / high_end_embedded) mapped on **all probe layers** for both:
- `yolo_world.yaml` — 640×640 input
- `yolo_world_320.yaml` — 320×320 input (spatial dims halved; text/matmul layers unchanged)
