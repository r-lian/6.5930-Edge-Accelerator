# @Time    : 2026-04-08 13:18
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : load_ethos_u55.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import accelforge as af


_REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_PATH = _REPO_ROOT / "arch" / "ethos_u55.yaml"

# Public Arm Vela presets
SYSTEM_PRESETS: Dict[str, Dict[str, float]] = {
    "deep_embedded": {
        "sram_read_bw_bytes_per_s": 1.6e9,
        "sram_write_bw_bytes_per_s": 1.6e9,
        "flash_read_bw_bytes_per_s": 0.1e9,
        # Flash writes are not the native Ethos-U55 story, but we keep them for
        # generic workloads that need a legal upper-level write target.
        "flash_write_bw_bytes_per_s": 0.1e9,
    },
    "high_end_embedded": {
        "sram_read_bw_bytes_per_s": 4.0e9,
        "sram_write_bw_bytes_per_s": 4.0e9,
        "flash_read_bw_bytes_per_s": 0.5e9,
        "flash_write_bw_bytes_per_s": 0.5e9,
    },
}


# Public Arm Vela memory-mode capacities
MEMORY_MODES: Dict[str, int] = {
    "dedicated_384kb": 384 * 1024,
    "dedicated_512kb": 512 * 1024,
}


DEFAULTS: Dict[str, Any] = {
    # Matmul workload rank sizes for workload/matmul.yaml Jinja
    "M": 1024,
    "N": 512,
    "K": 256,
    # Public Ethos-U55 configurations.
    "num_macs": 128,
    # Public Ethos-U55 headline frequency point.
    "clock_hz": 1.0e9,
    # Int8 is the default modeling mode.
    "bits_per_value": 8,
    # Modeling assumption for the small hidden local buffer near compute.
    "local_buffer_size_bytes": 32 * 1024,
    # Public Vela presets.
    "system_preset": "high_end_embedded",
    "memory_mode": "dedicated_384kb",
    # Provisional energy model. These are not public Ethos-U55 numbers.
    "mac_energy_j": 0.084e-12,
    "sram_read_energy_j": 1.88e-12,
    "sram_write_energy_j": 2.36e-12,
    "scratch_read_energy_j": 0.50e-12,
    "scratch_write_energy_j": 0.60e-12,
    "flash_read_energy_j": 15.0e-12,
    "flash_write_energy_j": 20.0e-12,
    # Provisional area model. Only the 0.1 mm^2 small-core anchor is public.
    # The rest are explicit assumptions so the architecture is usable in AccelForge.
    "u55_32_core_area_m2": 0.1e-6,
    "mac_area_m2": 9.0e-11,
    "sram_area_per_bit_m2": 2.5e-14,
    # Give the local scratchpad non-limiting (massive) bandwidth
    "scratch_read_bw_bytes_per_s": 1.0e12,
    "scratch_write_bw_bytes_per_s": 1.0e12,
}


def _compute_npu_core_area_m2(num_macs: int, u55_32_core_area_m2: float, mac_area_m2: float) -> float:
    """Calibrate a simple fixed-plus-MAC area model to the public 32-MAC anchor."""
    fixed_area_m2 = u55_32_core_area_m2 - 32 * mac_area_m2
    if fixed_area_m2 < 0:
        raise ValueError(
            "u55_32_core_area_m2 is smaller than 32 * mac_area_m2. "
            "Choose a smaller mac_area_m2 or a larger u55_32_core_area_m2."
        )
    return fixed_area_m2 + num_macs * mac_area_m2


def build_ethos_u55_jinja_data(**overrides: Any) -> Dict[str, Any]:
    params = dict(DEFAULTS)
    params.update(overrides)

    num_macs = int(params["num_macs"])
    if num_macs not in (32, 64, 128, 256):
        raise ValueError("num_macs must be one of {32, 64, 128, 256}.")

    system_preset = str(params["system_preset"])
    if system_preset not in SYSTEM_PRESETS:
        raise ValueError(f"Unknown system_preset '{system_preset}'. Choices: {sorted(SYSTEM_PRESETS)}")

    memory_mode = str(params["memory_mode"])
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"Unknown memory_mode '{memory_mode}'. Choices: {sorted(MEMORY_MODES)}")

    preset = SYSTEM_PRESETS[system_preset]
    system_sram_size_bytes = int(params.get("system_sram_size_bytes", MEMORY_MODES[memory_mode]))
    local_buffer_size_bytes = int(params["local_buffer_size_bytes"])
    bits_per_value = int(params["bits_per_value"])

    npu_core_area_m2 = _compute_npu_core_area_m2(
        num_macs=num_macs,
        u55_32_core_area_m2=float(params["u55_32_core_area_m2"]),
        mac_area_m2=float(params["mac_area_m2"]),
    )

    sram_area_per_bit_m2 = float(params["sram_area_per_bit_m2"])
    system_sram_area_m2 = system_sram_size_bytes * 8 * sram_area_per_bit_m2
    local_buffer_area_m2 = local_buffer_size_bytes * 8 * sram_area_per_bit_m2

    return {
        "M": int(params["M"]),
        "N": int(params["N"]),
        "K": int(params["K"]),
        "NUM_MACS": num_macs,
        "CLOCK_HZ": float(params["clock_hz"]),
        "BITS_PER_VALUE": bits_per_value,
        "SYSTEM_SRAM_SIZE_BITS": system_sram_size_bytes * 8,
        "LOCAL_BUFFER_SIZE_BITS": local_buffer_size_bytes * 8,
        "SRAM_READ_BW_BITS_PER_S": float(preset["sram_read_bw_bytes_per_s"]) * 8,
        "SRAM_WRITE_BW_BITS_PER_S": float(preset["sram_write_bw_bytes_per_s"]) * 8,
        "FLASH_READ_BW_BITS_PER_S": float(preset["flash_read_bw_bytes_per_s"]) * 8,
        "FLASH_WRITE_BW_BITS_PER_S": float(preset["flash_write_bw_bytes_per_s"]) * 8,
        "SCRATCH_READ_BW_BITS_PER_S": float(params["scratch_read_bw_bytes_per_s"]) * 8,
        "SCRATCH_WRITE_BW_BITS_PER_S": float(params["scratch_write_bw_bytes_per_s"]) * 8,
        "MAC_ENERGY_J": float(params["mac_energy_j"]),
        "SRAM_READ_ENERGY_J": float(params["sram_read_energy_j"]),
        "SRAM_WRITE_ENERGY_J": float(params["sram_write_energy_j"]),
        "SCRATCH_READ_ENERGY_J": float(params["scratch_read_energy_j"]),
        "SCRATCH_WRITE_ENERGY_J": float(params["scratch_write_energy_j"]),
        "FLASH_READ_ENERGY_J": float(params["flash_read_energy_j"]),
        "FLASH_WRITE_ENERGY_J": float(params["flash_write_energy_j"]),
        "NPU_CORE_AREA_M2": npu_core_area_m2,
        "SYSTEM_SRAM_AREA_M2": system_sram_area_m2,
        "LOCAL_BUFFER_AREA_M2": local_buffer_area_m2,
    }


def load_ethos_u55_spec(
    workload_yaml: str,
    mapping_yaml: Optional[str] = None,
    arch_yaml: str = str(ARCH_PATH),
    workload_jinja_parse: Optional[Dict[str, Any]] = None,
    **arch_overrides: Any,
) -> af.Spec:
    """
    Load a workload and optional mapping against the Ethos-U55-like architecture.

    Examples
    --------
    # Matmul rank sizes (Jinja in workload/matmul.yaml); defaults are M=1024, N=512, K=256.
    spec = load_ethos_u55_spec(
        workload_yaml="workload/matmul.yaml",
        num_macs=128,
        system_preset="high_end_embedded",
        memory_mode="dedicated_384kb",
    )

    spec = load_ethos_u55_spec(
        workload_yaml="workload/matmul.yaml",
        M=512,
        N=512,
        K=512,
        num_macs=256,
        mapping_yaml="mapping/some_mapping.yaml",
        local_buffer_size_bytes=64 * 1024,
    )
    """
    jinja_parse_data = build_ethos_u55_jinja_data(**arch_overrides)
    if workload_jinja_parse:
        jinja_parse_data |= workload_jinja_parse
    yaml_paths = []
    if mapping_yaml is not None:
        yaml_paths.append(mapping_yaml)
    yaml_paths.append(workload_yaml)
    yaml_paths.append(arch_yaml)
    return af.Spec.from_yaml(*yaml_paths, jinja_parse_data=jinja_parse_data)
