"""Utility functions for Lab 5 CiM experiments using AccelForge."""

from math import prod
from pathlib import Path
import os
import re
import sys

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(THIS_SCRIPT_DIR)
sys.path.insert(0, LAB_DIR)

import accelforge as af
from _load_spec import get_spec as _get_spec

DNN_WORKLOAD_DIR = os.path.join(LAB_DIR, "workload")

class Result:
    """Wrapper around AccelForge Mappings providing a convenient interface."""

    def __init__(self, mappings, variables=None):
        self._mappings = mappings
        self.variables = variables or {}
        self._per_component_energy = None
        self._computes = None

    @property
    def computes(self):
        if self._computes is None:
            self._computes = self._mappings.num_computes()
        return self._computes

    @property
    def per_component_energy(self):
        if self._per_component_energy is None:
            raw = self._mappings.energy(per_component=True)
            self._per_component_energy = {
                k: float(v) for k, v in raw.items()
            }
        return self._per_component_energy

    def per_compute(self, key):
        """Return per-compute metric.

        If key is 'energy', returns total energy / computes.
        If key is 'per_component_energy', returns dict of component energy / computes.
        """
        if key == "energy":
            total = sum(self.per_component_energy.values())
            return total / self.computes if self.computes else 0
        elif key == "per_component_energy":
            return {
                k: v / self.computes for k, v in self.per_component_energy.items()
            }
        else:
            raise ValueError(f"Unknown key: {key}")

    def clear_zero_energies(self):
        """Remove components with zero energy from per_component_energy."""
        self._per_component_energy = {
            k: v for k, v in self.per_component_energy.items() if v != 0
        }


# Variable name mapping from old UPPERCASE to new AccelForge names.
# Array dimensions are handled via Jinja, not variables.
_VAR_NAME_MAP = {
    "VOLTAGE_DAC_RESOLUTION": "voltage_dac_resolution",
    "BITS_PER_CELL": "bits_per_cell",
    "ADC_RESOLUTION": "adc_resolution",
    "TEMPORAL_DAC_RESOLUTION": "temporal_dac_resolution",
    "SUPPORTED_INPUT_BITS": "supported_input_bits",
    "SUPPORTED_WEIGHT_BITS": "supported_weight_bits",
    "SUPPORTED_OUTPUT_BITS": "supported_output_bits",
}

# These are Jinja variables (affect architecture structure)
_JINJA_VARS = {"ARRAY_ROWS", "ARRAY_COLUMNS", "ARRAY_COLS"}


def show_config(*paths):
    """Print YAML configuration files from provided paths.

    Args:
        *paths: Files or directories. Directories are scanned for `*.yaml`.

    Returns:
        None.
    """
    total = ""
    for path in paths:
        if isinstance(path, str):
            path = Path(path)

        if path.is_dir():
            for p in path.glob("*.yaml"):
                with p.open() as f:
                    total += f.read() + "\n"
        else:
            with path.open() as f:
                total += f.read() + "\n"
    print(total)


def _split_overrides(overrides):
    """Split variable overrides into Jinja and arch variable dicts."""
    jinja_data = {}
    arch_vars = {}
    for k, v in (overrides or {}).items():
        upper = k.upper()
        if upper in _JINJA_VARS or k in _JINJA_VARS:
            # Normalize to the Jinja variable names
            if upper in ("ARRAY_COLUMNS", "ARRAY_COLS"):
                jinja_data["ARRAY_COLS"] = v
            else:
                jinja_data[upper] = v
        elif upper in _VAR_NAME_MAP:
            arch_vars[_VAR_NAME_MAP[upper]] = v
        elif k in _VAR_NAME_MAP.values():
            arch_vars[k] = v
        else:
            # Try both arch.variables and variables
            arch_vars[k] = v
    return jinja_data, arch_vars


def _apply_overrides(spec, arch_vars):
    """Apply variable overrides to a spec."""
    for k, v in arch_vars.items():
        if hasattr(spec.variables, k) and k in spec.variables:
            setattr(spec.variables, k, v)
        if (
            hasattr(spec.arch, "variables")
            and spec.arch.variables
            and k in spec.arch.variables
        ):
            spec.arch.variables[k] = v


def quick_run(arch_name, variable_overrides=None):
    """Run the architecture's built-in workload with optional variable overrides.

    Parameters
    ----------
    arch_name : str
        Architecture name (e.g., 'basic_analog').
    variable_overrides : dict, optional
        Variable overrides. Supports both old UPPERCASE names and new lowercase names.
        ARRAY_ROWS and ARRAY_COLS/ARRAY_COLUMNS are passed as Jinja variables.

    Returns
    -------
    Result
    """
    jinja_data, arch_vars = _split_overrides(variable_overrides)

    spec = _get_spec(arch_name, add_dummy_main_memory=True, jinja_parse_data=jinja_data)
    _apply_overrides(spec, arch_vars)

    spec.mapper.metrics = af.mapper.Metrics.ENERGY
    mappings = spec.map_workload_to_arch(print_progress=False, print_number_of_pmappings=False)

    return Result(mappings, variables=variable_overrides or {})


def round_rank_sizes(workload):
    """ Rounds rank sizes to maximize the number of prime factors. """
    def _get_prime_factors(n) -> list[int]:
        factors = []
        for i in range(2, n + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        return factors

    def _round(n) -> int:
        prime_factors = _get_prime_factors(n)
        # Any prime factors > 16 get rounded
        new_prime_factors = []
        for factor in prime_factors:
            if factor < 8:
                new_prime_factors.append(factor)
            else:
                # Adding 1 will get a 2 into the mix
                new_prime_factors.extend(_round(factor + 1))
        return new_prime_factors

    for rank, size in list(workload.rank_sizes.items()):
        workload.rank_sizes[rank] = prod(_round(size))

    for einsum in workload.einsums:
        for i, expr in enumerate(einsum.iteration_space_shape):
            split = expr.split(" ")
            try:
                new_last = prod(_round(int(split[-1])))
                einsum.iteration_space_shape[i] = " ".join(split[:-1] + [str(new_last)])
            except ValueError:
                pass

    return workload


def run_dnn_layers(
    arch_name,
    dnn_name,
    variable_overrides=None,
    jinja_parse_data=None,
    max_layers=10,
    batch_size=1,
):
    """Run DNN layers through the architecture, returning per-layer results.

    Parameters
    ----------
    arch_name : str
        Architecture name.
    dnn_name : str
        DNN workload name (e.g., 'resnet18', 'mobilenet_v3').
    variable_overrides : dict, optional
        Variable overrides.
    jinja_parse_data : dict, optional
        Additional Jinja variables (e.g., in_out_buf_enabled).
    max_layers : int
        Maximum number of layers to run.
    batch_size : int
        Batch size.

    Returns
    -------
    list[Result]
        One Result per successfully mapped layer.
    """
    jinja_data, arch_vars = _split_overrides(variable_overrides)
    if jinja_parse_data:
        jinja_data.update(jinja_parse_data)

    dnn_path = os.path.join(DNN_WORKLOAD_DIR, f"{dnn_name}.yaml")
    dnn_jinja = {"BATCH_SIZE": batch_size}
    dnn_workload = af.Workload.from_yaml(
        dnn_path, top_key="workload", jinja_parse_data=dnn_jinja
    )
    renames = af.Renames.from_yaml(
        dnn_path, top_key="renames", jinja_parse_data=dnn_jinja
    )

    einsums = dnn_workload.einsums[:max_layers]

    def get_result(einsum):
        print(f'Running Einsum {einsum.name}...')
        spec = _get_spec(
            arch_name, add_dummy_main_memory=True, jinja_parse_data=jinja_data
        )
        _apply_overrides(spec, arch_vars)

        spec.workload = af.Workload(
            einsums=[einsum],
            rank_sizes=dnn_workload.rank_sizes,
            bits_per_value=dnn_workload.bits_per_value,
        )
        spec.mapper.max_pmapping_templates_per_einsum = 1
        round_rank_sizes(spec.workload)
        spec.renames = renames
        spec.mapper.metrics = af.mapper.Metrics.ENERGY
        mappings = spec.map_workload_to_arch(print_progress=False, print_number_of_pmappings=False)
        print(f'Einsum {einsum.name} complete. Per-compute energy: {mappings.per_compute().energy()*1e12:.2f} pJ')
        result = Result(mappings, variables=variable_overrides or {})
        return result

    return [get_result(e) for e in einsums]
