"""
Mapper script for YOLO-World-S on your AccelForge accelerator.
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils import run_dnn_layers

# Replace with your partner's architecture YAML name (no .yaml extension)
ARCH_NAME = "basic_analog"

VARIABLE_OVERRIDES = {
    # "ARRAY_ROWS": 256,
    # "ARRAY_COLS": 256,
    # "ADC_RESOLUTION": 6,
    # "BITS_PER_CELL": 2,
}


def run_sanity_check():
    results = run_dnn_layers(
        ARCH_NAME,
        "yolo_world",
        variable_overrides=VARIABLE_OVERRIDES,
        max_layers=1,
        batch_size=1,
    )
    return results


def run_yolo_world(max_layers=21):
    results = run_dnn_layers(
        ARCH_NAME,
        "yolo_world",
        variable_overrides=VARIABLE_OVERRIDES,
        max_layers=max_layers,
        batch_size=1,
    )
    return results


if __name__ == "__main__":
    run_sanity_check()
    run_yolo_world()
