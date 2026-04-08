# From lab 2

import inspect
import linecache
import os
import textwrap
from pathlib import Path
from collections.abc import Callable as AbcCallable
from typing import Any, Callable as TypingCallable, List, Optional, Tuple, Union, get_origin
from types import GenericAlias
from ruamel.yaml.compat import StringIO

import ruamel.yaml
import logging, sys
import numpy as np

logger = logging.getLogger("pytimeloop")
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    # return total
