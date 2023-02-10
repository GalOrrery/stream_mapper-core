"""Setup ``stream_ml.core``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mypyc.build import mypycify
from setuptools import setup

##############################################################################
# PARAMETERS

USE_MYPYC: bool = False

CURRENT_DIR = Path(__file__).parent
CORE = CURRENT_DIR / "src" / "stream_ml" / "core"

sys.path.insert(0, str(CURRENT_DIR))  # for setuptools.build_meta


##############################################################################
# CODE
##############################################################################


def find_python_files(base: Path, exclude: tuple[str, ...] = ("test_",)) -> list[Path]:
    """Recursively find python files in all subfolders of base.

    Parameters
    ----------
    base : Path
        Base path from which to search for python files.
    exclude : tuple[str, ...], optional
        Paths to exclude.

    Returns
    -------
    list[Path]
    """
    files: list[Path] = []

    for entry in base.iterdir():
        if entry.name.startswith(exclude):
            continue
        if entry.is_file() and entry.suffix == ".py":
            files.append(entry)
        elif entry.is_dir():
            files.extend(find_python_files(entry))

    return files


# To compile with mypyc, a mypyc checkout must be present on the PYTHONPATH
if len(sys.argv) > 1 and sys.argv[1] == "--use-mypyc":
    sys.argv.pop(1)
    USE_MYPYC = True
if os.getenv("STREAM_ML_CORE_USE_MYPYC", None) == "1":
    print("Found env var STREAM_ML_CORE_USE_MYPYC")  # noqa: T201
    USE_MYPYC = True


if not USE_MYPYC:
    ext_modules = []

else:
    print("BUILDING `stream_ml.core` with MYPYC")  # noqa: T201

    blocklist: list[Path] = [  # TODO: not block
        CORE / "api.py",
        CORE / "base.py",
        *find_python_files(CORE / "multi"),
        *find_python_files(CORE / "prior"),
        CORE / "utils" / "compat.py",
        CORE / "utils" / "funcs.py",
    ]
    discovered: list[Path] = [*find_python_files(CORE)]
    mypyc_targets = [str(p) for p in discovered if p not in blocklist]

    opt_level = os.getenv("MYPYC_OPT_LEVEL", "3")
    ext_modules = mypycify(mypyc_targets, opt_level=opt_level, verbose=True)


setup(name="stream_ml.core", package_dir={"": "src"}, ext_modules=ext_modules)
