# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

"""PEP 517 wrapper for scikit-build-core with DUNE-specific build steps."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


# Import the backend from the isolated build environment specified by
# [build-system].requires.
from scikit_build_core import build as _scikit_build


PROJECT_ROOT = Path(__file__).resolve().parent


def _cmake_args_from_environment() -> list[str]:
    """Collect additional CMake arguments from DUNE and generic env vars."""
    args: list[str] = []
    for variable in ("DUNE_CMAKE_FLAGS", "CMAKE_FLAGS"):
        value = os.environ.get(variable, "").strip()
        if value:
            args.extend(shlex.split(value))
    return args


def _with_environment_cmake_args(
    config_settings: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Pass environment-provided flags to scikit-build-core."""
    extra_args = _cmake_args_from_environment()
    if not extra_args:
        return config_settings

    settings = dict(config_settings or {})
    existing = settings.get("cmake.args", [])

    if isinstance(existing, str):
        existing_args = shlex.split(existing)
    elif isinstance(existing, (list, tuple)):
        existing_args = list(existing)
    else:
        existing_args = []

    settings["cmake.args"] = existing_args + extra_args
    return settings


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _scikit_build.get_requires_for_build_wheel(
        config_settings=_with_environment_cmake_args(config_settings)
    )


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _scikit_build.get_requires_for_build_sdist(
        config_settings=config_settings
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _scikit_build.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=_with_environment_cmake_args(config_settings),
    )


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _scikit_build.build_sdist(
        sdist_directory,
        config_settings=config_settings,
    )


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    wheel = _scikit_build.build_wheel(
        wheel_directory,
        config_settings=_with_environment_cmake_args(config_settings),
        metadata_directory=metadata_directory,
    )

    # Make the local dune Python package importable for the configure command,
    # including when the build is running in an isolated environment.
    env = os.environ.copy()
    python_dir = str(PROJECT_ROOT / "python")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        python_dir + os.pathsep + old_pythonpath if old_pythonpath else python_dir
    )

    subprocess.run(
        [sys.executable, "-m", "dune", "configure"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    return wheel
