#!/usr/bin/env python3
# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

"""Generate DUNE packaging metadata, build an sdist, and optionally upload it."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = Path.cwd().resolve()
BACKEND_SOURCE = SCRIPT_DIR / "dune_build_backend.py"
BACKEND = "dune_build_backend"
LEGACY_BUILD_TOOLS = {"setuptools", "wheel", "scikit-build"}

# Use packagemetadata from the dune-common checkout when this script is run
# from its bin/ directory; otherwise fall back to the installed package.
DUNE_COMMON_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "bin" else SCRIPT_DIR
LOCAL_DUNE_PYTHON = DUNE_COMMON_ROOT / "python" / "dune"

if LOCAL_DUNE_PYTHON.is_dir():
    sys.path.insert(0, str(LOCAL_DUNE_PYTHON))

try:
    from packagemetadata import Data, metaData
except ImportError:
    from dune.packagemetadata import Data, metaData


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _toml_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return _toml_string(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        contents = ", ".join(
            f"{_toml_key(str(key))} = {_toml_value(item)}"
            for key, item in value.items()
        )
        return "{ " + contents + " }"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _toml_table(name: str, values: dict[str, Any]) -> list[str]:
    lines = [f"[{name}]"]
    for key, value in values.items():
        if value is not None:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    return lines


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1).lower().replace("_", "-") if match else ""


def _read_existing_build_requirements(pyproject: Path) -> list[str]:
    """Read [build-system].requires from an existing pyproject, if present."""
    return []
    if not pyproject.is_file():
        return []

    in_build_system = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_build_system = stripped == "[build-system]"
            continue

        if in_build_system and stripped.startswith("requires"):
            _, separator, rhs = stripped.partition("=")
            if not separator:
                raise ValueError("Malformed [build-system].requires entry")

            requirements = ast.literal_eval(rhs.strip())
            if not isinstance(requirements, list) or not all(
                isinstance(item, str) for item in requirements
            ):
                raise ValueError(
                    "[build-system].requires must be a list of strings"
                )
            return requirements

    return []


def _make_build_requirements(
    data: Any,
    existing: list[str],
    additional: list[str],
) -> list[str]:
    requirements = list(existing)
    requirements.extend(
        data.asPythonRequirementString(data.depends + data.python_requires)
    )
    requirements.extend(additional)

    requirements = [
        requirement
        for requirement in requirements
        if _requirement_name(requirement) not in LEGACY_BUILD_TOOLS
    ]

    if not any(
        _requirement_name(requirement) == "scikit-build-core"
        for requirement in requirements
    ):
        requirements.insert(0, "scikit-build-core")

    return list(dict.fromkeys(requirements))


def render_pyproject(
    *,
    version: str | None = None,
    dependency_check: bool = False,
    additional_build_requirements: list[str] | None = None,
    existing_pyproject: Path | None = None,
) -> str:
    """Render pyproject.toml using metadata from the current working directory."""
    existing = _read_existing_build_requirements(
        existing_pyproject or (Path.cwd() / "pyproject.toml")
    )
    data, setup = metaData(version, dependencyCheck=dependency_check)
    build_requirements = _make_build_requirements(
        data,
        existing,
        additional_build_requirements or [],
    )

    project: dict[str, Any] = {
        "name": setup["name"],
        "version": str(setup["version"]),
        "description": setup.get("description", ""),
        "classifiers": setup.get("classifiers", []),
        "readme": {
            "text": setup.get("long_description", ""),
            "content-type": setup.get(
                "long_description_content_type", "text/markdown"
            ),
        },
        "dependencies": setup.get("install_requires", []),
    }

    if setup.get("python_requires"):
        project["requires-python"] = setup["python_requires"]

    if setup.get("author"):
        author: dict[str, str] = {"name": setup["author"]}
        if setup.get("author_email"):
            author["email"] = setup["author_email"]
        project["authors"] = [author]

    if setup.get("url"):
        project["urls"] = {"Homepage": setup["url"]}

    lines = [
        "# DO NOT MODIFY THIS FILE!",
        "# Generated by bin/dunepackaging.py.",
        "",
    ]
    lines += _toml_table(
        "build-system",
        {
            "requires": build_requirements,
            "build-backend": BACKEND,
            "backend-path": ["."],
        },
    )
    lines += [""] + _toml_table("project", project)
    lines += [
        "",
        "[tool.scikit-build]",
        "cmake.args = " + _toml_value(setup.get("cmake_args", [])),
    ]
    return "\n".join(lines) + "\n"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def build_sdist(
    project_root: Path,
    output_dir: Path,
    *,
    version: str | None = None,
    dependency_check: bool = False,
    additional_build_requirements: list[str] | None = None,
) -> None:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not BACKEND_SOURCE.is_file():
        raise FileNotFoundError(f"Backend file not found: {BACKEND_SOURCE}")

    with tempfile.TemporaryDirectory(prefix="dune-sdist-") as tmp:
        staging_dir = Path(tmp) / "source"
        staging_dir.mkdir()

        archive = Path(tmp) / "source.tar"
        _run(
            [
                "git", "-C", str(project_root),
                "archive", "--format=tar",
                f"--output={archive}", "HEAD",
            ]
        )
        _run(["tar", "-xf", str(archive), "-C", str(staging_dir)])

        # Generate packaging metadata from the staged, committed dune.module.
        pyproject_text = render_pyproject(
            version=version,
            dependency_check=dependency_check,
            additional_build_requirements=additional_build_requirements,
            existing_pyproject=staging_dir / "pyproject.toml",
        )
        (staging_dir / "pyproject.toml").write_text(
            pyproject_text, encoding="utf-8"
        )
        shutil.copy2(BACKEND_SOURCE, staging_dir / "dune_build_backend.py")

        # scikit-build-core may use git ls-files while constructing the sdist.
        _run(
            [
                "git", "-c", "init.defaultBranch=main",
                "-C", str(staging_dir), "init", "--quiet",
            ]
        )
        _run(["git", "-C", str(staging_dir), "add", "--all"])

        _run(
            [
                sys.executable, "-m", "build",
                "--sdist", "--no-isolation",
                "--outdir", str(output_dir),
                str(staging_dir),
            ]
        )


def clean_generated_files() -> None:
    for name in ("dist", "_skbuild", "build"):
        path = MODULE_ROOT / name
        if path.is_dir():
            print(f"Removing {path}")
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    for path in MODULE_ROOT.glob("*.egg-info"):
        print(f"Removing {path}")
        shutil.rmtree(path)

    cache = MODULE_ROOT / "__pycache__"
    if cache.exists():
        shutil.rmtree(cache)


def upload_sdist(repository: str) -> None:
    archives = sorted((MODULE_ROOT / "dist").glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError("No source distribution found in dist/")

    _run(
        [
            sys.executable, "-m", "twine", "upload",
            "--repository", repository,
            *[str(archive) for archive in archives],
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DUNE packaging metadata, build, or upload an sdist."
    )
    parser.add_argument("--version")
    parser.add_argument("--getversion", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--onlysdist",
        action="store_true",
        help="Build the sdist without uploading it.",
    )
    parser.add_argument(
        "--upload",
        nargs="?",
        const="gitlab",
        choices=("gitlab", "testpypi", "pypi"),
        metavar="REPOSITORY",
    )
    parser.add_argument(
        "--build-requirement",
        action="append",
        default=[],
        metavar="REQUIREMENT",
    )
    parser.add_argument("--dependency-check", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.getversion:
            print(Data(args.version).version)
            return 0

        if args.clean:
            clean_generated_files()
            return 0

        # No action requested: print the generated TOML, without writing it.
        if not args.onlysdist and args.upload is None:
            print(
                render_pyproject(
                    version=args.version,
                    dependency_check=args.dependency_check,
                    additional_build_requirements=args.build_requirement,
                ),
                end="",
            )
            return 0

        dist_dir = MODULE_ROOT / "dist"
        if not args.no_clean and dist_dir.exists():
            shutil.rmtree(dist_dir)

        build_sdist(
            MODULE_ROOT,
            dist_dir,
            version=args.version,
            dependency_check=args.dependency_check,
            additional_build_requirements=args.build_requirement,
        )

        if args.upload is not None:
            upload_sdist(args.upload)

        return 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"dunepackaging: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
