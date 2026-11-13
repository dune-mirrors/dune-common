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
import tempfile
import sys
from pathlib import Path
from typing import Any

# Import metadata from this dune-common checkout rather than an installed copy.

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = Path.cwd().resolve()
# Only useful when dunepackaging.py is running from a dune-common checkout.
DUNE_COMMON_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "bin"
    else SCRIPT_DIR
)
LOCAL_DUNE_PYTHON = DUNE_COMMON_ROOT / "python" / "dune"


if LOCAL_DUNE_PYTHON.is_dir():
    sys.path.insert(0, str(LOCAL_DUNE_PYTHON))

try:
    from packagemetadata import Data, metaData
except ImportError:
    from dune.packagemetadata import Data, metaData


BACKEND = "dune_build_backend"
LEGACY_BUILD_TOOLS = {"setuptools", "wheel", "scikit-build"}
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_SOURCE = SCRIPT_DIR / "dune_build_backend.py"


def install_backend(project_root: Path = Path('.')) -> None:
    destination = project_root / "dune_build_backend.py"
    shutil.copy2(BACKEND_SOURCE, destination)


def run(command: list[str], *, cwd: Path = MODULE_ROOT) -> None:
    """Run a command and raise an error if it fails."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


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
    """Retain existing requirements and add DUNE build requirements."""
    requirements = list(existing)

    # Preserve DUNE module requirements for isolated builds.
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

    # Keep the first occurrence of each exact requirement.
    return list(dict.fromkeys(requirements))


def generate_pyproject(
    *,
    version: str | None = None,
    dependency_check: bool = False,
    additional_build_requirements: list[str] | None = None,
    output: Path | None = None,
) -> Path:
    """Generate pyproject.toml from dune.module and existing build requirements."""
    output = output or (MODULE_ROOT / "pyproject.toml")
    existing_requirements = _read_existing_build_requirements(output)

    data, setup = metaData(version, dependencyCheck=dependency_check)
    build_requirements = _make_build_requirements(
        data,
        existing_requirements,
        additional_build_requirements or [],
    )

    project: dict[str, Any] = {
        # Preserve the old sdist name normalization.
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

    # module_setup.py wheel-package mapping is intentionally deferred.
    # Tutorial handling is module-dependent and is intentionally omitted here.

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")

    install_backend(output.parent)

    return output


def clean_generated_files() -> None:
    """Remove common generated build and distribution artifacts."""
    for name in ("dist", "_skbuild", "build", "pyproject.toml", "dune_build_backend.py"):
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


def _append_git_hash_to_readme() -> tuple[Path, bytes] | None:
    """Temporarily append the current commit hash to README.md, if possible."""
    readme = MODULE_ROOT / "README.md"
    if not readme.is_file():
        return None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=MODULE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    original = readme.read_bytes()
    with readme.open("a", encoding="utf-8") as stream:
        stream.write(f"\n\ngit-{commit}\n")
    return readme, original


def build_sdist(
    project_root: Path = Path("."),
    output_dir: Path = Path("dist"),
) -> None:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dune-sdist-") as tmp:
        staging_dir = Path(tmp) / "source"
        staging_dir.mkdir()

        archive = staging_dir / "source.tar"
        subprocess.run(
            [
                "git", "-C", str(project_root),
                "archive", "--format=tar",
                f"--output={archive}", "HEAD",
            ],
            check=True,
        )

        # Extract the committed snapshot into a clean staging directory.
        subprocess.run(
            ["tar", "-xf", str(archive), "-C", str(staging_dir)],
            check=True,
        )
        archive.unlink()

        subprocess.run(
                ["git", "-C", str(staging_dir), "init", "--quiet", "--initial-branch=main"],
                check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging_dir), "add", "--all"],
            check=True,
        )
        subprocess.run(
            [
                "python", "-m", "build",
                "--sdist",
                "--outdir", str(output_dir),
                str(staging_dir),
            ],
            check=True,
        )

def upload_sdist(repository: str) -> None:
    """Upload the contents of dist/ with Twine."""
    archives = sorted((MODULE_ROOT / "dist").glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError("No source distribution found in dist/")

    run(
        [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository",
            repository,
            *[str(archive) for archive in archives],
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DUNE packaging files and build/upload an sdist."
    )
    parser.add_argument(
        "--version",
        help="Use this package version instead of the version from dune.module.",
    )
    parser.add_argument(
        "--getversion",
        action="store_true",
        help="Print the package version and exit.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated build and distribution artifacts, then exit.",
    )
    parser.add_argument(
        "--onlysdist",
        action="store_true",
        help="Build the source distribution but do not upload it.",
    )
    parser.add_argument(
        "--upload",
        nargs="?",
        const="gitlab",
        choices=("gitlab", "testpypi", "pypi"),
        metavar="REPOSITORY",
        help="Build and upload to the selected repository (default: gitlab).",
    )
    parser.add_argument(
        "--build-requirement",
        action="append",
        default=[],
        metavar="REQUIREMENT",
        help="Add a build-system requirement; may be supplied more than once.",
    )
    parser.add_argument(
        "--dependency-check",
        action="store_true",
        help="Check dependencies against an existing pyproject.toml.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove previous dist/ contents before building.",
    )
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

        generate_pyproject(
            version=args.version,
            dependency_check=args.dependency_check,
            additional_build_requirements=args.build_requirement,
        )

        # A bare invocation generates pyproject.toml only. Build when the user
        # explicitly requests an sdist or an upload.
        if not args.onlysdist and args.upload is None:
            return 0

        if not args.no_clean:
            dist = MODULE_ROOT / "dist"
            if dist.exists():
                shutil.rmtree(dist)

        build_sdist()

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
