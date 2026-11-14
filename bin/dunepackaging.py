#!/usr/bin/env python3
# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

"""Generate DUNE packaging metadata, build an sdist, and optionally upload it."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = Path.cwd().resolve()
BACKEND_SOURCE = SCRIPT_DIR / "dune_build_backend.py"
BACKEND = "dune_build_backend"
LEGACY_BUILD_TOOLS = {"setuptools", "wheel", "scikit-build"}
WHEEL_CONFIG_MANIFEST = "wheel-data-files.json"

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
    if re.fullmatch(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*", value):
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


def _make_build_requirements(
    data: Any,
    additional: list[str],
) -> list[str]:
    requirements = list(
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
    if not any(
        _requirement_name(requirement) == "setuptools"
        for requirement in requirements
    ):
        requirements.append("setuptools")

    return list(dict.fromkeys(requirements))


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change directory for metadata code that reads from cwd."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _safe_source_path(source_text: str, manifest: Path) -> Path:
    source_path = Path(source_text)
    if (
        not source_text
        or source_path.is_absolute()
        or ".." in source_path.parts
        or "\\" in source_text
    ):
        raise ValueError(
            f"{manifest}: force-include source must be a safe relative path: "
            f"{source_text!r}"
        )
    return source_path


def _safe_wheel_path(destination_text: str, manifest: Path) -> PurePosixPath:
    destination = PurePosixPath(destination_text)
    if (
        not destination_text
        or destination.is_absolute()
        or not destination.parts
        or ".." in destination.parts
        or "\\" in destination_text
    ):
        raise ValueError(
            f"{manifest}: force-include destination must be a safe relative "
            f"POSIX path: {destination_text!r}"
        )
    return destination


def _expand_force_include(
    project_root: Path,
    manifest: Path,
    source_to_destination: Any,
) -> dict[str, str]:
    """Validate and expand the force-include source-to-destination mapping."""
    if not isinstance(source_to_destination, dict):
        raise ValueError(
            f"{manifest}: tool.scikit-build.wheel.force-include "
            "must be a JSON object"
        )

    project_root = project_root.resolve()
    resolved_root = project_root
    result: dict[str, str] = {}
    destination_sources: dict[str, str] = {}

    for source_text, destination_text in source_to_destination.items():
        if not isinstance(source_text, str) or not isinstance(
            destination_text, str
        ):
            raise ValueError(
                f"{manifest}: force-include source and destination paths "
                "must be strings"
            )

        source_relative = _safe_source_path(source_text, manifest)
        destination_relative = _safe_wheel_path(destination_text, manifest)
        source = project_root / source_relative

        if not source.exists():
            raise FileNotFoundError(
                f"{manifest}: force-include source does not exist: "
                f"{source_text!r}"
            )

        try:
            source.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"{manifest}: force-include source resolves outside the "
                f"project root: {source_text!r}"
            ) from error

        if source.is_file():
            files = [(source, destination_relative)]
        elif source.is_dir():
            files = []
            for child in sorted(source.rglob("*")):
                if child.is_symlink():
                    try:
                        child.resolve().relative_to(resolved_root)
                    except ValueError as error:
                        raise ValueError(
                            f"{manifest}: symlink resolves outside the "
                            f"project root: "
                            f"{child.relative_to(project_root)}"
                        ) from error

                if child.is_file():
                    files.append(
                        (
                            child,
                            destination_relative
                            / child.relative_to(source).as_posix(),
                        )
                    )
        else:
            raise ValueError(
                f"{manifest}: force-include source is not a file or "
                f"directory: {source_text!r}"
            )

        for file_path, wheel_path in files:
            source_key = file_path.relative_to(project_root).as_posix()
            destination_key = wheel_path.as_posix()

            previous_source = destination_sources.get(destination_key)
            if previous_source is not None and previous_source != source_key:
                raise ValueError(
                    f"{manifest}: multiple sources map to the same wheel "
                    f"destination {destination_key!r}"
                )

            result[source_key] = destination_key
            destination_sources[destination_key] = source_key

    return dict(sorted(result.items()))


def _load_scikit_build_config(project_root: Path) -> dict[str, Any]:
    """Read supported scikit-build-core wheel settings from JSON."""
    manifest = project_root / WHEEL_CONFIG_MANIFEST
    if not manifest.is_file():
        return {}

    try:
        config = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{manifest}: invalid JSON: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"{manifest}: expected a JSON object")

    allowed_sections = {
        "tool.scikit-build.wheel.force-include",
        "tool.scikit-build.wheel.exclude",
    }
    unknown_sections = set(config) - allowed_sections
    if unknown_sections:
        names = ", ".join(sorted(map(str, unknown_sections)))
        raise ValueError(
            f"{manifest}: unsupported configuration section(s): {names}"
        )

    result: dict[str, Any] = {}

    force_include_key = "tool.scikit-build.wheel.force-include"
    if force_include_key in config:
        result["wheel.force-include"] = _expand_force_include(
            project_root,
            manifest,
            config[force_include_key],
        )

    exclude_key = "tool.scikit-build.wheel.exclude"
    if exclude_key in config:
        patterns = config[exclude_key]
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise ValueError(
                f"{manifest}: {exclude_key} must be an array of "
                "non-empty strings"
            )
        result["wheel.exclude"] = patterns

    return result


def render_pyproject(
    *,
    project_root: Path | None = None,
    version: str | None = None,
    dependency_check: bool = False,
    additional_build_requirements: list[str] | None = None,
) -> str:
    """Render pyproject.toml using metadata and configuration from project_root."""
    root = (project_root or Path.cwd()).resolve()

    # packagemetadata reads module metadata relative to the current directory.
    with _working_directory(root):
        data, setup = metaData(version, dependencyCheck=dependency_check)

    build_requirements = _make_build_requirements(
        data,
        additional_build_requirements or [],
    )
    scikit_build_config = _load_scikit_build_config(root)

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
        "# Generated by bin/duneproject.py.",
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

    tool_config: dict[str, Any] = {
        "cmake.args": setup.get("cmake_args", []),
        "wheel.packages": ["python/dune"],
        # "wheel.install-dir": "python",
    }
    tool_config.update(scikit_build_config)
    tool_config.update({"sdist.exclude": ["setup.py"]})

    lines += [""] + _toml_table("tool.scikit-build", tool_config)
    return "\n".join(lines) + "\n"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _stage_git_archive(project_root: Path, staging_dir: Path) -> None:
    """Extract the committed HEAD tree into staging_dir."""
    archive_path = staging_dir.parent / "source.tar"
    _run(
        [
            "git",
            "-C",
            str(project_root),
            "archive",
            "--format=tar",
            "-o",
            str(archive_path),
            "HEAD",
        ]
    )

    # The archive is produced by Git from this repository. Use tarfile's
    # data filter where available to reject paths escaping the staging tree.
    with tarfile.open(archive_path, mode="r:") as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(staging_dir, filter="data")
        else:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        f"Unsafe path in Git archive: {member.name!r}"
                    )
                archive.extract(member, staging_dir)

    archive_path.unlink()


def _stage_tracked_files(project_root: Path, staging_dir: Path) -> None:
    """Copy tracked files using their current working-tree contents."""
    result = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )

    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue

        relative_path = Path(os.fsdecode(raw_path))
        source = project_root / relative_path
        destination = staging_dir / relative_path

        # A tracked file may have been deleted in the working tree.
        if not source.exists() and not source.is_symlink():
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(source.readlink())
        else:
            shutil.copy2(source, destination)


def build_sdist(
    project_root: Path,
    output_dir: Path,
    *,
    source_files: str = "archive",
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

        if source_files == "archive":
            _stage_git_archive(project_root, staging_dir)
        elif source_files == "tracked":
            _stage_tracked_files(project_root, staging_dir)
        else:
            raise ValueError(
                f"Unknown source-files mode: {source_files!r}"
            )

        # Render metadata and read the JSON manifest from the staging tree.
        # Untracked local files cannot be included through the manifest.
        pyproject_text = render_pyproject(
            project_root=staging_dir,
            version=version,
            dependency_check=dependency_check,
            additional_build_requirements=additional_build_requirements,
        )
        (staging_dir / "pyproject.toml").write_text(
            pyproject_text, encoding="utf-8"
        )
        shutil.copy2(BACKEND_SOURCE, staging_dir / "dune_build_backend.py")

        # scikit-build-core may use git ls-files while constructing the sdist.
        _run(
            [
                "git",
                "-c",
                "init.defaultBranch=main",
                "-C",
                str(staging_dir),
                "init",
                "--quiet",
            ]
        )
        _run(["git", "-C", str(staging_dir), "add", "--all"])

        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(output_dir),
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
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    cache = MODULE_ROOT / "__pycache__"
    if cache.exists():
        shutil.rmtree(cache)


def upload_sdist(repository: str) -> None:
    archives = sorted((MODULE_ROOT / "dist").glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError("No source distribution found in dist/")

    _run(
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
    parser.add_argument(
        "--source-files",
        choices=("archive", "tracked"),
        default="archive",
        help=(
            "Source tree for the sdist: committed HEAD via git archive "
            "(default), or current contents of git-tracked files via "
            "git ls-files."
        ),
    )
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

        # No action requested: print generated TOML without writing it.
        if not args.onlysdist and args.upload is None:
            print(
                render_pyproject(
                    project_root=MODULE_ROOT,
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
            source_files=args.source_files,
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
        tarfile.TarError,
    ) as error:
        print(f"duneproject: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
