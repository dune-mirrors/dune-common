#!/usr/bin/env python3
# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

"""Generate DUNE packaging metadata and build or upload a source distribution.

The script supports two modes:

* With no build or upload action, it prints a generated ``pyproject.toml`` to
  stdout. It does not write that file into the project.
* With ``--onlysdist`` or ``--upload``, it stages a source tree in a temporary
  directory, generates ``pyproject.toml`` there, copies in the PEP 517 backend,
  and invokes ``python -m build --sdist --no-isolation``.

By default, the staged tree comes from ``git archive HEAD``; this builds from
the committed tree and ignores working-tree changes. ``--source-files tracked``
instead copies the current contents of Git-tracked files, including uncommitted
edits while still excluding untracked files. The staging directory is
initialized as a Git repository because scikit-build-core may consult Git when
selecting sdist files.

Packaging metadata is obtained from DUNE's ``packagemetadata.py``. The
generated project metadata includes Python dependencies and requirements,
while CMake arguments are written to the scikit-build-core configuration.
Wheel data-file settings can be read from ``wheel-data-files.json``.

The script expects to be run from the module root. Building an sdist requires
the ``build`` package and the build requirements declared in the generated
``pyproject.toml`` to be installed, because this script uses ``--no-isolation``.
Uploading additionally requires ``twine`` and a configured repository.

Examples::

    # Print generated TOML
    python bin/dunepackaging.py

    # Build from committed HEAD
    python bin/dunepackaging.py --onlysdist

    # Build from current contents of tracked files
    python bin/dunepackaging.py --onlysdist --source-files tracked

    # Build and upload to the default repository
    python bin/dunepackaging.py --upload


Wheel data files (module specific hook)
---------------------------------------

A module may provide a ``wheel-data-files.json`` file in its root to configure
additional scikit-build-core wheel settings. The script currently supports
``tool.scikit-build.wheel.force-include`` and
``tool.scikit-build.wheel.exclude``.

For example::

    {
      "tool.scikit-build.wheel.force-include": {
        "data/schema.json": "dune/common/data/schema.json",
        "licenses": "dune/common/licenses"
      },
      "tool.scikit-build.wheel.exclude": [
        "**/CMakeLists.txt",
        "**/*.cmake"
      ]
    }

Force-include mappings use project-relative source paths and wheel-relative
POSIX destination paths. A directory source is expanded recursively. Sources
must exist within the project root; symlinks resolving outside the project are
rejected. Multiple sources may not map to the same wheel destination.

The manifest is read from the source tree staged for the sdist. Thus, in the
default ``git archive`` mode, it must be committed to ``HEAD`` to affect the
sdist; in ``--source-files tracked`` mode, the current contents of a tracked
manifest are used. Untracked files are not copied into either staged tree.
"""


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
    """Encode a string as a TOML basic string."""

    return json.dumps(str(value), ensure_ascii=False)


def _toml_key(value: str) -> str:
    """Format a TOML key, leaving valid bare and dotted keys unquoted."""

    if re.fullmatch(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*", value):
        return value
    return _toml_string(value)


def _toml_value(value: Any) -> str:
    """Serialize a supported Python value as TOML.

    Supports booleans, numbers, strings, lists, tuples, and dictionaries.
    Raises:
        TypeError: If ``value`` has an unsupported type.
    """

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
    """Render a TOML table and its key-value pairs as lines."""

    lines = [f"[{name}]"]
    for key, value in values.items():
        if value is not None:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    return lines


def _requirement_name(requirement: str) -> str:
    """Return a normalized package name from a requirement string."""

    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1).lower().replace("_", "-") if match else ""


def _make_build_requirements(
    data: Any,
    additional: list[str],
) -> list[str]:
    """Build the isolated-build requirement list from DUNE metadata.

    Legacy build tools are filtered out, scikit-build-core and setuptools are
    ensured to be present, and duplicate requirements are removed while
    preserving order.
    """

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
    """Temporarily change cwd, restoring the previous directory on exit.

    This is needed because DUNE's metadata code reads project files relative
    to the current working directory.
    """

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _safe_source_path(source_text: str, manifest: Path) -> Path:
    """Validate a relative source path from the wheel-data manifest.

    Rejects absolute paths, parent-directory traversal, and backslashes.
    """

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
    """Validate a relative POSIX destination path inside the wheel."""

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
    """Validate and expand wheel force-include mappings.

    A directory source expands recursively to individual files. Symlinks that
    resolve outside the project root and conflicting destination mappings are
    rejected. The returned mapping uses project-relative source paths and
    wheel-relative POSIX destinations.
    """

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
    """Load supported wheel settings from ``wheel-data-files.json``.

    Returns scikit-build-core configuration keys in the format expected by
    ``render_pyproject``. Missing manifests produce an empty configuration.
    Invalid JSON, unsupported sections, or malformed settings raise
    ``ValueError``.
    """

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
    """Generate a ``pyproject.toml`` string without writing it.

    Args:
        project_root: Module root used to read metadata and the wheel manifest.
            Defaults to the current working directory.
        version: Optional version override passed to DUNE metadata handling.
        dependency_check: Whether metadata generation should check dependencies.
        additional_build_requirements: Extra PEP 517 build requirements.

    Returns:
        The generated TOML text, including a final newline.
    """

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
    }
    tool_config.update(scikit_build_config)
    tool_config.update({"sdist.exclude": ["setup.py"]})

    lines += [""] + _toml_table("tool.scikit-build", tool_config)
    return "\n".join(lines) + "\n"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    """Print and run a subprocess command, raising on failure."""

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _stage_git_archive(project_root: Path, staging_dir: Path) -> None:
    """Populate ``staging_dir`` from the committed ``HEAD`` Git tree.

    Working-tree edits and untracked files are not included.
    """
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
    """Copy Git-tracked files using their current working-tree contents.

    Includes uncommitted edits to tracked files, skips tracked files deleted
    from the working tree, and does not copy untracked files.
    """

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
    """Stage a source tree and build an sdist using scikit-build-core.

    Args:
        project_root: The DUNE module root.
        output_dir: Directory where the sdist will be written.
        source_files: ``"archive"`` to use committed ``HEAD`` or ``"tracked"``
            to use current contents of tracked files.
        version: Optional package version override.
        dependency_check: Whether to check dependencies during metadata creation.
        additional_build_requirements: Extra PEP 517 build requirements.

    Raises:
        FileNotFoundError: If the backend file is missing.
        subprocess.CalledProcessError: If Git or the build command fails.
        ValueError: If ``source_files`` is unsupported or metadata is invalid.
    """

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
    """Remove common local build outputs from the module root."""

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
    """Upload all ``.tar.gz`` sdists in ``dist/`` with Twine.

    Args:
        repository: Twine repository alias, such as ``pypi`` or ``testpypi``.

    Raises:
        FileNotFoundError: If ``dist/`` contains no source distributions.
    """

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
    """Parse command-line arguments.

    ``argv`` is primarily useful for tests; when omitted, arguments are read
    from the command line.
    """

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
    """Run the requested CLI action and return its process exit code.

    Returns 0 on success and 1 for handled packaging, filesystem, or subprocess
    errors.
    """

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
