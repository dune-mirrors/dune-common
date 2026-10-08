#!/usr/bin/env python3
# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

"""Repeat independent MPI launches, retaining failures and per-rank PMPI traces."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


DEFAULT_TESTS = r"^(remoteindicestest|syncertest|mpidatatest|communicationtest)-mpi-(2|4)$"


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def capture(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                errors="replace", timeout=15)
        return {"command": command, "returncode": result.returncode,
                "output": result.stdout}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def discover(build_dir, pattern, probe):
    result = subprocess.run(
        ["ctest", "--show-only=json-v1", "-R", pattern], cwd=build_dir,
        check=True, stdout=subprocess.PIPE, text=True)
    tests = []
    for test in json.loads(result.stdout)["tests"]:
        match = re.fullmatch(r"(.+)-mpi-([24])", test["name"])
        if match is None:
            continue
        executable, ranks = match.groups()
        command = test.get("command", [])
        indices = [i for i, arg in enumerate(command) if Path(arg).name == executable]
        if len(indices) != 1 or not Path(command[indices[0]]).is_file():
            raise RuntimeError(f"Build the test executable first: {test['name']}")
        properties = {p["name"]: p["value"] for p in test.get("properties", [])}
        if properties.get("DISABLED") or command[0] == command[indices[0]]:
            raise RuntimeError(f"Test is disabled or has no MPI launcher: {test['name']}")
        tests.append({"name": test["name"], "ranks": int(ranks), "command": command,
                      "executable_index": indices[0], "cwd": properties.get(
                          "WORKING_DIRECTORY", str(Path(command[indices[0]]).parent))})
    if not tests:
        raise RuntimeError("No matching MPI tests. Configure with DUNE_MAX_TEST_CORES=4.")
    # Use the exact launcher and flags discovered by CMake for the plain-C controls.
    controls = []
    for ranks in (2, 4):
        source = next((t for t in tests if t["ranks"] == ranks), None)
        if source is None:
            raise RuntimeError(f"No {ranks}-rank test. Configure with DUNE_MAX_TEST_CORES=4.")
        index = source["executable_index"]
        controls.append({**source, "name": f"plain-c-mpi-{ranks}",
                         "command": source["command"][:index] + [str(probe)],
                         "cwd": str(probe.parent)})
    return controls + tests


def inspect_traces(directory, ranks):
    records = []
    for path in sorted(directory.glob("rank-*-pid-*.log")):
        events = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                  for line in path.read_text().splitlines()]
        records.append({"file": path.name, "events": events})
    expected = set(range(ranks))
    initialized = set()
    returned = set()
    entered = set()
    for record in records:
        for event in record["events"]:
            rank = int(event.get("rank", "-1"))
            if event.get("event") == "init_return" and event.get("rc") == "0":
                initialized.add(rank)
            if event.get("event") == "finalize_enter":
                entered.add(rank)
            if event.get("event") == "finalize_return" and event.get("rc") == "0":
                returned.add(rank)
    complete = (len(records) == ranks and initialized == expected
                and entered == expected and returned == expected)
    return {"records": records, "initialized_ranks": sorted(initialized),
            "entered_finalize_ranks": sorted(entered),
            "returned_successfully_ranks": sorted(returned), "complete": complete}


def classify(returncode, timed_out, tracing, traces):
    if timed_out:
        return "timeout"
    if tracing and traces["complete"]:
        return "passed" if returncode == 0 else "finalize_returned_but_launcher_failed"
    if tracing:
        return "incomplete_finalize_trace" if returncode == 0 else "failed_with_incomplete_trace"
    return "passed" if returncode == 0 else "launcher_failed"


def launch(test, iteration, phase, mode, args):
    relative = Path(f"{phase}-{mode}") / f"{test['name']}-{iteration:04d}"
    directory = args.output / relative
    directory.mkdir(parents=True)
    command = list(test["command"])
    if mode == "on":
        index = test["executable_index"]
        preload = str(args.trace_library)
        if os.environ.get("LD_PRELOAD"):
            preload += ":" + os.environ["LD_PRELOAD"]
        # Inject into ranks only, never into mpiexec or its runtime daemons.
        command[index:index] = ["/usr/bin/env", f"LD_PRELOAD={preload}",
                                f"DUNE_MPI_FINALIZE_LOG_DIR={directory}"]
    started = time.monotonic()
    timed_out = False
    launch_error = None
    returncode = None
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        try:
            process = subprocess.Popen(command, cwd=test["cwd"], stdout=out, stderr=err,
                                       start_new_session=True)
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            returncode = process.returncode
        except OSError as error:
            launch_error = str(error)
            err.write((launch_error + "\n").encode())
    traces = inspect_traces(directory, test["ranks"])
    result = {"test": test["name"], "ranks": test["ranks"], "iteration": iteration,
              "phase": phase, "tracing": mode, "command": command, "cwd": test["cwd"],
              "returncode": returncode, "timed_out": timed_out, "launch_error": launch_error,
              "seconds": time.monotonic() - started, "directory": str(relative),
              "traces": traces,
              "classification": "launch_error" if launch_error else classify(
                  returncode, timed_out, mode == "on", traces)}
    write_json(directory / "result.json", result)
    return result


def metadata(tests, args):
    commands = [[tests[0]["command"][0], "--version"], ["ompi_info", "--version"],
                ["pmix_info", "--version"], ["prte_info", "--version"],
                ["dpkg-query", "-W", "*openmpi*", "*pmix*", "*prrte*", "*mpich*", "*libevent*"],
                ["df", "-h", "/dev/shm", "/tmp"], ["ldd", str(args.trace_library)],
                ["ldd", str(args.probe)]]
    commands += [["ldd", t["command"][t["executable_index"]]] for t in tests if not t["name"].startswith("plain-c")]
    env = {k: v for k, v in os.environ.items() if k.startswith(("OMPI_MCA_", "PMIX_MCA_", "PRTE_MCA_"))
           or k in ("CI_JOB_URL", "CI_PIPELINE_URL", "CI_RUNNER_DESCRIPTION", "CI_RUNNER_ID",
                    "CI_COMMIT_SHA", "CI_JOB_IMAGE", "DUNECI_PARALLEL", "LD_PRELOAD")}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "hostname": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "environment": env, "iterations": args.iterations, "concurrent_launches": args.jobs,
            "timeout_seconds": args.timeout, "tests": tests,
            "commands": [capture(command) for command in commands]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--probe-build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--jobs", type=positive, default=4, help="independent launchers, not MPI ranks")
    parser.add_argument("--timeout", type=positive, default=90, help="seconds per launch")
    parser.add_argument("--test-regex", default=DEFAULT_TESTS)
    parser.add_argument("--phases", nargs="+", choices=("serial", "concurrent"), default=["serial", "concurrent"])
    parser.add_argument("--trace-modes", nargs="+", choices=("off", "on"), default=["off", "on"])
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.output = args.output.resolve()
    args.probe = args.probe_build_dir.resolve() / "mpi_finalize_probe"
    args.trace_library = args.probe_build_dir.resolve() / "libmpi_finalize_trace.so"
    if not args.probe.is_file() or not args.trace_library.is_file():
        parser.error("Build tools/mpi-finalize first (Linux shared MPI libraries required).")
    if len(set(args.phases)) != len(args.phases) or len(set(args.trace_modes)) != len(args.trace_modes):
        parser.error("Do not repeat phases or trace modes.")
    tests = discover(args.build_dir, args.test_regex, args.probe)
    args.output.mkdir(parents=True, exist_ok=False)
    # Match the standard DUNE CI settings, including on local runs.
    os.environ.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    os.environ.setdefault("OMPI_MCA_mpi_yield_when_idle", "1")
    write_json(args.output / "metadata.json", metadata(tests, args))
    results = []
    for phase in args.phases:
        for mode in args.trace_modes:
            tasks = [(test, iteration) for iteration in range(1, args.iterations + 1) for test in tests]
            workers = 1 if phase == "serial" else args.jobs
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(launch, test, iteration, phase, mode, args) for test, iteration in tasks]
                for future in futures:
                    result = future.result()
                    results.append(result)
                    if result["classification"] != "passed":
                        print(f"{result['directory']}: {result['classification']}", flush=True)
            failures = sum(r["classification"] != "passed" for r in results if r["phase"] == phase and r["tracing"] == mode)
            print(f"{phase}, tracing={mode}: {len(tasks)} launches, {failures} failures", flush=True)
            write_json(args.output / "summary.json", results)
    suite = ET.Element("testsuite", name="mpi-finalize-diagnostics", tests=str(len(results)),
                       failures=str(sum(r["classification"] != "passed" for r in results)))
    for result in results:
        case = ET.SubElement(suite, "testcase", name=result["directory"], time=str(result["seconds"]))
        if result["classification"] != "passed":
            ET.SubElement(case, "failure", message=result["classification"]).text = (
                f"Launcher return code: {result['returncode']}; artifacts: {result['directory']}")
    ET.ElementTree(suite).write(args.output / "junit.xml", encoding="utf-8", xml_declaration=True)
    return int(any(r["classification"] != "passed" for r in results))


if __name__ == "__main__":
    sys.exit(main())
