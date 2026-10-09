#!/usr/bin/env python3
# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

"""Repeat independent MPI launches, retaining failures, per-rank PMPI traces and host load."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


DEFAULT_TESTS = r"^(remoteindicestest|syncertest|mpidatatest|communicationtest)-mpi-(2|4)$"

# Launcher-side verbosity for the "verbose" trace mode. Open MPI 4 (ORTE) reads
# OMPI_MCA_*, Open MPI 5 (PRRTE) reads PRTE_MCA_*; unknown variables are ignored.
LAUNCHER_VERBOSE_ENV = {f"{prefix}_{framework}_base_verbose": "5"
                        for prefix in ("OMPI_MCA", "PRTE_MCA")
                        for framework in ("state", "odls", "errmgr")}

# /proc/stat CPU columns, in jiffies.
CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def capture(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                errors="replace", timeout=15)
        return {"command": command, "returncode": result.returncode,
                "output": result.stdout}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def parse_pressure(text):
    """Cumulative stall time in microseconds from a PSI file ("some"/"full" totals)."""
    totals = {}
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        kind, *fields = line.split()
        values = dict(field.split("=", 1) for field in fields if "=" in field)
        if "total" in values:
            totals[kind] = int(values["total"])
    return totals


def sample_host():
    """One snapshot of host-wide and container-level load counters.

    /proc/loadavg, /proc/stat and /proc/pressure are not namespaced, so they show
    the load caused by other CI containers on the same host. The cgroup files
    show this container's own CPU usage and throttling.
    """
    sample = {"monotonic": time.monotonic(), "realtime": time.time()}
    loadavg = read_text("/proc/loadavg")
    if loadavg:
        fields = loadavg.split()
        sample["loadavg"] = [float(x) for x in fields[:3]]
        sample["tasks_runnable"], sample["tasks_total"] = (int(x) for x in fields[3].split("/"))
    for line in (read_text("/proc/stat") or "").splitlines():
        if not line.strip():
            continue
        name, *values = line.split()
        if name == "cpu":
            sample["cpu"] = dict(zip(CPU_FIELDS, (int(v) for v in values)))
        elif name in ("procs_running", "procs_blocked", "ctxt"):
            sample[name] = int(values[0])
    for resource in ("cpu", "io", "memory"):
        pressure = parse_pressure(read_text(f"/proc/pressure/{resource}"))
        if pressure:
            sample[f"host_pressure_{resource}"] = pressure
    cgroup = {}
    for line in (read_text("/sys/fs/cgroup/cpu.stat") or "").splitlines():
        fields = line.split()
        if len(fields) == 2:
            cgroup[fields[0]] = int(fields[1])
    if cgroup:
        sample["cgroup_cpu"] = cgroup
    pressure = parse_pressure(read_text("/sys/fs/cgroup/cpu.pressure"))
    if pressure:
        sample["cgroup_pressure_cpu"] = pressure
    for line in (read_text("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            sample["mem_available_kib"] = int(line.split()[1])
    return sample


class HostSampler(threading.Thread):
    """Append load samples as JSON lines until stopped, independent of the launches."""

    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.stopped = threading.Event()

    def run(self):
        with self.path.open("w") as out:
            while True:
                out.write(json.dumps(sample_host()) + "\n")
                out.flush()
                if self.stopped.wait(self.interval):
                    break

    def stop(self):
        self.stopped.set()
        self.join()


def load_samples(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def window_load(samples, start, end):
    """Load statistics for the monotonic interval [start, end], from the enclosing samples."""
    before = [s for s in samples if s["monotonic"] <= start]
    after = [s for s in samples if s["monotonic"] >= end]
    if not before or not after:
        return None
    first, last = before[-1], after[0]
    inside = [s for s in samples if first["monotonic"] <= s["monotonic"] <= last["monotonic"]]
    seconds = last["monotonic"] - first["monotonic"]
    result = {"seconds": seconds, "samples": len(inside)}
    if "cpu" in first and "cpu" in last:
        delta = {k: last["cpu"].get(k, 0) - first["cpu"].get(k, 0) for k in CPU_FIELDS}
        total = sum(delta.values())
        if total > 0:
            result["host_cpu_busy"] = 1 - (delta["idle"] + delta["iowait"]) / total
            result["host_cpu_steal"] = delta["steal"] / total
            result["host_cpu_iowait"] = delta["iowait"] / total
    for key in ("procs_running", "procs_blocked", "tasks_runnable"):
        values = [s[key] for s in inside if key in s]
        if values:
            result[f"max_{key}"] = max(values)
    loads = [s["loadavg"][0] for s in inside if "loadavg" in s]
    if loads:
        result["max_loadavg1"] = max(loads)
    if seconds > 0:
        # Fraction of wall time in which at least one task stalled ("some").
        for key in ("host_pressure_cpu", "host_pressure_io", "host_pressure_memory",
                    "cgroup_pressure_cpu"):
            if key in first and key in last and "some" in first[key]:
                stalled = (last[key]["some"] - first[key]["some"]) / 1e6
                result[f"{key}_some"] = stalled / seconds
        if "cgroup_cpu" in first and "cgroup_cpu" in last:
            for key in ("nr_throttled", "throttled_usec"):
                if key in first["cgroup_cpu"]:
                    result[f"cgroup_{key}"] = last["cgroup_cpu"][key] - first["cgroup_cpu"][key]
    return result


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
    exited = set()
    signaled = {}
    finalize_seconds = {}
    for record in records:
        enter_time = None
        for event in record["events"]:
            rank = int(event.get("rank", "-1"))
            if event.get("event") == "init_return" and event.get("rc") == "0":
                initialized.add(rank)
            if event.get("event") == "finalize_enter":
                entered.add(rank)
                enter_time = float(event.get("monotonic", "nan"))
            if event.get("event") == "finalize_return" and event.get("rc") == "0":
                returned.add(rank)
                if enter_time is not None:
                    finalize_seconds[rank] = float(event.get("monotonic", "nan")) - enter_time
            if event.get("event") == "atexit":
                exited.add(rank)
            if event.get("event") == "signal":
                signaled[rank] = int(event.get("rc", "0"))
    complete = (len(records) == ranks and initialized == expected
                and entered == expected and returned == expected)
    return {"records": records, "initialized_ranks": sorted(initialized),
            "entered_finalize_ranks": sorted(entered),
            "returned_successfully_ranks": sorted(returned),
            "atexit_ranks": sorted(exited),
            "signaled_ranks": {str(k): v for k, v in sorted(signaled.items())},
            "finalize_seconds": {str(k): v for k, v in sorted(finalize_seconds.items())},
            "complete": complete}


def blamed_process(stderr):
    """Rank and PID that Open MPI's "exiting improperly" message names, if any."""
    match = re.search(r"process rank (\d+) with PID (\d+)", stderr)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def classify(returncode, timed_out, tracing, traces, blamed_rank=None):
    if timed_out:
        return "timeout"
    if tracing and traces["complete"]:
        return "passed" if returncode == 0 else "finalize_returned_but_launcher_failed"
    if (tracing and returncode != 0 and blamed_rank is not None
            and str(blamed_rank) in traces["finalize_seconds"]):
        # The launcher blames a rank whose trace shows it returned from finalize.
        return "launcher_blamed_finalized_rank"
    if tracing:
        return "incomplete_finalize_trace" if returncode == 0 else "failed_with_incomplete_trace"
    return "passed" if returncode == 0 else "launcher_failed"


def launch(test, iteration, phase, mode, args):
    relative = Path(f"{phase}-{mode}") / f"{test['name']}-{iteration:04d}"
    directory = args.output / relative
    directory.mkdir(parents=True)
    command = list(test["command"])
    tracing = mode in ("on", "verbose")
    if tracing:
        index = test["executable_index"]
        preload = str(args.trace_library)
        if os.environ.get("LD_PRELOAD"):
            preload += ":" + os.environ["LD_PRELOAD"]
        # Inject into ranks only, never into mpiexec or its runtime daemons.
        command[index:index] = ["/usr/bin/env", f"LD_PRELOAD={preload}",
                                f"DUNE_MPI_FINALIZE_LOG_DIR={directory}"]
    env = None
    if mode == "verbose":
        env = {**os.environ, **LAUNCHER_VERBOSE_ENV}
    started = time.monotonic()
    timed_out = False
    launch_error = None
    returncode = None
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        try:
            process = subprocess.Popen(command, cwd=test["cwd"], stdout=out, stderr=err,
                                       env=env, start_new_session=True)
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
    finished = time.monotonic()
    traces = inspect_traces(directory, test["ranks"])
    stderr = (directory / "stderr.log").read_text(errors="replace")
    blamed_rank, blamed_pid = blamed_process(stderr)
    result = {"test": test["name"], "ranks": test["ranks"], "iteration": iteration,
              "phase": phase, "tracing": mode, "command": command, "cwd": test["cwd"],
              "returncode": returncode, "timed_out": timed_out, "launch_error": launch_error,
              "started_monotonic": started, "finished_monotonic": finished,
              "seconds": finished - started, "directory": str(relative),
              "blamed_rank": blamed_rank, "blamed_pid": blamed_pid,
              "traces": traces,
              "classification": "launch_error" if launch_error else classify(
                  returncode, timed_out, tracing, traces, blamed_rank)}
    write_json(directory / "result.json", result)
    return result


def describe(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values),
            "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1]}


def analyze(results, samples):
    """Compare host load during failed, slow-finalize and normal launches."""
    for result in results:
        result["load"] = window_load(samples, result["started_monotonic"], result["finished_monotonic"])
        durations = result["traces"]["finalize_seconds"].values()
        result["max_finalize_seconds"] = max(durations) if durations else None
    groups = {
        "failed": [r for r in results if r["classification"] != "passed"],
        "slow_finalize": [r for r in results if r["classification"] == "passed"
                          and (r["max_finalize_seconds"] or 0) > 0.5],
        "normal": [r for r in results if r["classification"] == "passed"
                   and (r["max_finalize_seconds"] or 0) <= 0.5],
    }
    metrics = sorted({k for r in results if r["load"] for k in r["load"]} - {"seconds", "samples"})
    analysis = {"groups": {}, "failures": [], "samples": len(samples)}
    for name, members in groups.items():
        analysis["groups"][name] = {
            "launches": len(members),
            **{metric: describe([(r["load"] or {}).get(metric) for r in members]) for metric in metrics}}
    for r in groups["failed"]:
        analysis["failures"].append({
            "directory": r["directory"], "classification": r["classification"],
            "blamed_rank": r["blamed_rank"], "seconds": r["seconds"],
            "finalize_seconds": r["traces"]["finalize_seconds"],
            "atexit_ranks": r["traces"]["atexit_ranks"],
            "signaled_ranks": r["traces"]["signaled_ranks"], "load": r["load"]})
    finalize = [d for r in results for d in r["traces"]["finalize_seconds"].values()]
    analysis["finalize_seconds"] = describe(finalize)
    analysis["finalize_over_0.5s"] = sum(d > 0.5 for d in finalize)
    return analysis


def print_analysis(analysis):
    keys = ("host_cpu_busy", "host_cpu_steal", "max_procs_running", "max_loadavg1",
            "host_pressure_cpu_some", "cgroup_nr_throttled")
    print(f"Host load during launches ({analysis['samples']} samples; median / max):", flush=True)
    for name, group in analysis["groups"].items():
        cells = []
        for key in keys:
            stats = group.get(key)
            if stats:
                cells.append(f"{key}={stats['median']:.3g}/{stats['max']:.3g}")
        print(f"  {name} ({group['launches']} launches): " + ", ".join(cells), flush=True)
    stats = analysis["finalize_seconds"]
    if stats:
        print(f"  MPI_Finalize seconds: median {stats['median']:.3f}, max {stats['max']:.3f}, "
              f"{analysis['finalize_over_0.5s']} over 0.5 s", flush=True)


def metadata(tests, args):
    commands = [[tests[0]["command"][0], "--version"], ["ompi_info", "--version"],
                ["ompi_info"], ["pmix_info", "--version"], ["prte_info", "--version"],
                ["dpkg-query", "-W", "*openmpi*", "*pmix*", "*prrte*", "*mpich*", "*libevent*"],
                ["df", "-h", "/dev/shm", "/tmp"], ["ldd", str(args.trace_library)],
                ["ldd", str(args.probe)], ["nproc"], ["lscpu"], ["uptime"], ["free", "-m"],
                ["sh", "-c", "ulimit -a"], ["ps", "-eo", "pid,ppid,stat,etime,pcpu,comm"]]
    commands += [["ldd", t["command"][t["executable_index"]]] for t in tests if not t["name"].startswith("plain-c")]
    files = ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.weight", "/sys/fs/cgroup/cpuset.cpus.effective",
             "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/pids.max", "/proc/sys/kernel/pid_max",
             "/proc/sys/kernel/sched_autogroup_enabled"]
    cpuinfo = read_text("/proc/cpuinfo") or ""
    env = {k: v for k, v in os.environ.items() if k.startswith(("OMPI_MCA_", "PMIX_MCA_", "PRTE_MCA_"))
           or k in ("CI_JOB_URL", "CI_PIPELINE_URL", "CI_RUNNER_DESCRIPTION", "CI_RUNNER_ID",
                    "CI_COMMIT_SHA", "CI_JOB_IMAGE", "DUNECI_PARALLEL", "LD_PRELOAD")}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "hostname": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                               if line.startswith("model name")), None),
            "hypervisor": " hypervisor" in cpuinfo,
            "environment": env, "iterations": args.iterations, "concurrent_launches": args.jobs,
            "timeout_seconds": args.timeout, "sample_interval": args.sample_interval,
            "launcher_verbose_environment": LAUNCHER_VERBOSE_ENV, "tests": tests,
            "files": {path: read_text(path) for path in files},
            "commands": [capture(command) for command in commands]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--probe-build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory (must not exist)")
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--jobs", type=positive, default=4, help="independent launchers, not MPI ranks")
    parser.add_argument("--timeout", type=positive, default=90, help="seconds per launch")
    parser.add_argument("--sample-interval", type=float, default=0.25, help="seconds between host load samples")
    parser.add_argument("--test-regex", default=DEFAULT_TESTS)
    parser.add_argument("--phases", nargs="+", choices=("serial", "concurrent"), default=["serial", "concurrent"])
    parser.add_argument("--trace-modes", nargs="+", choices=("off", "on", "verbose"), default=["off", "on"])
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()
    args.output = args.output.resolve()
    args.probe = args.probe_build_dir.resolve() / "mpi_finalize_probe"
    args.trace_library = args.probe_build_dir.resolve() / "libmpi_finalize_trace.so"
    if not args.probe.is_file() or not args.trace_library.is_file():
        parser.error("Build tools/mpi-finalize first (Linux shared MPI libraries required).")
    if len(set(args.phases)) != len(args.phases) or len(set(args.trace_modes)) != len(args.trace_modes):
        parser.error("Do not repeat phases or trace modes.")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive.")
    tests = discover(args.build_dir, args.test_regex, args.probe)
    args.output.mkdir(parents=True, exist_ok=False)
    # Match the standard DUNE CI settings, including on local runs.
    os.environ.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    os.environ.setdefault("OMPI_MCA_mpi_yield_when_idle", "1")
    write_json(args.output / "metadata.json", metadata(tests, args))
    sampler = HostSampler(args.output / "host-load.jsonl", args.sample_interval)
    sampler.start()
    results = []
    try:
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
    finally:
        sampler.stop()
    analysis = analyze(results, load_samples(args.output / "host-load.jsonl"))
    write_json(args.output / "summary.json", results)
    write_json(args.output / "analysis.json", analysis)
    print_analysis(analysis)
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
