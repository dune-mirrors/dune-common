<!-- SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root -->
<!-- SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception -->

# MPI finalization diagnostics

This opt-in Linux diagnostic compares normal launcher execution with PMPI tracing,
under serial and concurrent independent MPI launches. Throughout the run it samples
the load of the host and container, so that failures can be correlated with it. It does not modify
`MPIHelper`, insert barriers or sleeps, or suppress launcher errors.

The default workload contains `remoteindicestest`, `syncertest`,
`communicationtest` (two and four ranks), and `mpidatatest` (two ranks).
Two plain-C controls, with two and four ranks, only initialize/finalize MPI and
do not link against DUNE. The first two DUNE tests finalize explicitly;
the other two use `MPIHelper`.

## Run locally

Use the same MPI installation for both builds. Start with a fresh DUNE build,
or use an existing build with MPI enabled and `DUNE_MAX_TEST_CORES=4`:

```sh
cmake -S . -B build-mpi-diagnostics \
  -DCMAKE_BUILD_TYPE=Release \
  -DDUNE_ENABLE_PYTHONBINDINGS=OFF -DDUNE_MAX_TEST_CORES=4 \
  -DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE
cmake --build build-mpi-diagnostics --parallel 4 \
  --target remoteindicestest syncertest mpidatatest communicationtest
cmake -S tools/mpi-finalize -B build-mpi-finalize -DCMAKE_BUILD_TYPE=Release
cmake --build build-mpi-finalize --parallel 2
python3 tools/mpi-finalize/run.py \
  --build-dir build-mpi-diagnostics --probe-build-dir build-mpi-finalize \
  --output mpi-finalize-results --iterations 20 --jobs 4
```

For a smoke test, use `--iterations 1`. The output directory must not already
exist, so use a new name for subsequent experiments. `--jobs` controls concurrent
**launchers**, each with two or four ranks; it is independent of the configured
`DUNE_MAX_TEST_CORES`. Twenty iterations mean 720 launches across the nine
program/rank combinations, two phases, and the two default tracing modes
(`--trace-modes off on`). Each launch has
a 90-second timeout (adjust with `--timeout`).

The driver inherits MPI tuning variables and records them. It defaults Open MPI
oversubscription and yielding to the standard DUNE CI settings. To match the
Debian CI images locally, also set `OMPI_MCA_pml='^ucx'`. A successful local run
does not exclude a problem dependent on the Dresden runner or its load.

Trace modes:

- `off`: uninstrumented baseline.
- `on`: the PMPI tracing library is preloaded into the ranks.
- `verbose`: like `on`, plus launcher state verbosity
  (`state`, `odls`, `errmgr` frameworks at level 5, set as both `OMPI_MCA_*` for
  Open MPI 4 and `PRTE_MCA_*` for Open MPI 5). The launcher's stderr then shows
  per-process transitions such as `IOF COMPLETE`, `WAITPID FIRED` and
  `NORMALLY TERMINATED`, or the abnormal path it took. This changes launcher
  timing and adds about 50 lines of stderr per launch.

Host load is sampled every `--sample-interval` seconds (default 0.25).

For a transport comparison, run a new experiment with
`OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp`. Do not combine this change with the
baseline when measuring its effect. This changes MPI message transport, not
PMIx/launcher finalization. When trying MPICH, use fresh builds and explicitly
select matching MPI C/C++ wrappers and `MPIEXEC_EXECUTABLE` in both configurations;
rebuild any MPI-dependent DUNE dependencies too.

## Run in GitLab CI

Pushing a branch whose name starts with **`diagnostics/mpi-finalize`** automatically
runs two additional jobs: Debian 12 / GCC 12 and Debian 13 / GCC 14. The rule also
handles merge-request pipelines. On other branches the jobs are manual; setting
the pipeline variable `MPI_FINALIZE_DIAGNOSTICS=1` enables them automatically.
The existing normal jobs remain unchanged.

Both jobs select the Dresden shared runner using `duneci-dresden`, verified as a
tag of runner 33. Set `MPI_DIAGNOSTIC_RUNNER_TAG=dresden` to compare with runner 1.
The runner must be available to the project, including when testing in a fork.

Pipeline variables:

| Variable | Default | Meaning |
|---|---|---|
| `MPI_DIAGNOSTIC_ITERATIONS` | `20` | Repetitions of each program per phase/tracing mode |
| `MPI_DIAGNOSTIC_JOBS` | `4` | Simultaneous launchers in the concurrent phase |
| `MPI_DIAGNOSTIC_TRACE_MODES` | `off on verbose` | Space-separated trace modes to run |
| `MPI_DIAGNOSTIC_RUNNER_TAG` | `duneci-dresden` | Runner selection |

Failures make the diagnostic job fail, but `allow_failure: true` keeps these
experimental/manual jobs from blocking the normal pipeline. Artifacts are
uploaded on success or failure and retained for two weeks. No retries are used
to turn a failure into a success. The matrix changes compiler and distribution
as well as MPI; inspect the recorded versions before attributing a difference
to a particular MPI component.

## Read the results

- `metadata.json`: actual launcher/package versions, full `ompi_info` output,
  linked libraries, runner identity, CPU model and whether a hypervisor is
  present, CPU affinity, cgroup CPU/memory/PID limits, shared-memory space,
  process list at start, relevant MPI environment, and commands discovered from CTest.
- `host-load.jsonl`: one sample per line with monotonic and wall-clock time,
  load average, runnable tasks, `/proc/stat` CPU counters (including `steal` and
  `iowait`), `procs_running`/`procs_blocked`, host pressure-stall information
  (PSI) for CPU, I/O, and memory, the container cgroup's CPU usage, throttling,
  and pressure, and available memory. `/proc/loadavg`, `/proc/stat`, and
  `/proc/pressure` are not namespaced, so they include the load of other CI
  containers on the same host.
- `analysis.json`: load during each failed launch, and load statistics for failed,
  slow-finalize (any rank over 0.5 s in `MPI_Finalize`), and normal launches.
  The same summary is printed at the end of the job log.
- `summary.json` and `junit.xml`: outcomes of all launches, including failures.
- `<phase>-<off|on>/<test>-<iteration>/`: command/exit status in `result.json`,
  `stdout.log`, `stderr.log`, and (when tracing) one file per MPI rank/PID.

The shared PMPI library is loaded **only into the rank executables**, not into
`mpiexec`. It intercepts `MPI_Init`, `MPI_Init_thread`, `MPI_Finalize`, and
`MPI_Abort`. After `MPI_Init` it also records receipt of `SIGTERM`, `SIGINT`,
`SIGHUP`, `SIGQUIT`, and `SIGPIPE` (as `event=signal rc=<number>`), then
re-delivers the signal with the previous action. This shows whether a rank was
killed by the launcher. Rank records use unbuffered writes and monotonic
timestamps, matching the monotonic clock in `host-load.jsonl`.
`finalize_return rc=0` is written after `PMPI_Finalize` returns without making
further MPI calls. An `atexit` record is also written, but is not proof that all
other exit handlers have completed. MPI initialization failures may produce no
rank file; inspect stderr in that case.

| Classification | Interpretation |
|---|---|
| `passed` | Launcher succeeded; traced runs also have successful init/finalize records from every rank |
| `launcher_blamed_finalized_rank` | The launcher failed and named a rank (`process rank N with PID …`) whose trace shows it returned successfully from finalize; other ranks may have been killed afterwards |
| `finalize_returned_but_launcher_failed` | Every rank returned successfully from finalize, but the launcher failed; strong evidence to investigate runtime termination handling |
| `failed_with_incomplete_trace` | Launcher failed and some ranks lack a complete trace; inspect whether they entered finalize, aborted, or failed earlier |
| `incomplete_finalize_trace` | Launcher succeeded but tracing was incomplete; do not count this as a verified successful finalize experiment |
| `launcher_failed` | Uninstrumented baseline failure |
| `timeout` | Launcher exceeded the per-launch time limit |
| `launch_error` | The launcher could not be started; inspect stderr for the OS error |

Even when every rank returns from finalize, check stderr for subsequent test
failures or signals before attributing the launcher status to a runtime bug.

The serial phase serializes this driver's launches only. Other CI containers can
still load the host; runner-level concurrency is a separate experiment. Tracing
itself changes timing, which is why the uninstrumented baseline is retained.
The control program failing in the same way would show that DUNE is unnecessary
to reproduce the failure. A clean finite run cannot prove the race is absent.

The driver's failure classification and timeout handling can be checked without
MPI using `python3 -m unittest discover -s tools/mpi-finalize -p 'test_*.py'`.
