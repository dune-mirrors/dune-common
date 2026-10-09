# SPDX-FileCopyrightInfo: Copyright © DUNE Project contributors, see file LICENSE.md in module root
# SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run


class DiagnosticsTest(unittest.TestCase):
    def test_missing_rank_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for rank in range(2):
                events = ("init_return", "finalize_enter", "finalize_return")
                (directory / f"rank-{rank}-pid-{100 + rank}.log").write_text(
                    "".join(f"event={event} rank={rank} rc=0\n" for event in events))
            traces = run.inspect_traces(directory, 2)
            self.assertTrue(traces["complete"])
            self.assertEqual(run.classify(1, False, True, traces),
                             "finalize_returned_but_launcher_failed")
            (directory / "rank-1-pid-101.log").write_text(
                "event=init_return rank=1 rc=0\nevent=finalize_enter rank=1 rc=0\n")
            traces = run.inspect_traces(directory, 2)
            self.assertFalse(traces["complete"])
            self.assertEqual(traces["returned_successfully_ranks"], [0])
            self.assertEqual(run.classify(0, False, True, traces), "incomplete_finalize_trace")
            self.assertEqual(run.classify(1, False, True, traces), "failed_with_incomplete_trace")

    def test_baseline_and_timeout_classification(self):
        self.assertEqual(run.classify(0, False, False, {}), "passed")
        self.assertEqual(run.classify(1, False, False, {}), "launcher_failed")
        self.assertEqual(run.classify(-15, True, True, {"complete": True}), "timeout")

    def test_no_mpi_tests_is_an_error(self):
        with patch("run.subprocess.run") as process:
            process.return_value.stdout = '{"tests": []}'
            with self.assertRaisesRegex(RuntimeError, "No matching MPI tests"):
                run.discover(Path("/tmp"), "pattern", Path("/tmp/probe"))

    def test_timeout_retains_output_and_result(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            args = argparse.Namespace(output=directory, timeout=0.2)
            test = {"name": "sleep-mpi-2", "ranks": 2, "cwd": name,
                    "command": [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"]}
            result = run.launch(test, 1, "serial", "off", args)
            self.assertEqual(result["classification"], "timeout")
            self.assertIsNotNone(result["returncode"])
            saved = directory / result["directory"]
            self.assertIn("started", (saved / "stdout.log").read_text())
            self.assertTrue(json.loads((saved / "result.json").read_text())["timed_out"])

    def test_launcher_blaming_a_finalized_rank(self):
        stderr = "mpiexec has exited due to process rank 0 with PID 0 on\nnode x exiting improperly."
        self.assertEqual(run.blamed_process(stderr), (0, 0))
        self.assertEqual(run.blamed_process("all good"), (None, None))
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "rank-0-pid-100.log").write_text(
                "event=init_return rank=0 monotonic=1.0 rc=0\n"
                "event=finalize_enter rank=0 monotonic=2.0 rc=0\n"
                "event=finalize_return rank=0 monotonic=4.0 rc=0\n"
                "event=atexit rank=0 monotonic=4.1 rc=0\n")
            (directory / "rank-1-pid-101.log").write_text(
                "event=init_return rank=1 monotonic=1.0 rc=0\n"
                "event=finalize_enter rank=1 monotonic=2.0 rc=0\n"
                "event=signal rank=1 monotonic=4.2 rc=15\n")
            traces = run.inspect_traces(directory, 2)
            self.assertEqual(traces["finalize_seconds"], {"0": 2.0})
            self.assertEqual(traces["atexit_ranks"], [0])
            self.assertEqual(traces["signaled_ranks"], {"1": 15})
            self.assertEqual(run.classify(1, False, True, traces, 0), "launcher_blamed_finalized_rank")
            self.assertEqual(run.classify(1, False, True, traces, 1), "failed_with_incomplete_trace")
            self.assertEqual(run.classify(1, False, False, traces, 0), "launcher_failed")

    def test_window_load(self):
        def sample(t, idle, steal, pressure, running):
            return {"monotonic": t, "loadavg": [running, 0, 0], "procs_running": running,
                    "cpu": {"user": 100 * t, "idle": idle, "steal": steal},
                    "host_pressure_cpu": {"some": pressure}}
        samples = [sample(0, 0, 0, 0, 1), sample(1, 50, 10, 500000, 7), sample(2, 100, 20, 1000000, 2)]
        load = run.window_load(samples, 0.5, 1.5)
        self.assertEqual(load["samples"], 3)
        self.assertAlmostEqual(load["host_cpu_steal"], 20 / 320)
        self.assertAlmostEqual(load["host_cpu_busy"], 1 - 100 / 320)
        self.assertAlmostEqual(load["host_pressure_cpu_some"], 0.5)
        self.assertEqual(load["max_procs_running"], 7)
        self.assertIsNone(run.window_load(samples, -1, 1))
        self.assertEqual(run.parse_pressure("some avg10=0.00 total=12\nfull avg10=0.00 total=3\n"),
                         {"some": 12, "full": 3})
        self.assertIn("monotonic", run.sample_host())

    def test_missing_launcher_retains_error(self):
        with tempfile.TemporaryDirectory() as name:
            args = argparse.Namespace(output=Path(name), timeout=1)
            test = {"name": "missing-mpi-2", "ranks": 2, "cwd": name,
                    "command": [str(Path(name) / "missing-launcher")]}
            result = run.launch(test, 1, "serial", "off", args)
            self.assertEqual(result["classification"], "launch_error")
            self.assertTrue((Path(name) / result["directory"] / "stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
