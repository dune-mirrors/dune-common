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
