import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TransformersMatrixContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = _load_script(
            "_tg058_verify_transformers_matrix",
            "scripts/verify_transformers_matrix.py",
        )
        cls.worker = _load_script(
            "_tg058_transformers_matrix_probe",
            "scripts/transformers_matrix_probe.py",
        )

    def test_frozen_matrix_has_exact_six_rows(self):
        rows = self.orchestrator.matrix_rows()

        self.assertEqual(self.orchestrator.PYTHON_VERSIONS, ("3.12", "3.13"))
        self.assertEqual(
            self.orchestrator.TRANSFORMERS_VERSIONS,
            ("4.57.3", "5.0.0", "5.14.1"),
        )
        self.assertEqual(self.orchestrator.TORCH_VERSION, "2.10.0")
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row.lane_id for row in rows}), 6)
        self.assertEqual(
            {(row.python_version, row.transformers_version) for row in rows},
            {
                (python_version, transformers_version)
                for python_version in self.orchestrator.PYTHON_VERSIONS
                for transformers_version in self.orchestrator.TRANSFORMERS_VERSIONS
            },
        )

    def test_output_path_is_restricted_to_tg058_temp_root(self):
        matrix_root = REPO_ROOT / ".tmp" / "tg058"
        allowed = self.orchestrator.resolve_output_path(
            REPO_ROOT, matrix_root / "custom" / "matrix.json"
        )
        self.assertTrue(allowed.is_relative_to(matrix_root.resolve()))

        with self.assertRaises(self.orchestrator.MatrixValidationError):
            self.orchestrator.resolve_output_path(
                REPO_ROOT, REPO_ROOT / "tests" / "matrix.json"
            )
        with self.assertRaises(self.orchestrator.MatrixValidationError):
            self.orchestrator.resolve_output_path(
                REPO_ROOT, matrix_root / ".." / "escaped.json"
            )

    def test_cache_environment_is_repo_local_and_offline_for_probe(self):
        matrix_root = (REPO_ROOT / ".tmp" / "tg058").resolve()
        environment = self.orchestrator.build_matrix_environment(
            REPO_ROOT, probe_offline=True
        )
        path_keys = {
            "CONDA_PKGS_DIRS",
            "PIP_CACHE_DIR",
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
            "TMP",
            "TEMP",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "CONDARC",
        }

        for key in path_keys:
            value = Path(environment[key]).resolve()
            self.assertTrue(value.is_relative_to(matrix_root), key)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["HF_HUB_DISABLE_TELEMETRY"], "1")

    def _passing_payload(self, row):
        return {
            "schema_version": 1,
            "lane": row.lane_id,
            "python": f"{row.python_version}.9",
            "transformers": row.transformers_version,
            "torch": "2.10.0+cpu",
            "command": "python scripts/transformers_matrix_probe.py",
            "result": "PASS",
            "surfaces": {
                surface: "PASS" for surface in self.worker.REQUIRED_SURFACES
            },
            "error_category": None,
        }

    def test_worker_result_schema_accepts_only_exact_complete_lane(self):
        row = self.orchestrator.matrix_rows()[0]
        payload = self._passing_payload(row)

        validated = self.orchestrator.validate_worker_result(row, payload)

        self.assertEqual(validated["lane"], row.lane_id)
        self.assertEqual(validated["result"], "PASS")
        self.assertEqual(
            tuple(validated["surfaces"]), self.worker.REQUIRED_SURFACES
        )

    def test_worker_result_rejects_version_surface_and_row_drift(self):
        row = self.orchestrator.matrix_rows()[0]
        mutations = []

        wrong_version = self._passing_payload(row)
        wrong_version["transformers"] = "5.14.0"
        mutations.append(wrong_version)

        wrong_lane = self._passing_payload(row)
        wrong_lane["lane"] = "duplicate-lane"
        mutations.append(wrong_lane)

        missing_surface = self._passing_payload(row)
        missing_surface["surfaces"].pop(self.worker.REQUIRED_SURFACES[-1])
        mutations.append(missing_surface)

        extra_surface = self._passing_payload(row)
        extra_surface["surfaces"]["unplanned_surface"] = "PASS"
        mutations.append(extra_surface)

        for payload in mutations:
            with self.subTest(payload=payload["lane"]):
                with self.assertRaises(self.orchestrator.MatrixValidationError):
                    self.orchestrator.validate_worker_result(row, payload)

        with self.assertRaises(self.orchestrator.MatrixValidationError):
            self.orchestrator.validate_worker_result(row, "{malformed-json-payload")

    def test_missing_lane_returns_categorized_failure_without_starting_worker(self):
        missing_row = self.orchestrator.MatrixRow("9.99", "9.99.9")

        payload = self.orchestrator.run_lane(REPO_ROOT, missing_row)

        self.assertEqual(payload["result"], "FAIL")
        self.assertEqual(payload["error_category"], "LANE_MISSING")
        self.assertEqual(payload["surfaces"]["imports"], "FAIL")

    def test_privacy_validator_rejects_paths_tracebacks_and_raw_errors(self):
        row = self.orchestrator.matrix_rows()[0]
        forbidden_values = (
            str(REPO_ROOT / ".tmp" / "tg058" / "private-file"),
            "Traceback (most recent call last):",
            "token=private-fixture",
            "https://private.invalid.example",
        )

        for forbidden_value in forbidden_values:
            payload = self._passing_payload(row)
            payload["error_category"] = forbidden_value
            with self.subTest(forbidden_value=forbidden_value):
                with self.assertRaises(self.orchestrator.MatrixValidationError):
                    self.orchestrator.validate_worker_result(row, payload)

        payload = self._passing_payload(row)
        payload["raw_error"] = "must never cross the worker boundary"
        with self.assertRaises(self.orchestrator.MatrixValidationError):
            self.orchestrator.validate_worker_result(row, payload)

    def test_duplicate_or_missing_aggregate_rows_fail(self):
        rows = self.orchestrator.matrix_rows()
        payloads = [self._passing_payload(row) for row in rows]

        self.orchestrator.validate_aggregate(rows, payloads)

        with self.assertRaises(self.orchestrator.MatrixValidationError):
            self.orchestrator.validate_aggregate(rows, payloads[:-1])
        with self.assertRaises(self.orchestrator.MatrixValidationError):
            self.orchestrator.validate_aggregate(rows, payloads + [payloads[0]])

    def test_table_and_json_artifact_do_not_expose_absolute_paths(self):
        rows = self.orchestrator.matrix_rows()
        payloads = [self._passing_payload(row) for row in rows]
        table = self.orchestrator.format_matrix_table(payloads)
        serialized = json.dumps({"rows": payloads}, sort_keys=True)

        self.assertNotIn(str(REPO_ROOT), table)
        self.assertNotIn(str(REPO_ROOT), serialized)
        self.assertIn("py312-tf4.57.3", table)
        self.assertIn("py313-tf5.14.1", table)

    def test_worker_failure_categories_never_include_exception_content(self):
        marker = "private-exception-content"
        cases = (
            (ImportError(marker), "IMPORT_FAILURE"),
            (AssertionError(marker), "CONTRACT_FAILURE"),
            (OSError(marker), "IO_FAILURE"),
            (RuntimeError(marker), "PROBE_FAILURE"),
        )

        for error, expected in cases:
            with self.subTest(error_type=type(error).__name__):
                category = self.worker.failure_category(error)
                self.assertEqual(category, expected)
                self.assertNotIn(marker, category)

    def test_worker_cleanup_target_is_exact_fixed_lane_directory(self):
        lane = "py312-tf4.57.3"
        expected = REPO_ROOT / ".tmp" / "tg058" / "work" / lane

        self.assertEqual(
            self.worker.resolve_work_dir(REPO_ROOT, lane, expected),
            expected.resolve(),
        )
        with self.assertRaises(self.worker.WorkerValidationError):
            self.worker.resolve_work_dir(REPO_ROOT, lane, REPO_ROOT / "utils")
        with self.assertRaises(self.worker.WorkerValidationError):
            self.worker.resolve_work_dir(REPO_ROOT, "invalid-lane", expected)

    def test_invalid_worker_arguments_are_not_echoed(self):
        private_marker = "private-worker-argument"
        output = io.StringIO()

        with redirect_stdout(output):
            result = self.worker.main(
                [
                    "--lane",
                    "py312-tf4.57.3",
                    "--expected-python",
                    "3.12",
                    "--expected-transformers",
                    private_marker,
                    "--expected-torch",
                    private_marker,
                    "--repo-root",
                    str(REPO_ROOT),
                    "--work-dir",
                    str(REPO_ROOT / "utils"),
                ]
            )

        self.assertEqual(result, 1)
        self.assertNotIn(private_marker, output.getvalue())
        self.assertIn("WORKER_INPUT_INVALID", output.getvalue())

    def test_atomic_result_writer_rejects_outside_root(self):
        rows = self.orchestrator.matrix_rows()
        aggregate = {"schema_version": 1, "rows": [self._passing_payload(row) for row in rows]}

        with tempfile.TemporaryDirectory(dir=REPO_ROOT / ".tmp") as temp_dir:
            output = Path(temp_dir) / "result.json"
            with self.assertRaises(self.orchestrator.MatrixValidationError):
                self.orchestrator.write_aggregate(REPO_ROOT, output, aggregate)


if __name__ == "__main__":
    unittest.main()
