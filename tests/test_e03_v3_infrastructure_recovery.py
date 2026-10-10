from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_module("e03_v3_worker_recovery", "scripts/run_e03_v3_b1_per_example_gradient_screen.py")


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True,
        text=True, encoding="utf-8",
    )
    return result.stdout.strip()


class E03DetachedIdentityIntegrationTests(unittest.TestCase):
    def make_repo(self, root: Path) -> tuple[dict, str, str]:
        """Create a frozen commit, advance its branch, then detach at frozen."""
        root.mkdir(parents=True)
        git(root, "init")
        git(root, "config", "user.email", "e03-test@example.invalid")
        git(root, "config", "user.name", "E03 Test")
        git(root, "checkout", "-b", "infra")
        (root / "scientific.json").write_text('{"frozen":true}\n', encoding="utf-8")
        git(root, "add", "scientific.json")
        git(root, "commit", "-m", "approved frozen fixture")
        frozen = git(root, "rev-parse", "HEAD")
        (root / "development_only.txt").write_text("branch advanced\n", encoding="utf-8")
        git(root, "add", "development_only.txt")
        git(root, "commit", "-m", "development branch advancement")
        advanced = git(root, "rev-parse", "HEAD")
        self.assertNotEqual(advanced, frozen)
        git(root, "checkout", "--detach", frozen)
        self.assertEqual(git(root, "rev-parse", "--abbrev-ref", "HEAD"), "HEAD")
        self.assertEqual(git(root, "rev-parse", "infra"), advanced)
        return {
            "execution_checkout_commit": frozen,
            "executable_source_commit": frozen,
        }, frozen, advanced

    def test_actual_detached_exact_sha_checkout_passes_after_branch_advances(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "repo"
            binding, frozen, advanced = self.make_repo(root)
            self.assertNotEqual(frozen, advanced)
            self.assertEqual(worker.validate_git_identity(binding, root), frozen)

    def test_full_preflight_passes_from_actual_detached_checkout_after_branch_advances(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "repo"
            binding, frozen, advanced = self.make_repo(root)
            self.assertNotEqual(frozen, advanced)
            model = root / "model"
            adapter = root / "adapter"
            model.mkdir()
            adapter.mkdir()
            checkpoint = {"base_path": str(model), "adapter_path": str(adapter), "base_files": [], "adapter_files": []}
            cohort = {"families": [
                {"canonical_family": f"family_{family_index}", "members": [
                    {"episode_id": f"TRAIN:TEST:{family_index}:{row}"} for row in range(8)
                ]} for family_index in range(9)
            ]}
            allow = ["layer.q_proj.lora_A.default.weight"]
            config = {
                "protocol_id": "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN",
                "execution_authorized": False,
                "checkpoint_manifest_path": "checkpoint.json",
                "checkpoint_manifest_sha256": hashlib.sha256(json.dumps(checkpoint).encode()).hexdigest(),
                "cohort_manifest_path": "cohort.json",
                "cohort_manifest_sha256": hashlib.sha256(json.dumps(cohort).encode()).hexdigest(),
                "lora_parameter_name_allowlist": allow,
                "lora_parameter_name_allowlist_sha256": hashlib.sha256(json.dumps(allow, separators=(",", ":")).encode()).hexdigest(),
            }
            response = {"decision": "CONTINUE_CONTROLLER"}
            for name, value in (("checkpoint.json", checkpoint), ("cohort.json", cohort), ("config.json", config), ("response.json", response)):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            output = Path(raw) / "fresh-output"
            binding.update({
                "execution_authorized": True, "status": "EXECUTION_AUTHORIZED_AFTER_DIRECTOR_REVIEW",
                "output_root": str(output.resolve()), "jobs": 1, "retry": False, "runtime_cap_seconds": 1800,
                "bound_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("checkpoint.json", "cohort.json", "config.json", "response.json")},
                "director_response_path": "response.json",
                "director_response_sha256": hashlib.sha256((root / "response.json").read_bytes()).hexdigest(),
            })
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            old_root, old_cap = worker.ROOT, os.environ.get("E03_EXTERNAL_CAP_ENFORCED")
            worker.ROOT = root
            os.environ["E03_EXTERNAL_CAP_ENFORCED"] = "1"
            try:
                result = worker.preflight(root / "config.json", binding_path, output)
                self.assertEqual(result[-1], frozen)
            finally:
                worker.ROOT = old_root
                if old_cap is None:
                    os.environ.pop("E03_EXTERNAL_CAP_ENFORCED", None)
                else:
                    os.environ["E03_EXTERNAL_CAP_ENFORCED"] = old_cap

    def test_wrong_commit_and_dirty_tracked_source_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "repo"
            binding, _, _ = self.make_repo(root)
            wrong = dict(binding, execution_checkout_commit="0" * 40)
            with self.assertRaisesRegex(RuntimeError, "EXECUTION_COMMIT_MISMATCH"):
                worker.validate_git_identity(wrong, root)
            (root / "scientific.json").write_text('{"frozen":false}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "TRACKED_SOURCE_NOT_CLEAN"):
                worker.validate_git_identity(binding, root)

    def test_wrong_scientific_artifact_hash_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "scientific.json"
            artifact.write_text('{"frozen":true}\n', encoding="utf-8")
            valid = hashlib.sha256(artifact.read_bytes()).hexdigest()
            worker.validate_bound_files({"bound_files": {"scientific.json": valid}}, root)
            with self.assertRaisesRegex(RuntimeError, "BOUND_FILE_HASH_MISMATCH"):
                worker.validate_bound_files({"bound_files": {"scientific.json": "0" * 64}}, root)

    def test_actual_preflight_rejects_invalid_execution_authorization(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "repo"
            binding, _, _ = self.make_repo(root)
            binding["execution_authorized"] = False
            config_path, binding_path = root / "config.json", root / "binding.json"
            config_path.write_text(json.dumps({"execution_authorized": False}), encoding="utf-8")
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            old_root, old_cap = worker.ROOT, os.environ.get("E03_EXTERNAL_CAP_ENFORCED")
            worker.ROOT = root
            os.environ["E03_EXTERNAL_CAP_ENFORCED"] = "1"
            try:
                with self.assertRaisesRegex(RuntimeError, "BINDING_AUTHORIZATION_REQUIRED"):
                    worker.preflight(config_path, binding_path, Path(raw) / "fresh-output")
            finally:
                worker.ROOT = old_root
                if old_cap is None:
                    os.environ.pop("E03_EXTERNAL_CAP_ENFORCED", None)
                else:
                    os.environ["E03_EXTERNAL_CAP_ENFORCED"] = old_cap


class E03LauncherIntegrationTests(unittest.TestCase):
    def launcher_command(self, repo: Path, output: Path, authorization: str, worker_script: Path, *extra: str) -> list[str]:
        return [
            sys.executable,
            str(ROOT / "scripts/launch_e03_v3_b1_per_example_gradient_screen.py"),
            "--cap-seconds", "30",
            "--output-root", str(output),
            "--repo-root", str(repo),
            "--authorization-id", authorization,
            "--required-module", "json",
            "--",
            sys.executable,
            str(worker_script),
            str(output),
            *extra,
        ]

    def write_probe(self, repo: Path, *, failing: bool = False, delay_seconds: float = 0.0) -> Path:
        repo.mkdir(parents=True)
        (repo / "probe_module.py").write_text("VALUE = 42\n", encoding="utf-8")
        probe = repo / "probe_worker.py"
        if failing:
            probe.write_text("import probe_module\nraise RuntimeError('ORIGINAL_PROBE_EXCEPTION')\n", encoding="utf-8")
        else:
            probe.write_text(
                "import json, pathlib, sys, time\n"
                "import probe_module\n"
                f"time.sleep({delay_seconds!r})\n"
                "root=pathlib.Path(sys.argv[1]); root.mkdir(parents=True)\n"
                "(root/'TERMINAL_RECEIPT.json').write_text(json.dumps({'status':'COMPLETE_NO_UPDATE','value':probe_module.VALUE}))\n",
                encoding="utf-8",
            )
        return probe

    def test_cpu_dry_run_creates_missing_parents_and_imports_from_launch_root(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            probe = self.write_probe(repo)
            output = root / "missing" / "deep" / "run"
            result = subprocess.run(self.launcher_command(repo, output, "auth-dry-run", probe), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((output / "TERMINAL_RECEIPT.json").read_text())["value"], 42)
            self.assertTrue((output / "LAUNCHER_RECEIPT.json").is_file())
            self.assertTrue((output.parent / f"{output.name}.launcher.log").is_file())

    def test_consumed_authorization_cannot_be_reused_for_new_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            probe = self.write_probe(repo)
            first = root / "runs" / "one"
            second = root / "runs" / "two"
            self.assertEqual(subprocess.run(self.launcher_command(repo, first, "one-shot", probe)).returncode, 0)
            reused = subprocess.run(self.launcher_command(repo, second, "one-shot", probe), capture_output=True, text=True)
            self.assertNotEqual(reused.returncode, 0)
            self.assertIn("E03_V3_DUPLICATE_LAUNCH_BLOCKED", reused.stderr)
            self.assertFalse(second.exists())

    def test_concurrent_launch_for_same_output_is_blocked_even_with_different_authorization(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            probe = self.write_probe(repo, delay_seconds=1.0)
            output = root / "runs" / "same-output"
            first = subprocess.Popen(self.launcher_command(repo, output, "auth-one", probe))
            lock_directory = output.parent / ".e03_active_runs"
            for _ in range(100):
                if lock_directory.is_dir() and any(lock_directory.iterdir()):
                    break
                import time
                time.sleep(0.01)
            second = subprocess.run(
                self.launcher_command(repo, output, "auth-two", probe),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("E03_V3_DUPLICATE_LAUNCH_BLOCKED", second.stderr)
            self.assertEqual(first.wait(timeout=10), 0)

    def test_preworker_dependency_failure_emits_receipt_and_preserves_traceback(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            probe = self.write_probe(repo)
            output = root / "runs" / "dependency-failure"
            command = self.launcher_command(repo, output, "dependency-failure", probe)
            module_index = command.index("json")
            command[module_index] = "module_that_must_not_exist_arc2_e03"
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            receipt = json.loads((output / "TERMINAL_RECEIPT.json").read_text())
            self.assertIn("E03_V3_DEPENDENCY_MISSING", receipt["error_class"])
            log = output.parent / f"{output.name}.launcher.log"
            self.assertIn("E03_V3_DEPENDENCY_MISSING", log.read_text(encoding="utf-8"))

    def test_worker_exception_is_preserved_in_log_and_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            probe = self.write_probe(repo, failing=True)
            output = root / "runs" / "worker-failure"
            result = subprocess.run(self.launcher_command(repo, output, "worker-failure", probe), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((output / "TERMINAL_RECEIPT.json").is_file())
            log = output.parent / f"{output.name}.launcher.log"
            self.assertIn("ORIGINAL_PROBE_EXCEPTION", log.read_text(encoding="utf-8"))

    def test_existing_frozen_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repo = root / "repo"
            probe = self.write_probe(repo)
            output = root / "runs" / "frozen"
            output.mkdir(parents=True)
            sentinel = output / "TERMINAL_RECEIPT.json"
            sentinel.write_text('{"status":"PRESERVED"}\n', encoding="utf-8")
            before = sentinel.read_bytes()
            result = subprocess.run(
                self.launcher_command(repo, output, "fresh-auth", probe),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("E03_V3_FROZEN_OUTPUT_ALREADY_EXISTS", result.stderr)
            self.assertEqual(sentinel.read_bytes(), before)


class E03FrozenScienceIdentityTests(unittest.TestCase):
    def test_train_and_model_manifests_are_unchanged(self):
        train = ROOT / "experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1/E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_MANIFEST_V1.json"
        model = ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
        self.assertEqual(hashlib.sha256(train.read_bytes()).hexdigest(), "2ff3fa4f71eaf4900fcfc75ba418c188048b76cd6287d92bdb1f1efcdbd8b762")
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), "1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549")
        manifest = json.loads(train.read_text(encoding="utf-8"))
        rows = [member for family in manifest["families"] for member in family["members"]]
        self.assertEqual(len(rows), 72)
        self.assertTrue(all(member["episode_id"].startswith("TRAIN:") for member in rows))

    def test_memory_smoke_is_one_example_no_update_and_uses_same_worker_primitives(self):
        source = (ROOT / "scripts/smoke_e03_v3_b1_memory_feasibility.py").read_text(encoding="utf-8")
        self.assertIn('binding.get("authorized_examples") != 1', source)
        self.assertIn('binding.get("optimizer_steps") != 0', source)
        self.assertIn("FROZEN_LONGEST_SEQUENCE_THEN_EPISODE_ID", source)
        self.assertIn("e03.selected_ce", source)
        self.assertNotIn("optimizer.step", source)


if __name__ == "__main__":
    unittest.main()
