from __future__ import annotations
import hashlib, importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
def module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file); assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value); return value
launch = module("decoder_launch", "decoder_characterization_launch_v2.py")
worker = module("decoder_worker", "collect_decoder_characterization_v1.py")
parser = module("token_parser", "arc2_token_grid_parser.py")

def write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))

class DecoderCharacterizationV2Tests(unittest.TestCase):
    def make_contract(self, root: Path) -> tuple[Path, dict]:
        cohort = root / "cohort.json"; write(cohort, {"episode_ids": [f"id{i}" for i in range(12)], "role_counts": {"TARGETED_EVALUATION": 4, "TARGETED_COMPOSITION": 4, "RETENTION_SENTINEL": 4}, "selection_rule": "for each required role, choose the four smallest SHA256(ARC2_TOKEN_CHARACTERIZATION_V1:episode_id) values; selection uses episode IDs and roles only"})
        target = root / "target.jsonl"; target.write_bytes(b"{}\n" * 192)
        retention = root / "retention.jsonl"; retention.write_bytes(b"{}\n" * 96)
        immutable = root / "immutable.txt"; immutable.write_text("fixed", encoding="utf-8")
        nonce = root / "nonce"; nonce.write_text("fresh", encoding="utf-8")
        source = ROOT
        argv = [str(Path(sys.executable).resolve()), str((ROOT / "scripts" / "collect_decoder_characterization_v1.py").resolve()), "--launch-contract", str(root / "contract.json"), "--governor-review", str(root / "review.json"), "--output", str(root / "out.json"), "--receipt", str(root / "receipt.json")]
        preflight_argv = [*argv, "--preflight"]
        c = {"contract_id": "ARC2_DECODER_CHARACTERIZATION_LAUNCH_V2", "source_root": str(source), "worker_source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), "argv": argv, "preflight_argv": preflight_argv, "environment": {"PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "0"}, "datasets": {"target_dev_path": str(target), "target_dev_sha256": launch.sha(target), "retention_path": str(retention), "retention_sha256": launch.sha(retention)}, "cohort": {"path": str(cohort), "sha256": launch.sha(cohort)}, "immutable_files": {"worker": {"path": str(immutable), "sha256": launch.sha(immutable)}, "token_parser": {"path": str(immutable), "sha256": launch.sha(immutable)}, "checkpoint_manifest": {"path": str(immutable), "sha256": launch.sha(immutable)}}, "nonce_path": str(nonce), "nonce_sha256": launch.sha(nonce), "nonce_consumed_path": str(root / "consumed.json"), "governor_review_path": str(root / "review.json"), "reviewed_brief_sha256": "brief", "output_path": str(root / "out.json"), "receipt_path": str(root / "receipt.json"), "terminal_failure_receipt_path": str(root / "failure.json"), "preflight_receipt_path": str(root / "preflight.json")}
        c["contract_sha256"] = launch.contract_identity(c); contract = root / "contract.json"; write(contract, c)
        return contract, c

    def test_contract_rejects_substitutions_nonce_replay_and_malformed_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); contract, c = self.make_contract(root); env = c["environment"]
            launch.validate_contract(contract, argv=c["preflight_argv"], environment=env, require_review=None, consume_nonce=False)
            bad = dict(env); bad["CUDA_VISIBLE_DEVICES"] = "9"
            with self.assertRaisesRegex(RuntimeError, "ENVIRONMENT"):
                launch.validate_contract(contract, argv=c["preflight_argv"], environment=bad, require_review=None, consume_nonce=False)
            c["immutable_files"]["worker"]["sha256"] = "0" * 64; write(contract, c)
            with self.assertRaisesRegex(RuntimeError, "IDENTITY|HASH"):
                launch.validate_contract(contract, argv=c["preflight_argv"], environment=env, require_review=None, consume_nonce=False)
            c = self.make_contract(root)[1]; contract = root / "contract.json"
            review = {"decision": "CONTINUE_CONTROLLER", "reviewed_brief_sha256": "brief", "launch_contract_sha256": c["contract_sha256"], "worker_source_commit": c["worker_source_commit"], "cohort_sha256": c["cohort"]["sha256"]}; write(Path(c["governor_review_path"]), review)
            bad_review = dict(review); bad_review["cohort_sha256"] = "wrong"; write(Path(c["governor_review_path"]), bad_review)
            with self.assertRaisesRegex(RuntimeError, "GOVERNOR_REVIEW"):
                launch.validate_contract(contract, argv=c["argv"], environment=env, require_review=Path(c["governor_review_path"]), consume_nonce=False)
            write(Path(c["governor_review_path"]), review)
            launch.validate_contract(contract, argv=c["argv"], environment=env, require_review=Path(c["governor_review_path"]), consume_nonce=True)
            with self.assertRaisesRegex(RuntimeError, "NONCE_ALREADY"):
                launch.validate_contract(contract, argv=c["argv"], environment=env, require_review=Path(c["governor_review_path"]), consume_nonce=True)
            Path(c["preflight_receipt_path"]).write_bytes(b'{"status":"PASS"}\\n')
            with self.assertRaisesRegex(RuntimeError, "MALFORMED"):
                launch.validate_preflight(c, source_commit=c["worker_source_commit"])

    def test_cohort_selection_and_output_collisions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); contract, c = self.make_contract(root); env = c["environment"]
            cohort = Path(c["cohort"]["path"]); bad = json.loads(cohort.read_text()); bad["role_counts"]["RETENTION_SENTINEL"] = 3; write(cohort, bad)
            with self.assertRaisesRegex(RuntimeError, "COHORT_HASH|COHORT_SELECTION"):
                launch.validate_contract(contract, argv=c["preflight_argv"], environment=env, require_review=None, consume_nonce=False)
            contract, c = self.make_contract(root); review = {"decision": "CONTINUE_CONTROLLER", "reviewed_brief_sha256": "brief", "launch_contract_sha256": c["contract_sha256"], "worker_source_commit": c["worker_source_commit"], "cohort_sha256": c["cohort"]["sha256"]}; write(Path(c["governor_review_path"]), review)
            Path(c["terminal_failure_receipt_path"]).write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "OUTPUT_PATH"):
                launch.validate_contract(contract, argv=c["argv"], environment=env, require_review=Path(c["governor_review_path"]), consume_nonce=False)

    def test_characterization_row_is_target_blind_and_complete(self) -> None:
        token_contract = parser.TokenGridContract(tuple(range(10)), 10, 15, 13)
        record = {"episode_id": "x", "role": "TARGETED_EVALUATION", "task": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}}
        with mock.patch.object(worker, "expected", create=True, side_effect=AssertionError("target read")):
            row = worker.character_row("TARGET_DEV", record, "p", [1, 2, 10, 3, 4, 15, 13], token_contract, {"id": "m"})
        required = {"episode_id", "generated_token_ids", "generated_length", "termination_status", "eos_observed", "trailing_pad_count", "prompt_sha256", "parse_valid", "canonical_prediction_sha256"}
        self.assertTrue(required.issubset(row)); self.assertNotIn("exact_grid_match", row); self.assertTrue(row["parse_valid"])

    def test_worker_cli_is_parseable_and_failure_receipt_is_atomic(self) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_decoder_characterization_v1.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dest = root / "terminal.json"
            launch.terminal_failure({"terminal_failure_receipt_path": str(dest), "contract_sha256": "c"}, "TEST")
            self.assertEqual(json.loads(dest.read_text())["reason"], "TEST")

if __name__ == "__main__": unittest.main()
