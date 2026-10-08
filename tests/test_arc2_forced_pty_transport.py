from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; SCRIPT = ROOT / "scripts" / "arc2_forced_pty_transport.py"

def module():
    spec = importlib.util.spec_from_file_location("forced_transport", SCRIPT); assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

class ForcedPtyTransportTests(unittest.TestCase):
    def test_binary_resume_idempotence_and_atomic_publish(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / "canary.jsonl"; source.write_bytes(bytes(range(256)) + "UTF-8-控制".encode())
            manifest = m.build(source, root / "manifest.json", root / "envelope.gz", 1024)
            envelope = (root / "envelope.gz").read_bytes(); encoded = [__import__('base64').b64encode(envelope[i*1024:(i+1)*1024]).decode() for i in range(manifest['chunk_count'])]
            stage = root / "stage"; self.assertEqual(m.receive(stage, manifest, 0, encoded[0])["status"], "ACK")
            self.assertEqual(m.receive(stage, manifest, 0, encoded[0])["status"], "IDEMPOTENT_ACK")
            for i in range(1, manifest['chunk_count']): m.receive(stage, manifest, i, encoded[i])
            result = m.finalize(stage, root / "mounted" / source.name, manifest, False)
            self.assertEqual(result["status"], "PUBLISHED"); self.assertEqual((root / "mounted" / source.name).read_bytes(), source.read_bytes())

    def test_corrupt_and_out_of_order_chunks_fail_closed(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / "canary.jsonl"; source.write_bytes(__import__('os').urandom(4096))
            manifest = m.build(source, root / "m", root / "e", 1024); envelope = (root / "e").read_bytes(); chunk = __import__('base64').b64encode(envelope[:1024]).decode(); second = __import__('base64').b64encode(envelope[1024:2048]).decode()
            with self.assertRaisesRegex(RuntimeError, "HASH_OR_SIZE"): m.receive(root / "s", manifest, 0, __import__('base64').b64encode(b"wrong").decode())
            with self.assertRaisesRegex(RuntimeError, "OUT_OF_ORDER"): m.receive(root / "s", manifest, 1, second)

    def test_incorrect_destination_is_never_overwritten(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / "canary.jsonl"; source.write_bytes(b"correct")
            manifest = m.build(source, root / "m", root / "e", 1024); chunk = __import__('base64').b64encode((root / "e").read_bytes()).decode(); stage = root / "s"; m.receive(stage, manifest, 0, chunk)
            dest = root / "dest" / source.name; dest.parent.mkdir(); dest.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "DESTINATION_CONFLICT"): m.finalize(stage, dest, manifest, False)
            self.assertEqual(dest.read_bytes(), b"wrong")

    def test_final_audit_requires_and_preserves_read_only_mode(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / "audit.jsonl"; source.write_bytes(b"sealed")
            manifest = m.build(source, root / "m", root / "e", 1024); stage = root / "stage"
            m.receive(stage, manifest, 0, __import__('base64').b64encode((root / "e").read_bytes()).decode())
            dest = root / "dest" / source.name; self.assertEqual(m.finalize(stage, dest, manifest, True)["status"], "PUBLISHED")
            self.assertEqual(dest.stat().st_mode & 0o777, 0o444)
            dest.chmod(0o666)
            with self.assertRaisesRegex(RuntimeError, "FINAL_AUDIT_NOT_READ_ONLY"):
                m.finalize(stage, dest, manifest, True)

    def test_persistent_ascii_frames_acknowledge_without_content_parsing(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / "canary.jsonl"; source.write_bytes(b"opaque-bytes\x00UTF-8-\xe9\x8e\xba")
            manifest_path = root / "manifest"; envelope_path = root / "envelope"; manifest = m.build(source, manifest_path, envelope_path, 1024)
            manifest_b64 = __import__('base64').b64encode(manifest_path.read_bytes()).decode()
            chunk_b64 = __import__('base64').b64encode(envelope_path.read_bytes()).decode()
            frames = [
                {"op": "receive", "manifest_b64": manifest_b64, "index": 0, "chunk_b64": chunk_b64},
                {"op": "finalize", "manifest_b64": manifest_b64, "destination": str(root / "dest" / source.name), "final_audit": True},
            ]
            stdin, stdout = io.StringIO("".join(__import__('base64').b64encode(json.dumps(frame, sort_keys=True, separators=(',', ':')).encode()).decode() + "\n" for frame in frames)), io.StringIO()
            original_in, original_out = m.sys.stdin, m.sys.stdout
            try:
                m.sys.stdin, m.sys.stdout = stdin, stdout; self.assertEqual(m.serve(root / "stage"), 0)
            finally:
                m.sys.stdin, m.sys.stdout = original_in, original_out
            replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual([reply["status"] for reply in replies], ["READY", "ACK", "PUBLISHED"])
            self.assertTrue(replies[-1]["read_only"]); self.assertFalse(replies[-1]["content_deserialized"])

    def test_persistent_session_binds_a_chunked_manifest_before_data(self) -> None:
        m = module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / "canary.jsonl"; source.write_bytes(b"opaque" * 500)
            manifest_path = root / "manifest"; envelope_path = root / "envelope"; manifest = m.build(source, manifest_path, envelope_path, 1024)
            manifest_bytes = manifest_path.read_bytes(); manifest_hash = __import__('hashlib').sha256(manifest_bytes).hexdigest()
            data = envelope_path.read_bytes(); chunks = [data[i:i + 1024] for i in range(0, len(data), 1024)]
            manifest_parts = [manifest_bytes[i:i + 512] for i in range(0, len(manifest_bytes), 512)]
            frames = ([{"op": "manifest", "index": i, "chunk_b64": __import__('base64').b64encode(part).decode()} for i, part in enumerate(manifest_parts)] +
                      [{"op": "manifest_finalize"}] +
                      [{"op": "receive", "index": i, "chunk_b64": __import__('base64').b64encode(part).decode()} for i, part in enumerate(chunks)] +
                      [{"op": "finalize", "destination": str(root / "dest" / source.name), "final_audit": False}])
            stdin, stdout = io.StringIO("".join(__import__('base64').b64encode(json.dumps(frame, sort_keys=True, separators=(',', ':')).encode()).decode() + "\n" for frame in frames)), io.StringIO()
            original_in, original_out = m.sys.stdin, m.sys.stdout
            try:
                m.sys.stdin, m.sys.stdout = stdin, stdout
                self.assertEqual(m.serve(root / "stage", manifest_hash, len(manifest_bytes), len(manifest_parts)), 0)
            finally:
                m.sys.stdin, m.sys.stdout = original_in, original_out
            replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(replies[0]["status"], "READY"); self.assertEqual(replies[-1]["status"], "PUBLISHED")
            self.assertEqual((root / "dest" / source.name).read_bytes(), source.read_bytes())
