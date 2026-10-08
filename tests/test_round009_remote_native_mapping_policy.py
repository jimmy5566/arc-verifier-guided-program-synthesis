from __future__ import annotations
import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MAPPER=ROOT/'scripts'/'run_round009_posthoc_reference_mapping_v1.py'
TRANSPORT=ROOT/'scripts'/'arc2_forced_pty_transport.py'

class RemoteNativeMappingPolicyTests(unittest.TestCase):
    def test_mapper_has_no_forced_pty_payload_dependency(self):
        text=MAPPER.read_text(encoding='utf-8')
        self.assertNotIn('arc2_forced_pty_transport', text)
        tree=ast.parse(text)
        imports='\n'.join((node.module or '') if isinstance(node, ast.ImportFrom) else ' '.join(alias.name for alias in node.names) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        self.assertNotIn('orchestration.supervisor.arc2_supervisor', imports)
        self.assertIn('atomic_json', text)
        self.assertIn('sealed-output', text)
        self.assertIn('receipt', text)

    def test_small_control_files_remain_allowed_and_scientific_payloads_fail_closed(self):
        spec=importlib.util.spec_from_file_location('pty_transport', TRANSPORT); self.assertIsNotNone(spec); self.assertIsNotNone(spec.loader)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            control=root/'receipt.json'; control.write_text('{}', encoding='utf-8')
            module.build(control, root/'manifest', root/'envelope', 1024)
            scientific=root/'reference_mapping.parquet'; scientific.write_bytes(b'opaque')
            with self.assertRaisesRegex(RuntimeError,'PTY_PAYLOAD_TRANSPORT_FORBIDDEN'):
                module.build(scientific, root/'scientific_manifest', root/'scientific_envelope', 1024)

if __name__=='__main__': unittest.main()
