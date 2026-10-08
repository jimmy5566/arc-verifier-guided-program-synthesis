import subprocess
import unittest
from unittest.mock import patch

from orchestration.supervisor.arc2_supervisor import decode_structured_utf8, prompt_controller


class Utf8SubprocessRegressionTest(unittest.TestCase):
    def test_non_ascii_structured_utf8_json_is_accepted(self):
        completed = subprocess.run(
            ["py", "-3", "-c", "import json,sys; s=''.join(chr(x) for x in (0x63a7,0x5236,0x5668,0x5df2,0x5524,0x9192)); sys.stdout.buffer.write(json.dumps({'message':s}, ensure_ascii=False).encode('utf-8'))"],
            capture_output=True, check=True,
        )
        self.assertEqual({"message": "".join(chr(x) for x in (0x63a7,0x5236,0x5668,0x5df2,0x5524,0x9192))}, __import__("json").loads(decode_structured_utf8(completed.stdout, "TEST")))

    def test_invalid_structured_bytes_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "UTF8_DECODE_ERROR"):
            decode_structured_utf8(b"\xff", "TEST")

    def test_herdr_malformed_json_is_visible_failure(self):
        fake = subprocess.CompletedProcess(["herdr"], 0, b"not-json", b"")
        with patch("orchestration.supervisor.arc2_supervisor.subprocess.run", return_value=fake):
            code, error = prompt_controller(["herdr"], 1)
        self.assertEqual(1, code)
        self.assertIn("HERDR_STRUCTURED_JSON_PARSE_ERROR", error)


if __name__ == "__main__":
    unittest.main()
