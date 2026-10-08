from __future__ import annotations

import unittest

from scripts.arc2_token_grid_parser import TokenGridContract, parse_generated_token_ids
from src.inference.arc_native_io import ARCNativeOutputParser


C = TokenGridContract(tuple(range(10)), 10, 15, 13)


class TokenGridParserTests(unittest.TestCase):
    def valid(self, values: list[int]) -> list[list[int]]:
        result = parse_generated_token_ids(values, C)
        self.assertIsNotNone(result.grid, result)
        return result.grid  # type: ignore[return-value]

    def invalid(self, values: list[int]) -> None:
        self.assertIsNone(parse_generated_token_ids(values, C).grid)

    def test_terminal_eos_and_padding_are_accepted(self) -> None:
        self.assertEqual(self.valid([1, 2, 10, 3, 4, 15]), [[1, 2], [3, 4]])
        result = parse_generated_token_ids([1, 2, 10, 3, 4, 15, 13, 13], C)
        self.assertEqual(result.grid, [[1, 2], [3, 4]])
        self.assertEqual(result.termination_status, "EOS")
        self.assertEqual(result.trailing_pad_count, 2)
        self.assertEqual(self.valid([1, 2, 10, 3, 4]), [[1, 2], [3, 4]])

    def test_invalid_special_and_structure_cases(self) -> None:
        for values in ([15, 1, 2], [1, 15, 2], [1, 2, 15, 15], [1, 99], [1, 2, 10, 3], []):
            self.invalid(values)

    def test_valid_tokens_have_native_parser_parity(self) -> None:
        result = parse_generated_token_ids([0, 1, 10, 2, 3, 15], C)
        self.assertEqual(result.grid, ARCNativeOutputParser.parse("01\n23"))


if __name__ == "__main__":
    unittest.main()
