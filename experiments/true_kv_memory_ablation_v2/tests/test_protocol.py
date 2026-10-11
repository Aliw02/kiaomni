"""Offline protocol tests; intentionally no model download or CUDA required."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "run_ablation.py"


def _isolate_function(name: str):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name
    )
    isolated = ast.Module(body=[found], type_ignores=[])
    space = {}
    exec(compile(isolated, str(SOURCE), "exec"), space)
    return space[name]


class ProtocolTests(unittest.TestCase):
    def test_nine_variants_and_no_streaming(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        assign = next(
            x for x in tree.body
            if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "VARIANTS"
                    for t in x.targets)
        )
        variants = ast.literal_eval(assign.value)
        self.assertEqual(
            set(variants),
            {"F0", "F1", "F2", "F3", "K0", "K1", "K2", "K3", "K4"},
        )
        for name, (chunk, original, deferred) in variants.items():
            self.assertFalse(original and deferred, name)
            if name in ("K2", "K3", "K4"):
                self.assertIsNotNone(chunk)
                self.assertTrue(deferred)

    def test_all_fixed_and_relative_budgets(self):
        schedule = _isolate_function("budget_schedule")
        names = (
            "B98", "B128", "B256", "B512",
            "r0.0625", "r0.125", "r0.25",
        )
        expected = {
            2048: [98, 128, 256, 512],
            4096: [98, 128, 256, 512, 1024],
            8192: [98, 128, 256, 512, 1024, 2048],
        }
        for context, tokens in expected.items():
            with self.subTest(context=context):
                groups = schedule(names, context)
                self.assertEqual([r["tokens"] for r in groups], tokens)
                self.assertEqual(
                    sorted(alias for r in groups for alias in r["aliases"]),
                    sorted(names),
                )

    def test_final_matrix_exactly_756(self):
        schedule = _isolate_function("budget_schedule")
        budgets = (
            "B98", "B128", "B256", "B512",
            "r0.0625", "r0.125", "r0.25",
        )
        kv = sum(6 * 2 * 5 * len(schedule(budgets, x)) for x in (4096, 8192))
        full = 6 * 2 * 2 * 4
        self.assertEqual(kv, 660)
        self.assertEqual(full, 96)
        self.assertEqual(kv + full, 756)


if __name__ == "__main__":
    unittest.main()
