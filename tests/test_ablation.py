import unittest
from src.models.ablation import LABELS, make_variants
from src.models.benchmark import TECHNICAL, SELECTED


class AblationTests(unittest.TestCase):
    def test_all_subsets_preserve_technical_and_column_order(self):
        technical = TECHNICAL.copy()
        variants = make_variants(technical)
        self.assertEqual(len(variants), 8)
        self.assertEqual(technical, TECHNICAL)
        self.assertEqual(variants['TECHNICAL + A+B+C'], TECHNICAL + SELECTED[5])
        subsets = set()
        for columns in variants.values():
            self.assertEqual(columns[:len(TECHNICAL)], TECHNICAL)
            self.assertNotIn('target', columns)
            self.assertEqual(len(columns), len(set(columns)))
            subsets.add(frozenset(columns[len(TECHNICAL):]))
        self.assertEqual(len(subsets), 8)
        self.assertEqual(set(LABELS), {'A', 'B', 'C'})


if __name__ == '__main__':
    unittest.main()
