import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diagnostics', ROOT / 'benchmarks/diagnose_spectral.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class SpectralDiagnosticsTest(unittest.TestCase):
    def test_regression_resets_sustained_attainment(self):
        observations = [{'step': i * 100, 'validation': {'accuracy': a}}
                        for i, a in enumerate([0.1, 0.95, 0.8, 0.9, 1.0])]
        self.assertEqual(diag.sustained_attainment(observations), 300)
        observations[-1]['validation']['accuracy'] = 0.8
        self.assertIsNone(diag.sustained_attainment(observations))

    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            diag.sustained_attainment([])
        with self.assertRaises(ValueError):
            diag.sustained_attainment([{'step': 0}, {'step': 0}])

    def test_influential_seed_can_reverse_mean(self):
        summary = diag.paired_summary([{'seed': i, 'test_ce_delta': d}
                                       for i, d in enumerate([-1, -1, 5])])
        self.assertEqual(summary['mean_delta'], 1)
        self.assertEqual(summary['median_delta'], -1)
        self.assertEqual(summary['improved_seeds'], 2)
        self.assertEqual(summary['leave_one_out'][2]['mean_delta'], -1)

    def test_invalid_pairs(self):
        for rows in ([{'seed': 1, 'test_ce_delta': 0}],
                     [{'seed': 1, 'test_ce_delta': 0}] * 2,
                     [{'seed': 1, 'test_ce_delta': float('nan')}, {'seed': 2, 'test_ce_delta': 0}]):
            with self.assertRaises(ValueError):
                diag.paired_summary(rows)

    def test_archive_and_checked_in_report_agree(self):
        import json
        expected = json.loads((ROOT / 'benchmarks/evidence/2026-11-10-spectral-diagnostics/summary.json').read_text())
        self.assertEqual(diag.diagnose(ROOT / 'benchmarks/evidence/2026-10-02-cuda-spectral'), expected)
