import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1] / 'benchmarks/evidence/2026-11-14-spectral-precision'
spec = importlib.util.spec_from_file_location('late', HERE / 'summarize.py')
late = importlib.util.module_from_spec(spec)
spec.loader.exec_module(late)


class SpectralPrecisionTest(unittest.TestCase):
    def test_report_reproduces(self):
        self.assertEqual(late.summarize(), json.loads((HERE / 'summary.json').read_text()))

    def test_incomplete_index_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'plan.json').write_bytes((HERE / 'plan.json').read_bytes())
            index = json.loads((HERE / 'index.json').read_text())
            index['complete'] = False
            (path / 'index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                late.summarize(path)

    def test_corrupted_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ('plan.json','index.json'):
                (path / name).write_bytes((HERE / name).read_bytes())
            first = json.loads((path / 'index.json').read_text())['results'][0]['file']
            (path / 'raw').mkdir()
            (path / 'raw' / (first+'.gz')).write_bytes(gzip.compress(b'{}'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                late.summarize(path)
