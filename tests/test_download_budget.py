"""Exercise the actual download preflight without network access or model files."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download-model.sh'
CODE = SCRIPT.read_text().split("<<'PYSPACE'\n", 1)[1].split('\nPYSPACE', 1)[0]

class BudgetTests(unittest.TestCase):
    def check_budget(self, free):
        usage = type('Usage', (), {'free': free})()
        args = ['preflight', '/models', '1560936091448', '109000000000', '50000000000']
        with patch.object(sys, 'argv', args), patch('shutil.disk_usage', return_value=usage), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(CODE, str(SCRIPT), 'exec'), {})

    def test_exact_budget(self):
        self.check_budget(1719936091448)

    def test_checkpoint_only_capacity_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            self.check_budget(1617245503488)
        self.assertEqual(caught.exception.code, 1)

    def test_large_disk_no_scientific_notation_or_overflow(self):
        self.check_budget(10**20)

if __name__ == '__main__':
    unittest.main()
