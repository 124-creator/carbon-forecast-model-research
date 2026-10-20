import unittest
import numpy as np
from pathlib import Path

class Tests(unittest.TestCase):
    def test_mechanism_exists_and_maturity_support(self):
        self.assertTrue(Path(__file__).with_name('prototype.py').exists(), 'missing implementation')
        from prototype import support
        self.assertEqual(support(100,20,60).tolist(), list(range(40,81)))

if __name__ == '__main__': unittest.main()
