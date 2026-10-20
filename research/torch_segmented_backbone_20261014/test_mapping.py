import unittest
class MappingTest(unittest.TestCase):
 def test_mapping(self):
  try: from backbone import mapping_check
  except ImportError: self.fail('mature torch backbone missing')
  result=mapping_check()
  self.assertLess(result['forward_max_error'],1e-10)
  self.assertLess(result['gradient_max_error'],2e-7)
if __name__=='__main__': unittest.main()
