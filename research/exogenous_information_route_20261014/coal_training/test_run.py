import importlib.util, pathlib, unittest
class ContractTest(unittest.TestCase):
 def test_real_run_complete_and_causal(self):
  path=pathlib.Path(__file__).parent/'run.py'
  self.assertTrue(path.exists(),'coal execution implementation missing')
  spec=importlib.util.spec_from_file_location('coal_run',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  result=m.main()
  self.assertEqual(result['origins'],451)
  self.assertLess(abs(result['mle_score']-.9497932301207621),1e-12)
  self.assertTrue(result['checks_passed'])
if __name__=='__main__':unittest.main()
