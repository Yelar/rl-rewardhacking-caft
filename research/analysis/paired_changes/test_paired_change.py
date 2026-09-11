import importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('pair_change',p/'paired_change.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class PairedChangeTests(unittest.TestCase):
    def test_direction_and_whole_problem_pairing(self):
        before=(np.array([.1,.5]),np.array([.1,.5]));after=(np.array([.4,.4]),np.array([.4,.4]))
        weights=np.array([[1,0],[0,1],[.5,.5]])
        r=m.change(before,after,weights)
        self.assertAlmostEqual(r['estimate'],.1)
        np.testing.assert_allclose(r['problem_lower_differences'],[.3,-.1])
        self.assertEqual(r['identification_bounds'][0],r['identification_bounds'][1])

    def test_unknowns_are_bounds_not_imputed_points(self):
        weights=np.array([[1.]])
        r=m.change((np.array([0.]),np.array([1.])),(np.array([0.]),np.array([1.])),weights)
        self.assertIsNone(r['estimate']);self.assertEqual(r['identification_bounds'],[-1.,1.])
        self.assertEqual(r['ci95'],[-1.,1.])

    def test_zero_event_bootstrap_degeneracy_is_explicit(self):
        a=(np.zeros(2),np.zeros(2));r=m.change(a,a,np.array([[1,0],[0,1]]))
        self.assertEqual(r['estimate'],0.);self.assertEqual(r['ci95'],[0.,0.])

if __name__=='__main__':unittest.main(verbosity=2)
