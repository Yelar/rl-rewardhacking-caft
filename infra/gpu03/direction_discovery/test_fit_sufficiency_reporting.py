import copy
import unittest
try:
    from . import refine_fit_sufficiency as r
    from . import summarize_fit_sufficiency as s
except ImportError:
    import refine_fit_sufficiency as r
    import summarize_fit_sufficiency as s


class ReportTests(unittest.TestCase):
    def cell(self):
        return {'pca':{'approximation_warning':False,
                'solver_repeat_agreement':{'same_index_absolute_cosine':[1.]*10},
                'full':{'relative_covariance_residuals':[.0001]*10},
                'solver_repeat':{'relative_covariance_residuals':[.0001]*10}}}

    def test_numeric_rule_ignores_scientific_stability(self):
        cell=self.cell();cell['pca']['groups']={'disjoint_55':{'unstable':True}}
        self.assertFalse(r.needs_refinement(cell))
        cell['pca']['approximation_warning']=True
        self.assertTrue(r.needs_refinement(cell))

    def test_low_solver_agreement_or_large_residual_get_rechecked(self):
        for mutate in [lambda x:x['solver_repeat_agreement']['same_index_absolute_cosine'].__setitem__(9,.98),
                       lambda x:x['full']['relative_covariance_residuals'].__setitem__(9,.06),
                       lambda x:x['solver_repeat']['relative_covariance_residuals'].__setitem__(9,.06)]:
            cell=self.cell();mutate(cell['pca']);self.assertTrue(r.needs_refinement(cell))

    def test_missing_vectors_are_not_zero_stability(self):
        cell={'mean':{'family':{'v_change':{'groups':{'disjoint_55':{'pair_signed_cosine':{'statistics':None}}}}}}}
        self.assertIsNone(s.stat(cell,'family','disjoint_55'))
        self.assertIsNone(s.median([None,None]))
        self.assertEqual(s.median([None,-.5,0,.5]),0.)


if __name__=='__main__':unittest.main()
