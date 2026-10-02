import unittest
import numpy as np
from lob_memory.toy_lob import simulate_lob,aggregate_records,observables

class ToyLOB(unittest.TestCase):
    def test_path_invariants(self):
        r=simulate_lob(10000,seed=42)
        for q in ('qb','qa'):
            self.assertTrue(np.all((r[q]>=1)&(r[q]<=8)))
        self.assertTrue(np.all(np.abs(r['return'])<=1))
        self.assertTrue(np.all(r['ask_removed'][r['return']>0]==1))
        self.assertTrue(np.all(r['bid_removed'][r['return']<0]==1))
        self.assertTrue(np.all(r['ask_removed']+r['bid_removed']<=1))
        self.assertTrue(np.all(np.isfinite(observables(r))))

    def test_aggregate_raw_records_associative(self):
        r=simulate_lob(1031,seed=43)
        direct=aggregate_records(r,8)
        staged=aggregate_records(aggregate_records(r,2),4)
        for k in direct: np.testing.assert_array_equal(direct[k],staged[k])
        np.testing.assert_allclose(observables(direct),observables(staged))
        self.assertEqual(direct['return'].sum(),r['return'][:1024].sum())

    def test_no_removal_depletion_is_zero(self):
        r=dict(qb=np.ones(2),qa=np.ones(2),ask_removed=np.zeros(2),bid_removed=np.zeros(2),
               **{'return':np.zeros(2),'count':np.ones(2)})
        np.testing.assert_array_equal(observables(r),np.zeros((2,3)))
