import unittest
from .phase_entry import check_limits

class LimitsTests(unittest.TestCase):
    def test_real_numeric_limits_and_overlimits(self):
        expected={'memory_max':4096,'tasks_max':16,'cpu_cores':2}
        v={'memory.max':'4096','memory.current':'100','memory.swap.max':'0','pids.max':'16','cpu.max':'200000 100000'}
        check_limits(v,expected)
        for field,value in [('memory.max','max'),('memory.max','4097'),('memory.swap.max','1'),('pids.max','max'),('pids.max','17'),('cpu.max','max 100000'),('cpu.max','300000 100000'),('memory.current','4096')]:
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                check_limits(dict(v,**{field:value}),expected)

if __name__=='__main__':unittest.main()
