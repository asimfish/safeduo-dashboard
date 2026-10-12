"""Execute the real worker observer expressions with disjoint arm/finger values."""
from pathlib import Path
import ast,os,unittest
import torch
import numpy as np

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')


class EndpointObserverContracts(unittest.TestCase):
    def expression(self,name):
        version=int(os.environ['NATIVE_OBSERVER_TEST_VERSION'])
        tree=ast.parse((H/f'fourhand_native_probe_v{version}/native_common_beam_v1.py').read_text())
        nodes=[node for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)]
        self.assertEqual(len(nodes),1)
        return compile(ast.Expression(nodes[0].value),'<actual-native-endpoint-expression>','eval')

    def test_full50_target_preserves_arm_endpoint_comparison(self):
        arms=('F_L','F_R','U_L','U_R');applied={};endpoints={};selection={}
        for a in arms:
            k=7 if a.startswith('F') else 6
            # Select from the END of the vector to catch hardcoded first-k slices.
            selection[a]=np.arange(6,6+k)
            applied[a]=torch.ones((1,k+6))*91.
            applied[a][:,selection[a]]=-.125
            endpoints[a]=torch.stack([torch.full((k,),-.125),torch.full((k,),-.125)])
        env=dict(applied=applied,endpoints=endpoints,arm_in_full50=selection,ARMS=arms)
        for name in ('queue_error','pre_queue_error'):
            self.assertEqual(eval(self.expression(name),env),0.)


if __name__=='__main__':unittest.main()
