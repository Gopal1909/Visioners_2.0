import copy
import unittest
import torch
from learning_v3_optim import FunctionalAdamW


class OptimizerTests(unittest.TestCase):
    def test_native_functional_adamw_matches_stock_trajectory_and_restart(self):
        torch.manual_seed(19)
        a=[torch.nn.Parameter(torch.randn(13,7)),torch.nn.Parameter(torch.randn(7))]
        b=[torch.nn.Parameter(p.detach().clone()) for p in a]
        stock=torch.optim.AdamW(a,lr=.0006,weight_decay=.01,foreach=False)
        light=FunctionalAdamW(b,lr=.0006,weight_decay=.01)
        for step in range(12):
            rate=.0006/(step+1)
            stock.param_groups[0]['lr']=rate;light.param_groups[0]['lr']=rate
            for x,y in zip(a,b):
                x.grad=torch.randn_like(x);y.grad=x.grad.clone()
            stock.step();light.step()
            for i,(x,y) in enumerate(zip(a,b)):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
                torch.testing.assert_close(stock.state[x]['exp_avg'],light.m[i],rtol=0,atol=0)
                torch.testing.assert_close(stock.state[x]['exp_avg_sq'],light.v[i],rtol=0,atol=0)
            if step==5:
                state=copy.deepcopy(light.state_dict());b=[torch.nn.Parameter(p.detach().clone()) for p in b]
                light=FunctionalAdamW(b);light.load_state_dict(state)


if __name__=='__main__':unittest.main(verbosity=2)
