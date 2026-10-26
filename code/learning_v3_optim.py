"""Thin eager AdamW state wrapper using the installed PyTorch update kernel.

Avoids importing compiler/symbolic packages through Optimizer.add_param_group.
Update math is PyTorch's own foreach=False, noncapturable functional AdamW.
"""
import torch
from torch.optim import _functional


class FunctionalAdamW:
    def __init__(self,parameters,lr=.0006,weight_decay=.01):
        self.parameters=list(parameters)
        self.param_groups=[dict(lr=lr,weight_decay=weight_decay)]
        self.m=[torch.zeros_like(p) for p in self.parameters]
        self.v=[torch.zeros_like(p) for p in self.parameters]
        self.steps=[torch.zeros((),dtype=torch.float32,device='cpu') for p in self.parameters]

    def zero_grad(self,set_to_none=True):
        for p in self.parameters:
            if set_to_none:p.grad=None
            elif p.grad is not None:p.grad.zero_()

    @torch.no_grad()
    def step(self):
        active=[i for i,p in enumerate(self.parameters) if p.grad is not None]
        g=self.param_groups[0]
        _functional.adamw([self.parameters[i] for i in active],[self.parameters[i].grad for i in active],
            [self.m[i] for i in active],[self.v[i] for i in active],[],[self.steps[i] for i in active],
            foreach=False,capturable=False,differentiable=False,fused=False,amsgrad=False,beta1=.9,beta2=.999,
            lr=g['lr'],weight_decay=g['weight_decay'],eps=1e-8,maximize=False)

    def state_dict(self):
        return dict(m=self.m,v=self.v,steps=self.steps,param_groups=self.param_groups)

    def load_state_dict(self,state):
        assert len(state['m'])==len(self.parameters)
        self.param_groups=state['param_groups']
        self.m=[x.to(p.device) for x,p in zip(state['m'],self.parameters)]
        self.v=[x.to(p.device) for x,p in zip(state['v'],self.parameters)]
        self.steps=[x.cpu() for x in state['steps']]
