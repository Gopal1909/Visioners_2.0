"""Exact legacy PointNet kernels without geometry/plotting imports."""
import numpy as np
import torch
from torch import nn

class SiamesePointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(387,64),nn.ReLU(),nn.Dropout(.1),
                                nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def encode(self,p,mask):
        f=self.encoder(p)
        maximum=f.masked_fill(~mask[...,None],-1e9).max(dim=1).values
        maximum=torch.where(mask.any(dim=1,keepdim=True),maximum,torch.zeros_like(maximum))
        mean=(f*mask[...,None]).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
        return torch.cat((maximum,mean),dim=1)

    def forward(self,p0,m0,p1,m1,aux):
        a,b=self.encode(p0,m0),self.encode(p1,m1)
        return self.head(torch.cat((a,b,torch.abs(a-b),aux),dim=1)).squeeze(-1)


def tensor_batch(features,ids,device):
    return [torch.from_numpy(features[key][ids]).to(device) for key in ('p0','m0','p1','m1','aux')]


@torch.no_grad()
def predict(model,f,device,batch=256):
    model.eval();parts=[]
    for start in range(0,len(f['aux']),batch):
        ids=slice(start,start+batch)
        parts.append(torch.sigmoid(model(*tensor_batch(f,ids,device))).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0,dtype='float32')

