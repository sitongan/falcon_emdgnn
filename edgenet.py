import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import EdgeConv, global_mean_pool

class EdgeNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1, n_iters=10, aggr='add'):
        super(EdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim), hidden_dim),
                               nn.Sigmoid(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.Sigmoid()
        )
        self.n_iters = n_iters
        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*(hidden_dim+input_dim),output_dim),nn.Sigmoid())
        
        self.nodenetwork = EdgeConv(nn=convnn,aggr=aggr)
        
        self.outputnetwork = nn.Sequential(nn.Linear( input_dim + hidden_dim, 16))

    def forward(self, data):
        X = data.x
        H = self.inputnet(X)
        data.x = torch.cat([H,X],dim=-1)
        row,col = data.edge_index
        for i in range(self.n_iters):
            H = self.nodenetwork(data.x,data.edge_index)
            data.x = torch.cat([H,X],dim=-1)
        
        
        return self.outputnetwork(global_mean_pool(data.x, data.batch))