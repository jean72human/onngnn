import torch
import torch.nn as nn

class GCNModel(nn.Module):
    def __init__(self, num_layers=2, sz_in=1433, sz_hid=32, sz_out=7):
        super().__init__()

        ### BEGIN SOLUTION
        self.hidden = nn.Linear(sz_in,sz_hid, bias=False)
        self.out = nn.Linear(sz_hid,sz_out, bias=False)
        self.ReLU = nn.ReLU()
        ### END SOLUTION

    def forward(self,  x, edges, node_graph_ind):
        """
        Args:
            fts: [N, 1433]
            adj: [N, N]

        Returns:
            new_fts: [N, 7]
        """
        adj = torch.zeros((x.size(0),x.size(0)), device=x.device)
        for ind in edges[:,:2]: 
            adj[ind[0],ind[1]]=1
            adj[ind[1],ind[0]]=1
        deg = adj.sum(axis=1, keepdim=True) # Degree of nodes, shape [N, 1]

        ### BEGIN SOLUTION
        x = self.hidden(1/deg * adj @ x) #self.ReLU(self.hidden(1/deg * adj @ x))
        x = self.out(1/deg * adj @ x)
        ### END SOLUTION
        return x
