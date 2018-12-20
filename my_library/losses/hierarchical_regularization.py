# Hierarchical Regularization follow http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf
import torch

class HierarchicalRegularization(torch.nn.Module):
    def __init__(self, W, childs_idx, parents_idx):
        super(HierarchicalRegularization, self).__init__()
        self._W = W
        self._childs_idx = childs_idx
        self._parents_idx = parents_idx
        
    def forward(self):
        return torch.norm(self._W[self._childs_idx]-self._W[self._parents_idx],2,1).sum() / 2.0