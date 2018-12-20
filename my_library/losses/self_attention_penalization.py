"""
A simple self attention mechanism follow 'A structured self-attentive sentence embedding'
"""
import torch
import numpy as np

class SelfAttentionPenalization(torch.nn.Module):
    def __init__(self):
        super(SelfAttentionPenalization, self).__init__()
        
    def forward(self, A):
        dim = A.size(1)
        return sum(torch.norm(i).pow(2) for i in torch.bmm(A, A.transpose(1,2).contiguous()) - A.new(np.eye(dim)))