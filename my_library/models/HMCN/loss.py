import torch
import torch.nn.functional as F

class HMCNLoss(torch.nn.Module):
    def __init__(self, num_classes, bce_pos_weight=10, childs_idx=None, parents_idx=None, penalty_lambda=0.1):
        super(HMCNLoss, self).__init__()
        self._local_loss = local_loss(num_classes, bce_pos_weight)
        self._global_loss = global_loss(num_classes, bce_pos_weight)
        if childs_idx is not None and parents_idx is not None:
            self._hierarchical_violation_regularization = hierarchical_violation_regularization(childs_idx=childs_idx, 
                                                                                                parents_idx=parents_idx, 
                                                                                                penalty_lambda=penalty_lambda)
        else:
            self._hierarchical_violation_regularization = lambda x:0
        
    def forward(self, local_logits, global_logits, local_labels, global_labels):
        # Here, logits and labels are concatenated probability vectors of each hierarchical level
        L = self._local_loss(local_logits, local_labels)
        G = self._global_loss(global_logits, global_labels)
        H = self._hierarchical_violation_regularization(local_logits)
        return L + G + H

class local_loss(torch.nn.Module):
    # Simply the usual BCE Loss
    def __init__(self, num_classes, bce_pos_weight):
        super(local_loss, self).__init__()
        m = (bce_pos_weight - 1) / (len(num_classes) - 1)
        c = 1 - m
        weights = [torch.ones(j)*(i*m+c) for i,j in enumerate(num_classes, 1)]
        self._bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.cat(weights))
        
    def forward(self, logits, labels):
        # Here, logits and labels are concatenated probability vectors of each hierarchical level
        return self._bce_loss(logits, labels)    

class global_loss(torch.nn.Module):
    # Simply the usual BCE Loss
    def __init__(self, num_classes, bce_pos_weight):
        super(global_loss, self).__init__()
        self._bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones(sum(num_classes))*bce_pos_weight)
        
    def forward(self, logits, labels):
        return self._bce_loss(logits, labels)

class hierarchical_violation_regularization(torch.nn.Module):
    def __init__(self, childs_idx, parents_idx, penalty_lambda = 0.1):
        super(hierarchical_violation_regularization, self).__init__()
        self._lambda = penalty_lambda
        self._childs_idx = childs_idx
        self._parents_idx = parents_idx
        
    def forward(self, logits):
        logits = torch.sigmoid(logits)
        return self._lambda * (F.relu(logits[:,self._childs_idx]-logits[:,self._parents_idx]).pow(2).sum(1).mean())