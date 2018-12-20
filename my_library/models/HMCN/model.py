from typing import Sequence, Union

import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder

from typing import Sequence, Union, List

import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder

class HMCNRecurrent(torch.nn.Module):
    def __init__(self, 
                 hierarchy_level: List[int],
                 input_dim: int,
                 hierarchy_recurrent_dim: int,
                 hidden_states_dropout: float = 0.1):
        super(HMCNRecurrent, self).__init__()
        
        self._hierarchy_level = hierarchy_level
        self._num_hierarchy_level = len(hierarchy_level)
        self._num_classes = sum(hierarchy_level)
        self._hierarchy_recurrent_dim = hierarchy_recurrent_dim
        self._hierarchy_recurrent = torch.nn.LSTM(input_dim, hierarchy_recurrent_dim, batch_first=True)
        
        # local weight dimension: C * (h*n)
#         self._hierarchy_feedforward_weight = Parameter(torch.Tensor(self._num_classes, 
#                                                                     self._num_hierarchy_level * hierarchy_recurrent_dim))
#         self._hierarchy_feedforward_bias = Parameter(torch.Tensor(self._num_classes))
#         t = []
#         for l,n in enumerate(hierarchy_level):
#             f = torch.zeros(n, hierarchy_recurrent_dim*self._num_hierarchy_level)
#             f[:,l*hierarchy_recurrent_dim:(l+1)*hierarchy_recurrent_dim] = 1
#             t.append(f)
#         self._hierarchy_feedforward_mask = Parameter(torch.cat(t), requires_grad=False)

        # local weight dimension: sum(C_i * n) for i in hierarchy level
        self._hierarchy_feedforward_weight = [Parameter(torch.Tensor(i, hierarchy_recurrent_dim)) for i in hierarchy_level]
        self._hierarchy_feedforward_weight = torch.nn.ParameterList(self._hierarchy_feedforward_weight)
        self._hierarchy_feedforward_bias = Parameter(torch.Tensor(self._num_classes))
        
        self._global_feedforward = torch.nn.Linear(input_dim + hierarchy_recurrent_dim, self._num_classes)
        
        self._hidden_states_dropout = torch.nn.Dropout(hidden_states_dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
         # local weight dimension: C * (h*n)
#         stdv = 1. / math.sqrt(self._hierarchy_feedforward_weight.size(1))
#         self._hierarchy_feedforward_weight.data.uniform_(-stdv, stdv)
        # local weight dimension: sum(C_i * n) for i in hierarchy level
        for w in self._hierarchy_feedforward_weight:
#             stdv = 1. / math.sqrt(w.size(1))
            stdv = math.sqrt(6 / (w.size(1) + 1))
            w.data.uniform_(-stdv, stdv)
        self._hierarchy_feedforward_bias.data.uniform_(-stdv, stdv)
        
    def forward(self, encoded_sentences):
        encoded_sentences = encoded_sentences.unsqueeze(0) if len(encoded_sentences.size()) < 2 else encoded_sentences
        x = encoded_sentences.unsqueeze(1).expand(-1, self._num_hierarchy_level, -1)
        recurrent_output, _ = self._hierarchy_recurrent(x)
        last_hidden_state = recurrent_output[:, -1, :]
        # local weight dimension: C * (h*n)
#         recurrent_output = recurrent_output.contiguous().view(-1, self._num_hierarchy_level*self._hierarchy_recurrent_dim)
#         local_feedforward_output = F.linear(recurrent_output, 
#                                            self._hierarchy_feedforward_weight*self._hierarchy_feedforward_mask, 
#                                            self._hierarchy_feedforward_bias)
        # local weight dimension: sum(C_i * n) for i in hierarchy level
        local_feedforward_output = torch.cat([F.linear(recurrent_output[:, i, :], 
                                                       self._hierarchy_feedforward_weight[i], 
                                                       None) for i in range(self._num_hierarchy_level)], 
                                             dim=1)
        local_feedforward_output = local_feedforward_output + self._hierarchy_feedforward_bias
        global_feedforward_output = self._global_feedforward(torch.cat([encoded_sentences,last_hidden_state],dim=1))
        
        local_feedforward_output = self._hidden_states_dropout(local_feedforward_output)
        global_feedforward_output = self._hidden_states_dropout(global_feedforward_output)
        
        return local_feedforward_output, global_feedforward_output
        
        