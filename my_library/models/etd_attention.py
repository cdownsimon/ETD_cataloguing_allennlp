# follow https://arxiv.org/pdf/1708.00107.pdf section 5

from typing import Sequence, Union

import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.nn import util

from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention

class AttentionEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 combination: str = 'x,y',
                 dropout_prob  = 0.0) -> None:

        super(AttentionEncoder, self).__init__()
        
        self._self_attention = LinearMatrixAttention(input_dim, input_dim, combination)
        self._linear_layers = torch.nn.Linear(input_dim, 1)
        self._dropout = torch.nn.Dropout(dropout_prob)
        
        self._output_dim = input_dim
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # [B,T,N] * [B,N,T] -> [B,T,T]
        # self_attentive_weights = torch.bmm(inputs, inputs.transpose(1,2))
        # self_attentive_weights = F.softmax(self_attentive_weights,dim=1)
        self_attentive_weights = self._self_attention(inputs, inputs)
#         self_attentive_weights = F.softmax(self_attentive_weights,dim=1)
        self_attentive_weights =  util.masked_softmax(self_attentive_weights.transpose(1,2).contiguous(), 
                                                      inputs_mask, dim=2).transpose(1,2).contiguous()
        # [B,T,T] * [B,T,N] -> [B,T,N]
        # context = torch.bmm(self_attentive_weights.transpose(1,2), inputs)
        try:
            context = torch.bmm(self_attentive_weights, inputs)
        except Exception as e:
            print(self_attentive_weights.size(), inputs.size())
            raise e
        # [B,N,T] * [B,T,1] -> [B,N]
        weights = self._linear_layers(context)
#         weights = F.softmax(weights,dim=1)
        weights = util.masked_softmax(weights.transpose(1,2).contiguous(), inputs_mask, dim=2).transpose(1,2).contiguous()
        output = torch.bmm(context.transpose(1,2).contiguous(), weights).squeeze()
        output = self._dropout(output)
        
        return output

    # Requires custom logic around the activations (the automatic `from_params`
    # method can't currently instatiate types like `Union[Activation, List[Activation]]`)
    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop_int('input_dim')
        combination = params.pop('combination', 'x,y')
        dropout_prob = params.pop_float('dropout_prob', 0.0)
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   combination=combination,
                   dropout_prob=dropout_prob)
    
class SelfAttentionEncoder(torch.nn.Module):
    """
    A simple self attention mechanism follow 'A structured self-attentive sentence embedding'
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_hops: int, 
                 dropout_prob: float = 0.0) -> None:

        super(SelfAttentionEncoder, self).__init__()
        self._self_attention = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim, False), 
                                                   torch.nn.Tanh(),
                                                   torch.nn.Linear(hidden_dim, num_hops, False)
                                                  )
        self._dropout = torch.nn.Dropout(dropout_prob)
        
        self._output_dim = input_dim * num_hops
        self.input_dim = input_dim
        
    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim
    
    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # [B,T,N] -> [B,r,T]
        A = self._self_attention(inputs)
        A = util.masked_softmax(A.transpose(1,2).contiguous(), inputs_mask)
    
        # [B,r,T] x [B,T,N] -> [B,r,N] -> [B,r*N]
        output = torch.bmm(A,inputs).view(-1, self._output_dim)
        output = self._dropout(output)
    
        return output, A
    
    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop_int('input_dim')
        hidden_dim = params.pop_int('hidden_dim')
        num_hops = params.pop_int('num_hops')
        dropout_prob = params.pop_float('dropout_prob', 0.0)
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   num_hops=num_hops,
                   dropout_prob=dropout_prob)
    
class MultiHeadAttentionEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 dropout_prob  = 0.0) -> None:

        super(MultiHeadAttentionEncoder, self).__init__()
        
        self._self_attention = MultiHeadSelfAttention(1, input_dim, input_dim, input_dim, 1)
        self._dropout = torch.nn.Dropout(dropout_prob)
        
        self._output_dim = input_dim
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # [B,T,N] -> [B,T,1]
        self_attentive_weights = self._self_attention(inputs, inputs_mask)
        self_attentive_weights = util.masked_softmax(self_attentive_weights.transpose(1,2).contiguous(), inputs_mask)
        
        output = util.weighted_sum(inputs, self_attentive_weights).squeeze()
        output = self._dropout(output)
        
        return output

    # Requires custom logic around the activations (the automatic `from_params`
    # method can't currently instatiate types like `Union[Activation, List[Activation]]`)
    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop_int('input_dim')
        dropout_prob = params.pop_float('dropout_prob', 0.0)
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   dropout_prob=dropout_prob)