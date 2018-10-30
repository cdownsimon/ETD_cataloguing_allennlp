from typing import Sequence, Union

import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder

from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention

class BiAttentionEncoder(torch.nn.Module):
    """
    A Bi-attentive classification network follow https://arxiv.org/pdf/1708.00107.pdf section 5
    """
    def __init__(self,
                 input_dim: int,
                 integrator: Seq2SeqEncoder,
                 integrator_dropout: float = 0.0,
                 combination: str = 'x,y') -> None:

        super(BiAttentionEncoder, self).__init__()
        
        self._self_attention = LinearMatrixAttention(input_dim, input_dim, combination)

        self._integrator = integrator
        self._integrator_dropout = torch.nn.Dropout(integrator_dropout)
        
        self._x_linear_layers = torch.nn.Linear(integrator.get_output_dim() , 1)
        self._y_linear_layers = torch.nn.Linear(integrator.get_output_dim(), 1)
        
        self._output_dim = input_dim
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, 
                inputs_x: torch.Tensor, inputs_y: torch.Tensor,
                inputs_x_mask: torch.Tensor, inputs_y_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # [B,T,N] * [B,N,L] -> [B,T,L]
        self_attentive_weights = self._self_attention(inputs_x, inputs_y)
        # [B,T,L]
#         self_attentive_weights_x = F.softmax(self_attentive_weights,dim=2)
        self_attentive_weights_x = util.masked_softmax(self_attentive_weights.transpose(1,2),inputs_x_mask,dim=2).transpose(1,2)
        # [B,L,T]
#         self_attentive_weights_y = F.softmax(self_attentive_weights.transpose(1,2),dim=2)
        self_attentive_weights_y = util.masked_softmax(self_attentive_weights,inputs_y_mask,dim=2).transpose(1,2)
        # [B,T,T] * [B,T,N] -> [B,T,N]
        # context = torch.bmm(self_attentive_weights.transpose(1,2), inputs)
        try:
            # [B,L,T] * [B,T,N] -> [B,L,N]
            context_x = torch.bmm(self_attentive_weights_x.transpose(1,2), inputs_x)
            # [B,T,L] * [B,L,N] -> [B,T,N]
            context_y = torch.bmm(self_attentive_weights_y.transpose(1,2), inputs_y)
        except Exception as e:
            print(self_attentive_weights_x.size(), inputs_x.size())
            print(self_attentive_weights_y.size(), inputs_y.size())
            raise e
            
        # [B,T,N] -> [B,T,3N]
        x_attend_y = torch.cat((inputs_x,inputs_x-context_y,inputs_x*context_y),dim=2)
        x_attend_y = self._integrator(x_attend_y, inputs_x_mask)
        # [B,L,N] -> [B,L,3N]
        y_attend_x = torch.cat((inputs_y,inputs_y-context_x,inputs_y*context_x),dim=2)
        y_attend_x = self._integrator(y_attend_x, inputs_y_mask)
        
        # [B,T,3N] -> [B,T,1]
        weights = self._x_linear_layers(x_attend_y)
#         weights = F.softmax(weights,dim=1)
        weights = util.masked_softmax(weights.transpose(1,2),inputs_x_mask,dim=2).transpose(1,2)
        # [B,3N,T] * [B,T,1] -> [B,3N]
        outputs_x = torch.bmm(x_attend_y.transpose(1,2), weights).squeeze()
        
        # [B,L,3N] -> [B,L,1]
        weights = self._y_linear_layers(y_attend_x)
#         weights = F.softmax(weights,dim=1)
        weights = util.masked_softmax(weights.transpose(1,2),inputs_y_mask,dim=2).transpose(1,2)
        # [B,3N,L] * [B,L,1] -> [B,3N]
        outputs_y = torch.bmm(y_attend_x.transpose(1,2), weights).squeeze()
        
        # [B,6N]
        outputs_x = outputs_x.unsqueeze(0) if len(outputs_x.size()) < 2 else outputs_x
        outputs_y = outputs_y.unsqueeze(0) if len(outputs_y.size()) < 2 else outputs_y
        outputs = torch.cat((outputs_x,outputs_y),dim=1)
        
        outputs = self._integrator_dropout(outputs)
        
        return outputs

    # Requires custom logic around the activations (the automatic `from_params`
    # method can't currently instatiate types like `Union[Activation, List[Activation]]`)
    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop_int('input_dim')
        integrator = Seq2SeqEncoder.from_params(params.pop("integrator"))
        integrator_dropout = params.pop_float('integrator_dropout', 0.0)
        combination = params.pop('combination', 'x,y')
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   integrator=integrator,
                   integrator_dropout=integrator_dropout,
                   combination=combination)