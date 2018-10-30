# follow https://arxiv.org/pdf/1708.00107.pdf section 5

from typing import Sequence, Union

import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.nn import util

from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention

class AttentionEncoder(torch.nn.Module):
    """
    This ``Module`` is a feed-forward neural network, just a sequence of ``Linear`` layers with
    activation functions in between.
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    num_layers : ``int``
        The number of ``Linear`` layers to apply to the input.
    hidden_dims : ``Union[int, Sequence[int]]``
        The output dimension of each of the ``Linear`` layers.  If this is a single ``int``, we use
        it for all ``Linear`` layers.  If it is a ``Sequence[int]``, ``len(hidden_dims)`` must be
        ``num_layers``.
    activations : ``Union[Callable, Sequence[Callable]]``
        The activation function to use after each ``Linear`` layer.  If this is a single function,
        we use it after all ``Linear`` layers.  If it is a ``Sequence[Callable]``,
        ``len(activations)`` must be ``num_layers``.
    dropout : ``Union[float, Sequence[float]]``, optional
        If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
        versus ``Sequence[float]`` is the same as with other parameters.
    """
    def __init__(self,
                 input_dim: int,
                 combination: str = 'x,y') -> None:

        super(AttentionEncoder, self).__init__()
        
        self._self_attention = LinearMatrixAttention(input_dim, input_dim, combination)
        self._linear_layers = torch.nn.Linear(input_dim, 1)
        
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
        self_attentive_weights =  util.masked_softmax(self_attentive_weights.transpose(1,2), inputs_mask, dim=2).transpose(1,2)
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
        weights = util.masked_softmax(weights.transpose(1,2), inputs_mask, dim=2).transpose(1,2)
        output = torch.bmm(context.transpose(1,2), weights).squeeze()
        
        return output

    # Requires custom logic around the activations (the automatic `from_params`
    # method can't currently instatiate types like `Union[Activation, List[Activation]]`)
    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop_int('input_dim')
        combination = params.pop('combination', 'x,y')
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   combination=combination)