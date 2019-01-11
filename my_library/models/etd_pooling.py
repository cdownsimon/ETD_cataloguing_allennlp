# follow https://arxiv.org/pdf/1708.00107.pdf section 5

from typing import Sequence, Union

import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.nn import util

class Pooling(torch.nn.Module):
    def __init__(self,
                 mode: str = 'max',
                 dropout_prob: float  = 0.0) -> None:

        super(Pooling, self).__init__()
        if mode == 'mean':
            self._pool = lambda x:x.mean(1)
        elif mode == 'sum':
            self._pool = lambda x:x.sum(1)
        else:
            self._pool = lambda x:x.max(1)[0]
            
        self._dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, inputs: torch.Tensor, inputs_mask: torch.Tensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self._dropout(self._pool(inputs))

    # Requires custom logic around the activations (the automatic `from_params`
    # method can't currently instatiate types like `Union[Activation, List[Activation]]`)
    @classmethod
    def from_params(cls, params: Params):
        mode = params.pop('mode', 'max')
        dropout_prob = params.pop_float('dropout_prob', 0.0)
        params.assert_empty(cls.__name__)
        return cls(mode=mode,
                   dropout_prob=dropout_prob)
    