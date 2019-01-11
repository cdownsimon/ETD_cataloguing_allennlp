import io
import tarfile
import zipfile
import re
import logging
import warnings
import itertools
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple

from overrides import overrides
import numpy
import torch
from torch.nn.functional import embedding
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import get_file_extension, cached_path
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("region_embedding")
class RegionEmbedding(TokenEmbedder):
    """
    Region Embedding implementation followed https://github.com/schelotto/Region_Embedding_Text_Classification_Pytorch
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 context_window: int,
                 mode: str = "max",
                 projection_dim: int = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super(RegionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.context_window = context_window
        self.mode = mode
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.output_dim = projection_dim or embedding_dim

        weights = []
        for _ in range(2 * context_window + 1):
            weight = torch.nn.Parameter(torch.FloatTensor(num_embeddings, embedding_dim), requires_grad=trainable)
            torch.nn.init.xavier_uniform_(weight)
            weights.append(weight)
        self.weights = torch.nn.ParameterList(weights)

        if self.padding_index is not None:
            for w in self.weights:
                w.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = inputs.size()
                                        
#         padding_ = torch.LongTensor([self.padding_index] * self.context_window).expand(original_size[0],self.context_window)
#         padding_ = inputs.new(padding_.numpy())
#         inputs_pad = torch.cat((padding_, inputs, padding_), dim=1)
        inputs_pad = torch.nn.functional.pad(inputs, pad=(self.context_window, self.context_window))
        
        original_inputs = inputs_pad
        if original_inputs.dim() > 2:
            inputs_pad = inputs_pad.view(-1, inputs_pad.size(-1))

        embedded = []
        for i, w in enumerate(self.weights):
            e = embedding(inputs_pad[:, i:(i + original_size[-1])], w,
                          max_norm=self.max_norm,
                          norm_type=self.norm_type,
                          scale_grad_by_freq=self.scale_grad_by_freq,
                          sparse=self.sparse)
            embedded.append(e)
        x_embed = torch.stack(embedded, dim=3)
        multiple = torch.stack([x_embed[:, :, :, 1] * x_embed[:, :, :, i] for i in range(2 * self.context_window + 1)], dim=3)

        if self.mode == "sum":
            local_context = torch.sum(multiple, dim=3)
        elif self.mode == "max":
            local_context = torch.max(multiple, dim=3)[0]
        else:
            local_context = torch.mean(multiple, dim=3)
                   
        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()) + [local_context.size(-1)]
            local_context = local_context.view(*view_args)

        if self._projection:
            projection = self._projection
            for _ in range(local_context.dim() - 2):
                projection = TimeDistributed(projection)
            local_context = projection(local_context)
            
        return local_context

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':  # type: ignore
        # pylint: disable=arguments-differ
        num_embeddings = params.pop_int('num_embeddings', None)
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        context_window = params.pop_int('context_window')
        mode = params.pop('mode', 'max')                      
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        params.assert_empty(cls.__name__)

        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   context_window=context_window,
                   mode=mode,
                   projection_dim=projection_dim,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse)