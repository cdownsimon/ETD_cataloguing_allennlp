from typing import List, Dict, Optional, Union

import numpy
import json
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy

from my_library.metrics.roc_auc_score import RocAucScore
from my_library.metrics.hit_at_k import *
from my_library.metrics.macro_f1 import *
from my_library.metrics.precision_at_k import *
from my_library.models.etd_attention import AttentionEncoder
from my_library.models.customized_allennlp.maxout import Maxout
from my_library.models.HMCN.model import HMCNRecurrent
from my_library.models.HMCN.loss import HMCNLoss


@Model.register("etd_HMCN")
class EtdHMCN(Model):
    """
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    abstract_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 sh_hierarchy_dir: str,
                 text_field_embedder: TextFieldEmbedder,
                 abstract_text_encoder: Seq2SeqEncoder,
                 attention_encoder: AttentionEncoder,
                 local_globel_tradeoff: float = 0.5,
                 bce_pos_weight: int = 10,
                 use_positional_encoding: bool = False,
                 child_parent_index_pair_dir: str = None,
                 hv_penalty_lambda: float = 0.1,
                 hidden_states_dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EtdHMCN, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
#         self.num_classes = self.vocab.get_vocab_size("labels")
        self.abstract_text_encoder = abstract_text_encoder
        self.attention_encoder = attention_encoder
        self.local_globel_tradeoff = local_globel_tradeoff
        self.use_positional_encoding = use_positional_encoding
        
        with open(sh_hierarchy_dir,'r') as f:
            sh_hierarchy = json.load(f)
        # Use same dimension of encoders as HMCN dimension
        self.HMCN_recurrent = HMCNRecurrent([len(l) for _,l in sh_hierarchy.items()], 
                                            self.attention_encoder.get_output_dim(),
                                            self.attention_encoder.get_output_dim(),
                                            hidden_states_dropout=hidden_states_dropout)
        

        if text_field_embedder.get_output_dim() != abstract_text_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_text_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_text_encoder.get_input_dim()))

        self.metrics = {
#             "roc_auc_score": RocAucScore()            
            "hit_5": HitAtK(5),
            "hit_10": HitAtK(10),
            "precision_5": PrecisionAtK(5),
            "precision_10": PrecisionAtK(10)
#             "hit_100": HitAtK(100),
#             "macro_measure": MacroF1Measure(top_k=5,num_label=self.num_classes)
        }
        
        if child_parent_index_pair_dir:
            child_parent_pairs = []
            with open(child_parent_index_pair_dir,'r') as f:
                for l in f.readlines():
                    pair = l.strip().split(',')
                    child_parent_pairs.append((int(pair[0]),int(pair[1])))
            childs_idx, parents_idx = map(list, zip(*child_parent_pairs))
            self.loss = HMCNLoss(num_classes=[len(l) for _,l in sh_hierarchy.items()],
                                 bce_pos_weight=bce_pos_weight,
                                 childs_idx=childs_idx,
                                 parents_idx=parents_idx,
                                 penalty_lambda=hv_penalty_lambda)
        else:
            self.loss = HMCNLoss(num_classes=[len(l) for _,l in sh_hierarchy.items()], bce_pos_weight=bce_pos_weight)
        
#         self.loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones(self.num_classes)*bce_pos_weight)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                abstract_text: Dict[str, torch.LongTensor],
                local_label: torch.LongTensor = None, 
                global_label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        embedded_abstract_text = self.text_field_embedder(abstract_text)
        abstract_text_mask = util.get_text_field_mask(abstract_text)
        if self.use_positional_encoding:
            embedded_abstract_text = util.add_positional_features(embedded_abstract_text)
        encoded_abstract_text = self.abstract_text_encoder(embedded_abstract_text, abstract_text_mask)
        
        attended_abstract_text = self.attention_encoder(encoded_abstract_text, abstract_text_mask)
        local_outputs, global_outputs = self.HMCN_recurrent(attended_abstract_text)
        logits = self.local_globel_tradeoff * global_outputs + (1 - self.local_globel_tradeoff) * local_outputs
        logits = torch.sigmoid(logits)
        logits = logits.unsqueeze(0) if logits.dim() < 2 else logits
        output_dict = {'logits': logits}

        if local_label is not None and global_label is not None:
            local_outputs = local_outputs.unsqueeze(0) if local_outputs.dim() < 2 else local_outputs
            global_outputs = global_outputs.unsqueeze(0) if global_outputs.dim() < 2 else global_outputs
            loss = self.loss(local_outputs, global_outputs, local_label.squeeze(-1), global_label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, global_label.squeeze(-1))
            output_dict["loss"] = loss
            
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        # class_probabilities = output_dict['logits']
        # output_dict['class_probabilities'] = class_probabilities

        # predictions = class_probabilities.cpu().data.numpy()
        # labels = [list(self.vocab.get_index_to_token_vocabulary(namespace="labels").values()) for i in range(len(output_dict['logits']))]
        # output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         metric_dict = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        metric_dict = {}
        metric_dict['hit_5'] = self.metrics['hit_5'].get_metric(reset)
        metric_dict['hit_10'] = self.metrics['hit_10'].get_metric(reset)
#         metric_dict['hit_100'] = self.metrics['hit_100'].get_metric(reset)
        metric_dict['precision_5'] = self.metrics['precision_5'].get_metric(reset)
        metric_dict['precision_10'] = self.metrics['precision_10'].get_metric(reset)
#         macro_measure = self.metrics['macro_measure'].get_metric(reset)
#         metric_dict['mac_prec'] = macro_measure[0]
#         metric_dict['mac_rec'] = macro_measure[1]
#         metric_dict['mac_f1'] = macro_measure[2]
        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'EtdHMCN':
        sh_hierarchy_dir = params.pop("sh_hierarchy_dir")
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        abstract_text_encoder = Seq2SeqEncoder.from_params(params.pop("abstract_text_encoder"))
        attention_encoder = params.pop("attention_encoder")
        if attention_encoder.pop('type') == 'linear_attention':
            attention_encoder = AttentionEncoder.from_params(attention_encoder)
        else:
            attention_encoder = MultiHeadAttentionEncoder.from_params(attention_encoder)
        local_globel_tradeoff = params.pop_float("local_globel_tradeoff", 0.5)
        bce_pos_weight = params.pop_int("bce_pos_weight", 10)
        use_positional_encoding = params.pop("use_positional_encoding", False)
        child_parent_index_pair_dir = params.pop("child_parent_index_pair_dir", None)
        hv_penalty_lambda = params.pop_float("hv_penalty_lambda", 0.1)
        hidden_states_dropout = params.pop_float("hidden_states_dropout", 0.1)
        
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   sh_hierarchy_dir=sh_hierarchy_dir,
                   text_field_embedder=text_field_embedder,
                   abstract_text_encoder=abstract_text_encoder,
                   attention_encoder=attention_encoder,
                   local_globel_tradeoff=local_globel_tradeoff,
                   bce_pos_weight=bce_pos_weight,
                   use_positional_encoding=use_positional_encoding,
                   child_parent_index_pair_dir=child_parent_index_pair_dir,
                   hv_penalty_lambda=hv_penalty_lambda,
                   hidden_states_dropout=hidden_states_dropout,
                   initializer=initializer,
                   regularizer=regularizer)
