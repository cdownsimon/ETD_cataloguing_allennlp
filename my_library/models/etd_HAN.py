from typing import List, Dict, Optional, Union

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy

from my_library.metrics.roc_auc_score import RocAucScore
from my_library.metrics.hit_at_k import *
from my_library.metrics.macro_f1 import *
from my_library.metrics.precision_at_k import *
from my_library.models.etd_attention import AttentionEncoder, MultiHeadAttentionEncoder
from my_library.models.customized_allennlp.maxout import Maxout

@Model.register("etd_HAN")
class EtdHAN(Model):
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
                 text_field_embedder: TextFieldEmbedder,
                 word_encoder: Seq2SeqEncoder,
                 sentence_encoder: Seq2SeqEncoder,
                 classifier_feedforward: Union[FeedForward, Maxout],
                 attended_text_dropout: float  = 0.0,
                 bce_pos_weight: int = 10,
                 use_positional_encoding: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EtdHAN, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.word_encoder = word_encoder
        self.word_level_attention = FeedForward(word_encoder.get_output_dim(), 
                                                2, 
                                                [word_encoder.get_output_dim(), 1],
                                                [Activation.by_name("tanh")(), Activation.by_name("linear")()],
                                                [True, False])
        self.sentence_encoder = sentence_encoder
        self.sentence_level_attention = FeedForward(sentence_encoder.get_output_dim(), 
                                                    2, 
                                                    [sentence_encoder.get_output_dim(), 1],
                                                    [Activation.by_name("tanh")(), Activation.by_name("linear")()],
                                                    [True, False])
        self.classifier_feedforward = classifier_feedforward
        self.use_positional_encoding = use_positional_encoding

        self._dropout = torch.nn.Dropout(attended_text_dropout)
        
        if text_field_embedder.get_output_dim() != word_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the word_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            word_encoder.get_input_dim()))

        self.metrics = {
#             "roc_auc_score": RocAucScore()            
            "hit_5": HitAtK(5),
            "hit_10": HitAtK(10),
            "precision_5": PrecisionAtK(5),
            "precision_10": PrecisionAtK(10)
#             "hit_100": HitAtK(100),
#             "macro_measure": MacroF1Measure(top_k=5,num_label=self.num_classes)
        }
        
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones(self.num_classes)*bce_pos_weight)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                abstract_text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        embedded_abstract_text = self.text_field_embedder(abstract_text, num_wrapping_dims=1)
        abstract_text_mask = util.get_text_field_mask(abstract_text, num_wrapping_dims=1)
        
        num_instance, num_sentence, num_word, _ = embedded_abstract_text.size()
        embedded_abstract_text = embedded_abstract_text.view(num_instance*num_sentence, num_word, -1)
        abstract_text_mask = abstract_text_mask.view(num_instance*num_sentence, -1)

        if self.use_positional_encoding:
            embedded_abstract_text = util.add_positional_features(embedded_abstract_text)
        encoded_abstract_text = self.word_encoder(embedded_abstract_text, abstract_text_mask)
        
        word_attentive_weights = self.word_level_attention(encoded_abstract_text)
        word_attentive_weights = util.masked_softmax(word_attentive_weights.transpose(1,2).contiguous(), 
                                                     abstract_text_mask, dim=2).transpose(1,2).contiguous()
#         attended_sentences = util.weighted_sum(encoded_abstract_text,word_attentive_weights.squeeze(2))
        attended_sentences = torch.bmm(encoded_abstract_text.transpose(1,2), word_attentive_weights).squeeze()
        
        attended_sentences = attended_sentences.view(num_instance, num_sentence, -1)
        abstract_text_mask = abstract_text_mask.view(num_instance, num_sentence, -1).sum(2).ge(1).long()
        
        if self.use_positional_encoding:
            attended_sentences = util.add_positional_features(attended_sentences)
        attended_sentences = self.sentence_encoder(attended_sentences, abstract_text_mask)
        
        sentence_attentive_weights = self.sentence_level_attention(attended_sentences)
        sentence_attentive_weights = util.masked_softmax(sentence_attentive_weights.transpose(1,2).contiguous(), 
                                                         abstract_text_mask, dim=2).transpose(1,2).contiguous()
#         attended_abstract_text = util.weighted_sum(attended_sentences,sentence_attentive_weights.squeeze(2))
        attended_abstract_text = torch.bmm(attended_sentences.transpose(1,2), sentence_attentive_weights).squeeze()
        attended_abstract_text = self._dropout(attended_abstract_text)
        
        outputs = self.classifier_feedforward(attended_abstract_text)
        logits = torch.sigmoid(outputs)
        logits = logits.unsqueeze(0) if logits.dim() < 2 else logits
        output_dict = {'logits': logits}

        if label is not None:
            outputs = outputs.unsqueeze(0) if outputs.dim() < 2 else outputs
            loss = self.loss(outputs, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'EtdHAN':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        word_encoder = Seq2SeqEncoder.from_params(params.pop("word_encoder"))
        sentence_encoder = Seq2SeqEncoder.from_params(params.pop("sentence_encoder"))
        classifier_feedforward = params.pop("classifier_feedforward")
        if classifier_feedforward.pop('type') == 'feedforward':
            classifier_feedforward = FeedForward.from_params(classifier_feedforward)
        else:
            classifier_feedforward = Maxout.from_params(classifier_feedforward)
        use_positional_encoding = params.pop("use_positional_encoding", False)
        bce_pos_weight = params.pop_int("bce_pos_weight", 10)
        attended_text_dropout = params.pop_float("attended_text_dropout", 0.0)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   word_encoder=word_encoder,
                   sentence_encoder=sentence_encoder,
                   classifier_feedforward=classifier_feedforward,
                   attended_text_dropout=attended_text_dropout,
                   bce_pos_weight=bce_pos_weight,
                   use_positional_encoding=use_positional_encoding,
                   initializer=initializer,
                   regularizer=regularizer)