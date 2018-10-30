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
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy

from my_library.metrics.roc_auc_score import RocAucScore
from my_library.models.etd_attention import AttentionEncoder

@Model.register("etd_rnn")
class EtdRNN(Model):
    """
    This ``Model`` performs text classification for an academic paper.  We assume we're given a
    title and an abstract, and we predict some output label.
    The basic model structure: we'll embed the title and the abstract, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    title_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
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
                 abstract_text_encoder: Seq2SeqEncoder,
                 attention_encoder: AttentionEncoder,
                 classifier_feedforward: FeedForward,
                 use_positional_encoding: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EtdRNN, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.abstract_text_encoder = abstract_text_encoder
        self.attention_encoder = attention_encoder
        self.classifier_feedforward = classifier_feedforward
        self.use_positional_encoding = use_positional_encoding

        if text_field_embedder.get_output_dim() != abstract_text_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_text_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_text_encoder.get_input_dim()))

        self.metrics = {
#                 "roc_auc_score": RocAucScore()
        }
        
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones(self.num_classes)*10)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                abstract_text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_abstract_text = self.text_field_embedder(abstract_text)
        abstract_text_mask = util.get_text_field_mask(abstract_text)
        encoded_abstract_text = self.abstract_text_encoder(embedded_abstract_text, abstract_text_mask)

        if self.use_positional_encoding:
            encoded_abstract_text = util.add_positional_features(encoded_abstract_text)
        
        attended_abstract_text = self.attention_encoder(encoded_abstract_text, abstract_text_mask)
        outputs = self.classifier_feedforward(attended_abstract_text)
        logits = torch.sigmoid(outputs)
        logits = logits.unsqueeze(0) if len(logits.size()) < 2 else logits
        output_dict = {'logits': logits}

        if label is not None:
            outputs = outputs.unsqueeze(0) if len(outputs.size()) < 2 else outputs
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
        metric_dict = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'EtdRNN':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        abstract_text_encoder = Seq2SeqEncoder.from_params(params.pop("abstract_text_encoder"))
        attention_encoder = AttentionEncoder.from_params(params.pop("attention_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        use_positional_encoding = params.pop("use_positional_encoding", False)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   abstract_text_encoder=abstract_text_encoder,
                   attention_encoder=attention_encoder,
                   classifier_feedforward=classifier_feedforward,
                   use_positional_encoding=use_positional_encoding,
                   initializer=initializer,
                   regularizer=regularizer)
