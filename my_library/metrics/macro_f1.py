from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError
        
class MacroF1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, top_k: int, num_label: int) -> None:
        self._top_k = top_k
        self._num_label = num_label
#         self._true_positives = {i:0.0 for i in range(num_label)}
#         self._true_negatives = {i:0.0 for i in range(num_label)}
#         self._false_positives = {i:0.0 for i in range(num_label)}
#         self._false_negatives = {i:0.0 for i in range(num_label)}
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
#         predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        predictions = predictions.detach()
        gold_labels = gold_labels.detach()
#         mask = mask.detach()
    
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to MacroF1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
#         if mask is None:
#             mask = torch.ones_like(gold_labels)
#         mask = mask.float()
        
        top_k = predictions.topk(self._top_k)[0][:,self._top_k-1]
        batch_size = predictions.size(0)
        predictions = torch.ge(predictions,top_k.unsqueeze(1).expand(batch_size, gold_labels.size(1))).float()
        gold_labels = gold_labels.float()
        
#         for l in range(self._num_label):
#             gl = gold_labels[:,l]
#             pred = predictions[:,l]

#             # True Negatives: correct non-positive predictions.
#             self._true_negatives[l] += ((1-gl) * (1-pred)).sum()

#             # True Positives: correct positively labeled predictions.
#             self._true_positives[l] += (gl * pred).sum()

#             # False Negatives: incorrect negatively labeled predictions.
#             self._false_negatives[l] += (gl * (1-pred)).sum()

#             # False Positives: incorrect positively labeled predictions
#             self._false_positives[l] += ((1-gl) * pred).sum()

        # Tensor manner
#         if predictions.is_cuda and not self._true_negatives.is_cuda:
#             device = predictions.get_device()
#             self._true_negatives = self._true_negatives.cuda(device)
#             self._true_positives = self._true_positives.cuda(device)
#             self._false_negatives = self._false_negatives.cuda(device)
#             self._false_positives = self._false_positives.cuda(device)
            
        # True Negatives: correct non-positive predictions.
        self._true_negatives += ((1-gold_labels) * (1-predictions)).sum()

        # True Positives: correct positively labeled predictions.
        self._true_positives += (gold_labels * predictions).sum()

        # False Negatives: incorrect negatively labeled predictions.
        self._false_negatives += (gold_labels * (1-predictions)).sum()

        # False Positives: incorrect positively labeled predictions
        self._false_positives += ((1-gold_labels) * predictions).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
#         precisions = []
#         recalls = []
#         f1_measures = []
#         for l in range(self._num_label):
#             precision = float(self._true_positives[l]) / float(self._true_positives[l] + self._false_positives[l] + 1e-13)
#             recall = float(self._true_positives[l]) / float(self._true_positives[l] + self._false_negatives[l] + 1e-13)
#             f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

#             precisions.append(precision)
#             recalls.append(recall)
#             f1_measures.append(f1_measure)

        # Tensor manner
        precision = self._true_positives / (self._true_positives + self._false_positives + 1e-13)
        recall = self._true_positives / (self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        marco_f1_measure = 2. * ((macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-13))
            
        if reset:
            self.reset()
            
        return marco_f1_measure.cpu().item()

    def reset(self):
#         self._true_positives = {i:0.0 for i in range(self._num_label)}
#         self._true_negatives = {i:0.0 for i in range(self._num_label)}
#         self._false_positives = {i:0.0 for i in range(self._num_label)}
#         self._false_negatives = {i:0.0 for i in range(self._num_label)}
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
