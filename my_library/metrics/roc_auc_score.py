from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
from sklearn.metrics import roc_auc_score


@Metric.register("roc_auc_score")
class RocAucScore(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._predictions = None
        self._gold_labels = None

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        # Get the data from the Variables.
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)


        self._predictions = predictions
        self._gold_labels = gold_labels

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        one_label_columns = [i for i,x in enumerate(self._gold_labels.sum(0)) if x == 0]
        if one_label_columns != []:
            accuracy_one = 1 - self._predictions[:,one_label_columns].mean(0)
            accuracy_one = accuracy_one.mean()
            two_label_columns = [i for i,x in enumerate(self._gold_labels.sum(0)) if x != 0]
            if two_label_columns != []:
                gold_labels = self._gold_labels[:,two_label_columns]
                predictions = self._predictions[:,two_label_columns]
                try:
                    accuracy_two = roc_auc_score(gold_labels,predictions)
                except ValueError as ve:
                    print(ve)
                    accuracy_two = 0
                accuracy = (accuracy_one + accuracy_two)/2
            else:
                accuracy = accuracy_one
        else:
            accuracy = roc_auc_score(self._gold_labels,self._predictions)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._predictions = None
        self._gold_labels = None