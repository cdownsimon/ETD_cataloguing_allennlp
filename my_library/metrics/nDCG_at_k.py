from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

@Metric.register("nDCG_at_k")
class nDCGAtK(Metric):
    def __init__(self, k=5) -> None:
        self._k = k
        self._nDCG_at_k = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0

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
#         predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        predictions = predictions.detach()
        gold_labels = gold_labels.detach()
#         mask = mask.detach()

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)
        
        self._batch_size = batch_size
        self._predictions = predictions
        self._gold_labels = gold_labels
        self._ttl_size += batch_size

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated nDCG@k.
        """
        top_k = self._predictions.topk(self._k)[0][:,self._k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._gold_labels.size(1))).float()
        gold_labels = self._gold_labels.float()
        self._precision_at_k += ((gold_labels * predictions).sum(1) / self._k

        precision_at_k = self._precision_at_k / self._ttl_size
        
        if reset:
            self.reset()
        return precision_at_k.cpu().item()

    @overrides
    def reset(self):
        self._precision_at_k = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0