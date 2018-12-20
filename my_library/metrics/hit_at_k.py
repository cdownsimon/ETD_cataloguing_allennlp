from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("hit_at_5")
class HitAt5(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self, k=5) -> None:
        self._k = k
        self._hit_at_5 = 0.0
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
        The accumulated accuracy.
        """
        top_k = self._predictions.topk(self._k)[0][:,self._k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._gold_labels.size(1))).float()
        gold_labels = self._gold_labels.float()
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.cpu().item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0
        
@Metric.register("hit_at_10")        
class HitAt10(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._hit_at_5 = 0.0
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

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
        The accumulated accuracy.
        """
        k = 10
        top_k = self._predictions.topk(k)[0][:,k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._predictions.size(1))).float()
        gold_labels = self._gold_labels
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0
        
@Metric.register("hit_at_20")
class HitAt200(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._hit_at_5 = 0.0
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

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
        The accumulated accuracy.
        """
        k = 20
        top_k = self._predictions.topk(k)[0][:,k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._predictions.size(1))).float()
        gold_labels = self._gold_labels
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0
        
@Metric.register("hit_at_200")
class HitAt200(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._hit_at_5 = 0.0
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

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
        The accumulated accuracy.
        """
        k = 200
        top_k = self._predictions.topk(k)[0][:,k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._predictions.size(1))).float()
        gold_labels = self._gold_labels
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0
        
@Metric.register("hit_at_all")
class HitAtAll(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._hit_at_5 = 0.0
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

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
        The accumulated accuracy.
        """
        k = self._predictions.size(1)
        top_k = self._predictions.topk(k)[0][:,k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._predictions.size(1))).float()
        gold_labels = self._gold_labels
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0
        
@Metric.register("hit_at_k")
class HitAtK(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self, k=5) -> None:
        self._k = k
        self._hit_at_5 = 0.0
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
        The accumulated accuracy.
        """
        top_k = self._predictions.topk(self._k)[0][:,self._k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._gold_labels.size(1))).float()
        gold_labels = self._gold_labels.float()
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.cpu().item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0
        
@Metric.register("hit_at_k_cpu")
class HitAtKCPU(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self, k=5) -> None:
        self._k = k
        self._hit_at_5 = 0.0
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
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
#         predictions = predictions.detach()
#         gold_labels = gold_labels.detach()
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
        The accumulated accuracy.
        """
        top_k = self._predictions.topk(self._k)[0][:,self._k-1]
        predictions = torch.ge(self._predictions,top_k.unsqueeze(1).expand(self._batch_size,self._gold_labels.size(1))).float()
        gold_labels = self._gold_labels.float()
#         self._hit_at_5 += (((gold_labels + predictions) >= 2).sum(1).float()/gold_labels.sum(1).float()).sum()
        self._hit_at_5 += ((gold_labels * predictions).sum(1) / gold_labels.sum(1)).sum()

        hit_at_5 = self._hit_at_5 / self._ttl_size
        
        if reset:
            self.reset()
        return hit_at_5.item()

    @overrides
    def reset(self):
        self._hit_at_5 = 0.0
        self._batch_size = 0
        self._predictions = None
        self._gold_labels = None
        self._ttl_size = 0