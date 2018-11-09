from typing import List, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

import numpy as np

@Predictor.register('etd-abstract-predictor')
class EtdAbstractPredictor(Predictor):
    """"Predictor wrapper for the EtdRNN/EtdTransformer"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        abstract_text = json_dict['etdAbstract']
        instance = self._dataset_reader.text_to_instance(abstract_text=abstract_text)

        # label_dict will be like {0: "toxic", 1: "severe_toxic", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["toxic", "severe_toxic", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        num_output = 5
        if 'num_lcsh' in inputs:
            if inputs['num_lcsh'] != '':
                num_output = int(inputs['num_lcsh'])
        # outputs = self._model.forward_on_instance(instance, cuda_device)
        outputs = self._model.forward_on_instance(instance)
        return_dict.update(outputs)
        first_n_labels = np.array(return_dict['all_labels'])
        first_n_logits = return_dict['logits']
        ind = np.argpartition(first_n_logits, -num_output)[-num_output:]
        ind = ind[np.argsort(first_n_logits[ind])]
        first_n_labels = first_n_labels[ind]
        first_n_logits = first_n_logits[ind]
        label_prob = dict(zip(first_n_labels,first_n_logits))
#         label_prob = dict(zip(return_dict['all_labels'],return_dict['logits']))
        sanitize_dict = sanitize(label_prob)
        first_n = sanitize_dict
#         first_n = {k:sanitize_dict[k] for k in sorted(sanitize_dict,
#                                                       key=lambda x:sanitize_dict[x],reverse=True)[:num_output]}
        return first_n
    
    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances, return_dicts = zip(*self._batch_json_to_instances(inputs))
        num_output = 5
        if 'num_lcsh' in inputs:
            if inputs['num_lcsh'] != '':
                num_output = int(inputs['num_lcsh'])
        # outputs = self._model.forward_on_instances(instances, cuda_device)
        outputs = self._model.forward_on_instances(instances)
        first_ns = []
        for output, return_dict in zip(outputs, return_dicts):
            return_dict.update(output)
            first_n_labels = np.array(return_dict['all_labels'])
            first_n_logits = return_dict['logits']
            ind = np.argpartition(first_n_logits, -num_output)[-num_output:]
            ind = ind[np.argsort(first_n_logits[ind])]
            first_n_labels = first_n_labels[ind]
            first_n_logits = first_n_logits[ind]
            label_prob = dict(zip(first_n_labels,first_n_logits))
#             label_prob = dict(zip(return_dict['all_labels'], return_dict['logits']))
            sanitize_dict = sanitize(label_prob)
#             first_n = {k:sanitize_dict[k] for k in sorted(sanitize_dict,
#                                                           key=lambda x:sanitize_dict[x],reverse=True)[:num_output]}
            first_ns.append(sanitize_dict)
        return first_ns
    
@Predictor.register('etd-title-abstract-predictor')
class EtdTitleAsbtractPredictor(Predictor):
    """"Predictor wrapper for the EtdBCN"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        title_text = json_dict['title_text']
        abstract_text = json_dict['abstract_text']
        instance = self._dataset_reader.text_to_instance(title_text=title_text,abstract_text=abstract_text)

        # label_dict will be like {0: "toxic", 1: "severe_toxic", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["toxic", "severe_toxic", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        if inputs['num_lcsh'] != '':
            num_output = int(inputs['num_lcsh'])
        else:
            num_output = 5
        # outputs = self._model.forward_on_instance(instance, cuda_device)
        outputs = self._model.forward_on_instance(instance)
        return_dict.update(outputs)
        label_prob = dict(zip(return_dict['all_labels'],return_dict['logits']))
        sanitize_dict = sanitize(label_prob)
        first_n = {k:sanitize_dict[k] for k in sorted(sanitize_dict,
                                                      key=lambda x:sanitize_dict[x],reverse=True)[:num_output]}
        return first_n
    
    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances, return_dicts = zip(*self._batch_json_to_instances(inputs))
        # outputs = self._model.forward_on_instances(instances, cuda_device)
        outputs = self._model.forward_on_instances(instances)
        label_probs = []
        for output, return_dict in zip(outputs, return_dicts):
            return_dict.update(output)
            label_prob = dict(zip(return_dict['all_labels'], return_dict['logits']))
            label_probs.append(label_prob)
        return sanitize(label_probs)
    