from typing import List, Dict, Iterable
import json
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
#from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField, ListField
from allennlp.data.fields.multilabel_field import MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# from multi_toxic_label.fields.multi_label_field import MultiLabelField

@DatasetReader.register("etd_title_abstract")
class EtdTitleAbstractReader(DatasetReader):
    """
    Reads a JSON-lines file containing title and abstract from ETD records
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 start_tokens: List[str] = ["<start>"], 
                 end_tokens: List[str] = ["<end>"]) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(start_tokens=start_tokens,end_tokens=end_tokens)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                etd_json = json.loads(line)
                title = etd_json['etdTitle']
                abstract = etd_json['etdAbstract']
                labels = etd_json['lcsh']

                yield self.text_to_instance(title, abstract, labels)

    @overrides
    def text_to_instance(self, title_text: str, abstract_text: str, labels: Dict[str, int] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_title_text = self._tokenizer.tokenize(title_text)
        title_text_field = TextField(tokenized_title_text, self._token_indexers)
        tokenized_abstract_text = self._tokenizer.tokenize(abstract_text)
        abstract_text_field = TextField(tokenized_abstract_text, self._token_indexers)
        fields = {'title_text': title_text_field, 'abstract_text': abstract_text_field}
        
        if labels is not None:
            fields['label'] = MultiLabelField([label for label,value in labels.items() if value == 1])
                
        return Instance(fields)

#     @classmethod
#     def from_params(cls, params: Params) -> 'EtdDatasetReader':
#         tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
#         token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
#         lazy = params.pop('lazy', False)
#         params.assert_empty(cls.__name__)
#         return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)