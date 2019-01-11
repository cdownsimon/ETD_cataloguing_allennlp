from collections import Counter

import spacy
from tqdm import tqdm

class RulePredictor():
    def __init__(self):
        self.spacy_en = spacy.load('en_core_web_sm')
        self.rules = None
        
    def load(self, rule_path):
        self.rules = []
        with open(rule_path,'r') as f:
            for l in tqdm(f.readlines()):
                w,p = l.rstrip().split(' => ')
                self.rules.append((w.split(' ^ '),p))
        
    def _tokenize(self, sentence):
        return [token.text for token in self.spacy_en(sentence)]
    
    def _make_prediction(self, sentence):
        if type(sentence) == str:
            sentence = self._tokenize(sentence)
        predicted = []
        class_priority = []
        for cond,l in self.rules:
            if cond[0] == '':
                predicted.append(l)
                class_priority.append(l)
                continue
            for w in cond:
                not_fulfill = False
                if w not in sentence:
                    not_fulfill = True
                    break;
            if not not_fulfill:
                predicted.append(l)
        
        return predicted, class_priority
    
    def predict(self, sentence):
        label, _ = self._make_prediction(sentence)
        return label
    
    def predict_first(self, sentence):
        label, _ = self._make_prediction(sentence)
        return label[0]
    
    def predict_vote(self, sentence):
        predicted, class_priority = self._make_prediction(sentence)
        counter = Counter(predicted)
        majority = 0
        label = None
        for c in class_priority:
            if counter[c] > majority:
                majority = counter[c]
                label = c
        return label
    
    def predict_k(self, sentence, k=5):
        predicted, class_priority = self._make_prediction(sentence)
        counter = Counter(predicted)
        return dict(counter.most_common(k))
        