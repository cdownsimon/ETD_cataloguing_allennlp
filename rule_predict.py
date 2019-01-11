import json
import time
from tqdm import tqdm
from src.rule_predictor import RulePredictor
from collections import Counter
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from argparse import ArgumentParser

def predict(etd):
    if 'etdTitle' in etd:
        sentence = '%s %s'%(etd['etdTitle'], etd['etdAbstract'])
    else:
        sentence = etd['etdAbstract']
    p = rule_predictor.predict_k(sentence, 10)
    return p

parser = ArgumentParser()
parser.add_argument("-f", "--testing-file", help="Path to testing file directory", dest="test_file_dir", type=str)
parser.add_argument("-r", "--rules-file", help="Path to rules file directory", dest="rule_file_dir", type=str)
parser.add_argument("-o", "--output-file", help="Path to output file directory", dest="output_file_dir", type=str)
args = parser.parse_args()

print('>>> Loading testing file %s'%args.test_file_dir)
# Load testing and prediction files
etd_json = []
with open(args.test_file_dir,'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if not line:
            continue
        etd_json.append(json.loads(line))
        
print('>>> Loading rules file %s'%args.rule_file_dir)
# Load the rules
rule_predictor = RulePredictor()
rule_predictor.load(args.rule_file_dir)

print('>>> Predicting...')
# Perform prediction using the rules
predicted_results = []
before = time.time()
with multiprocessing.Pool(processes=None) as pool:
    for i in tqdm(pool.imap(predict, 
                            etd_json, 
                            chunksize=10), 
                  total=len(etd_json)):
        predicted_results.append(i)
after = time.time()

# predicted_results = []
# for etd in tqdm(etd_json):
#     if 'etdTitle' in etd:
#         sentence = '%s %s'%(etd['etdTitle'], etd['etdAbstract'])
#     else:
#         sentence = etd['etdAbstract']
#     p = rule_predictor.predict_k(sentence, 10)
#     predicted_results.append(p)
    
print('>>> Writing prediction to %s'%args.output_file_dir)
# Output the prediction to desired file
with open(args.output_file_dir,'w') as f:
    for p in predicted_results:
        f.write('%s\n'%json.dumps(p))
        
print('>>> Finished')