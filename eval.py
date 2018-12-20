import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--testing-file", help="Path to testing file directory", dest="test_file_dir", type=str)
parser.add_argument("-p", "--prediction-file", help="Path to prediction file directory", dest="prediction_file_dir", type=str)
args = parser.parse_args()

# Load testing and prediction files
etd_json = []
with open(args.test_file_dir,'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if not line:
            continue
        etd_json.append(json.loads(line))
        
etd_result = []
with open(args.prediction_file_dir,'r') as f:
    for line in f.readlines():
        line = line.strip("\n")
        if not line:
            continue
        etd_result.append(json.loads(line))    

# Calculate f1-measure
targeted_lcsh = []
for i in etd_json:
    targeted_lcsh += list(i['lcsh'].keys())
targeted_lcsh = set(targeted_lcsh)

TP = {i:0.0 for i in targeted_lcsh}
TN = {i:0.0 for i in targeted_lcsh}
FP = {i:0.0 for i in targeted_lcsh}
FN = {i:0.0 for i in targeted_lcsh}
preds = [[i for i,_ in sorted(pred.items(), key=lambda x:x[1], reverse=True)[:5]] for pred in etd_result]
for i in tqdm(targeted_lcsh):
    for true,pred in zip(etd_json,preds):
        t = true['lcsh'].keys()
        p = pred
#         p = [i for i,_ in sorted(pred.items(), key=lambda x:x[1], reverse=True)[:5]]
        if i in t and i in p:
            TP[i] += 1
        if i not in t and i not in p:
            TN[i] += 1
        if i not in t and i in p:
            FP[i] += 1
        if i in t and i not in p:
            FN[i] += 1
            
f_measure = {i:{} for i in targeted_lcsh}
for i in targeted_lcsh:
    f_measure[i]['precision'] = TP[i] / (TP[i] + FP[i] + 1e-13)
    f_measure[i]['recall'] = TP[i] / (TP[i] + FN[i] + 1e-13)
    f_measure[i]['f1'] = 2*f_measure[i]['precision']*f_measure[i]['recall'] / (f_measure[i]['precision']+f_measure[i]['recall']+1e-13)

macro_precision = sum(f_measure[i]['precision'] for i in targeted_lcsh)/len(targeted_lcsh)
macro_recall = sum(f_measure[i]['recall'] for i in targeted_lcsh)/len(targeted_lcsh)
macro_f1 = 2*macro_precision*macro_recall/(macro_precision+macro_recall+1e-13)
print(">>> Macro Precision: %.4f"%macro_precision)
print(">>> Maro Recall: %.4f"%macro_recall)
print(">>> Macro F1 measure: %.4f"%macro_f1)

# Calculate hit@5
hit_5 = 0.0
cnt = 0
for true,pred in zip(etd_json,etd_result):
    matched = 0
    t = true['lcsh'].keys()
#     p = pred.keys()
    p = [i for i,_ in sorted(pred.items(), key=lambda x:x[1], reverse=True)[:5]]
    if any(elem in t for elem in targeted_lcsh):
        for i in t:
            if i in p:
                matched += 1
        cnt += 1
    hit_5 += matched/len(t)
hit_5 = hit_5 / cnt
print(">>> Hit@5: %.4f"%hit_5)

# Calculate hit@10
hit_10 = 0.0
cnt = 0
for true,pred in zip(etd_json,etd_result):
    matched = 0
    t = true['lcsh'].keys()
    p = pred.keys()
    if any(elem in t for elem in targeted_lcsh):
        for i in t:
            if i in p:
                matched += 1
        cnt += 1
    hit_10 += matched/len(t)
hit_10 = hit_10 / cnt
print(">>> Hit@10: %.4f"%hit_10)
