import json
import itertools
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", "--subject-hierarchy", help="Path to Subject Heading Hierarchy", dest="subject_heading_hierarchy", type=str)
parser.add_argument("-p", "--subject-path", help="Path to Subject Heading Path", dest="subject_heading_path", type=str)
parser.add_argument("-f", "--training-file", help="Path to training file", dest="training_file_dir", type=str)
parser.add_argument("-i", "--hierarchy-output-file", help="Path to hierarchy output file directory", dest="hierarchy_output_file_dir", type=str)
parser.add_argument("-o", "--index-output-file", help="Path to index output file directory", dest="index_output_file_dir", type=str)
parser.add_argument("-c", "--child-parent-output-file", help="Path to child-parent indexes output file directory", dest="child_parent_output_file_dir", type=str)
args = parser.parse_args()

with open(args.subject_heading_path,'r') as d:
    subject_paths = {}
    lines = d.readlines()
    for l in lines:
        subject_paths.update(json.loads(l.strip()))

with open(args.subject_heading_hierarchy, "r") as hierarchy_file:
    etd_hierarchy = json.load(hierarchy_file)
    
with open(args.training_file_dir, "r") as data_file:
    etd_jsons = []
    for line in data_file:
        line = line.strip("\n")
        if not line:
            continue
        etd_json = json.loads(line)
        abstract = etd_json['etdAbstract']
        labels = etd_json['lcsh']

        etd_jsons.append((abstract,labels))
        
lcsh = set(itertools.chain(*[l.keys() for _,l in etd_jsons]))
print(">>> Number of Subject Heading in training file: %d"%len(lcsh))

all_labels = [list(l.keys()) for _,l in etd_jsons]
all_labels = set(itertools.chain(*all_labels))
all_labels = set(itertools.chain(*[subject_paths[l] for l in all_labels]))
print(">>> Number of All Potential Subject Heading (through the paths) in trainin file: %d"%len(all_labels))

print(">>> Indexing...")
# groupby_level = {}
# for l,sh in etd_hierarchy.items():
#     intersect = list(lcsh.intersection(sh))
#     if intersect:
#         groupby_level[int(l)-1] = intersect
    
groupby_level = {}
for l,sh in etd_hierarchy.items():
    intersect = list(all_labels.intersection(sh))
    if intersect:
        groupby_level[int(l)-1] = intersect
    
class_index = {}
idx = 0
for l,sh in groupby_level.items():
    for i in sh:
        class_index[i] = idx
        idx += 1

child_parent_pairs = []
for i in class_index.keys():
    p = subject_paths[i]
    if len(p) > 1:
        child_parent_pairs.append((class_index[i],class_index[p[-2]]))
child_parent_pairs = set(child_parent_pairs)
        
print(">>> Writing index JSON file to %s"%(args.index_output_file_dir))
with open(args.index_output_file_dir,'w') as f:
    json.dump(class_index, f)

print(">>> Writing hierarchy JSON file to %s"%(args.hierarchy_output_file_dir))
with open(args.hierarchy_output_file_dir,'w') as f:
    json.dump(groupby_level, f)
    
print(">>> Writing child-parent indexes file to %s"%(args.child_parent_output_file_dir))
with open(args.child_parent_output_file_dir,'w') as f:
    for i,j in child_parent_pairs:
        f.write("%s,%s\n"%(i,j))