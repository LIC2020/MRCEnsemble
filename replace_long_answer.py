import sys
from collections import OrderedDict
import json
from transformers.tokenization_bert import BasicTokenizer
tokenizer = BasicTokenizer(do_lower_case=True)

data1 = json.load(open(sys.argv[1]),object_pairs_hook=OrderedDict)
data2 = json.load(open(sys.argv[2]))

for key,value in data1.items():
    value = tokenizer.tokenize(value)
    if len(value) >30 and data2[key] != "" and  data2[key] != "empty":
        data1[key] = data2[key]

json.dump(data1,open(sys.argv[3],"w"),ensure_ascii=False,indent=4)
