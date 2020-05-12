import sys
import json
import os
import numpy as np
from tqdm import tqdm
def softmax(x):
    x_row_max = np.max(x,axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def softmax_logits(file_dir,file_dir_new,filename):

    data = json.load(open(os.path.join(file_dir,filename)))

            
    logit = [data["start_logits"][str(index)][2] for index in range(len(data["start_logits"]))]
    logit = np.array(logit)
    probs = softmax(logit)

    for index in range(len(data["start_logits"])):
            data["start_logits"][str(index)][2] = probs[index]
            
    logit = [data["end_logits"][str(index)] for index in range(len(data["end_logits"]))]
            
    logit = np.array(logit)
    probs = softmax(logit) 
    for index in range(len(data["end_logits"])):
                data["end_logits"][str(index)] = probs[index]

    json.dump(data,open(os.path.join(file_dir_new,filename),"w"),ensure_ascii=False,indent=4)


if __name__ == "__main__":
    
    for filename in tqdm(os.listdir(sys.argv[1])):
        new_dir = os.path.join(sys.argv[2],sys.argv[1].split("/")[-1])
        try:
            os.mkdir(new_dir)
        except:
            pass
        softmax_logits(sys.argv[1],new_dir,filename)
