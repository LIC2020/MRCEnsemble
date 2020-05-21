import json
import sys
import collections
import logging
import numpy as np
import os
from tqdm import tqdm 
import math
import re

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax
def calc_corr(prob1,prob2):
    return np.corrcoef([prob1,prob2])[1,0]

def ensemble_corr(file_path1,file_path2,filename):

    data1 = json.load(open(os.path.join(file_path1,filename)))
    data2 = json.load(open(os.path.join(file_path2,filename)))

    start_1 =   softmax(np.array(data1["start_logits"]))
    start_2 =   softmax(np.array(data2["start_logits"]))
    
    end_1 =   softmax(np.array(data1["end_logits"]))
    end_2 =   softmax(np.array(data2["end_logits"]))
    
    return calc_corr(start_1,start_2), calc_corr(end_1,end_2)

if __name__ == "__main__":
    #ensemble_data_files =  sys.argv[1:-1]
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    
    v1 = re.findall(r"\d\d_\d",file_path1)[0]
    v2 = re.findall(r"\d\d_\d",file_path2)[0]

    start_corrs = []
    end_corrs = []
    for filename in os.listdir(file_path1):
        s,e = ensemble_corr(file_path1,file_path2,filename)
        start_corrs.append(s)
        end_corrs.append(e)
    print (v1,"vs",v2,"\t",np.mean(start_corrs) + np.mean(end_corrs),"\t",np.mean(start_corrs),"\t",np.mean(end_corrs))
