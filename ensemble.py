# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
from collections import OrderedDict
import os
import numpy as np
from tqdm import tqdm
from hparams import Hparams
from ensemble_feature import *

single_dict = {
    'output_albert_xxlarge_utf8': 
        {
            '2_1': 75.66131,
            '2_3': 74.78274,
            '3_2': 74.83024,
            '4_2': 74.71253
        },
    'output_data_join_utf8':
        {
            '1_2': 73.90175,
            '5_1': 75.89828,
            '6_2': 74.77744,
            '8_3': 76.59578,
            '14_3': 77.01477,
            '17_2': 77.51061,
            '17_3':78.87438,
            '21_5':77.42361,
            '22_5': 77.63886,
            '23_2': 77.44467,
            '23_3': 77.96674,
            '23_4':78.5541,
            '23_5': 78.48484,
            '25_4':78.18604,
            '26_1': 77.13503,
            '26_2':77.49921,
            '26_5':78.35698,
            '27_1':78.2233,
            '27_2':77.68519,
            '27_5':78.04175,
            '28_5':78.26716,
            '29_4':77.75663,
            '33_5':77.09937,
            '34_5':78.6033,
            '35_5':77.20208,
            '36_5':78.73412,
            '37_5':78.49709,
            '37_1':78.22473,
            '38_3':78.51518,
            '39_3':79.10322
        },
    'output_roberta_utf8':
        {
            '2_5': 75.2279,
            '3_2': 74.54158,
            '4_4': 75.02555,
            '5_4': 74.44874,
            '6_3': 74.87999
        },
    'output_utf8':
        {
            '0': 69.42017,
            '1': 70.98569,
            '2': 73.52785,
            '3': 73.26201,
            '4': 71.64866, 
            '5': 71.30978,
            '6': 72.71333,
            '7': 73.31789,
            '11':73.84171,
            '12':75.60135
        },
    'output_data_join_utf8_2':
        {
            '38_3':78.7334,
            '39_3':79.69988
        }
}

v_dict = { 
    'output_data_join_utf8':
        {
            '14': 77.01477,
            '17': 77.51061,
            '21': 76.99629,
            '22': 77.63886,
            '23': 77.44467,
            '25':78.18604,
            '26': 77.13503,
            '27':77.68519,
            '28':78.26716,
            '29':77.75663,
            '33':77.09937,
            '34':78.6033,
            '35':77.20208,
            '36':78.73412,
            '37':78.22473,
            '38':78.51518,
            '39':79.10322,
            '40':79.33913
        }
        }

def get_id_question(f):
    ref_obj = json.load(open(f,'r'))['data'][0]['paragraphs']
    id_question = {}
    for para in ref_obj:
        for qas in para['qas']:
            id_question[qas['id']] = qas['question']
    return id_question

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ensemble_v1(res_list, weight = None, use_weight = False, use_rule = True, agg_method = "sum",use_logit = False, topN = None,id_question =None):

    """
    对多个模型的TOPN个answer的概率进行累加,选得分最高的答案
    Input:
         res_list: List, 其中每个元素是单模型输出的nbest json格式数据 
         weight: List,每个模型的权重,默认为None
         use_weight: bool，是否使用权重，默认为False
         use_rule：是否使用额外的规则,默认为True
         agg_method:使用什么方式融合所有答案得分，"sum":对每个答案的得分求和；"avg":对每个答案得分求均值。默认为"sum"
         use_logit:是否使用logit作为每个答案的得分，True: 每个答案的得分为sigmoid(start_logit+end_logit); False：使用答案的概率作为得分.默认为False.
    Output:
        Dict，字典的key是问题id,value是问题对应的答案      
    """
    if use_weight and weight is None:
        raise ValueError("You set use_weight = True, please feed the weight list")
    
    if not use_weight:
        use_weight = [1] * len(res_list) 

    res_json = {}
    for k in tqdm(list(res_list[0].keys())):
        text_list = {}
        for i in range(len(res_list)):
            for j in range(len(res_list[i][k])):
                text = res_list[i][k][j]["text"]
                
                if topN is not None and j>=topN:
                    score = 0
                elif use_logit:
                    score =  sigmoid(res_list[i][k][j]["start_logit"] + res_list[i][k][j]["end_logit"])
                else:
                    score = res_list[i][k][j]["probability"]
                
                if not text_list.get(text):
                    text_list[text] = [weight[i] * score]
                else:
                    text_list[text] += [weight[i] * score]  
            for text in text_list.keys():
                if agg_method.lower() == "sum":
                    text_list[text] = np.sum(text_list[text])
                elif agg_method.lower() == "avg":
                    text_list[text] = np.mean(text_list[text])
                else:
                    raise ValueError("{} is illegal, please input correct aggregate method.".format(agg_method))

        res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]
        if use_rule:
            res_json[k] = my_num_contain_feature(text_list, id_question[k])
    
    return res_json

def ensemble_v2(res_list, weight = None, use_weight = False, use_rule = True, agg_method = "sum",use_decay = False,decay_rate=None, topN = None,id_question = None):
    """
    对多个模型的TOPN个answer的概率进行累加,选得分最高的答案
    Input:
         res_list: List, 其中每个元素是单模型输出的nbest json格式数据 
         weight: List,每个模型的权重,默认为None
         use_weight: bool，是否使用权重，默认为False
         use_rule：是否使用额外的规则,默认为True
         agg_method:使用什么方式融合所有答案得分，"sum":对每个答案的得分求和；"avg":对每个答案得分求均值。默认为"sum"
         use_logit:是否使用logit作为每个答案的得分，True: 每个答案的得分为sigmoid(start_logit+end_logit); False：使用答案的概率作为得分.默认为False.
    Output:
        Dict，字典的key是问题id,value是问题对应的答案      
    """
    if use_decay and decay_rate is None:
        raise ValueError("You set use_decay = True, please feed the decay list")
    
    if use_weight and weight is None:
        raise ValueError("You set use_weight = True, please feed the weight list")
    
    if not use_weight:
        use_weight = [1] * len(res_list) 
    
    res_json = {}
    for k in tqdm(list(res_list[0].keys())):
        text_list = {}
        for i in range(len(res_list)):
            for j in range(len(res_list[i][k])):
                text = res_list[i][k][j]["text"]
                
                if topN is not None and j>=topN:
                    score = 0
                elif use_decay:
                    score =  10 * (deacy**j)
                else:
                    score = 10 - j 
                
                if not text_list.get(text):
                    text_list[text] = [weight[i] * score]
                else:
                    text_list[text] += [weight[i] * score]  
            for text in text_list.keys():
                if agg_method.lower() == "sum":
                    text_list[text] = np.sum(text_list[text])
                elif agg_method.lower() == "avg":
                    text_list[text] = np.mean(text_list[text])
                else:
                    raise ValueError("{} is illegal, please input correct aggregate method.".format(agg_method))

        res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]
        if use_rule:
            res_json[k] = my_num_contain_feature(text_list, id_question[k])
    
    return res_json


def get_json_dict(args,model_lst=[]):
    json_dict = {}
    if len(model_lst) != 0: # 只load指定文件
        for item in model_lst:
            model_type, param = item.split('/')
            #每个模型的结果命名方式如下，可以根据需要自己定义
            obj = json.load(open(args.nbest_dir+'{}_{}_nbest_predictions_utf8.json'.format(item, args.division)))
            if not args.use_weight: #这类集成方式不需要权重
                json_dict[model_type+'/'+ param] = (obj, 0)
            try:
                json_dict[model_type+'/'+ param] = (obj, single_dict[model_type][param]) 
            except:
                json_dict[model_type+'/'+ param] = (obj, v_dict[model_type][param.split("_")[0]])
        return json_dict
    

def ensemble(model_lst, args):
    #首先读取文件，这个比较耗时
    obj_dict = get_json_dict(args, model_lst)

    res_list = [obj_dict[key][0] for key in model_lst]
    
    #获取每个id对应的文本，在使用规则时需要用到
    if args.use_rule:
        id_question = get_id_question(args.question_path)
    else:
        id_question = None
    
    f1_np  = None
    if args.use_weight:    
        f1_np = np.array([obj_dict[key][1] for key in model_lst])
        f1_np = (f1_np - args.weight_constant) / f1_np.max()
    
    if args.version == 'v1':
        #使用nbest输出的答案得分
        pred_ans = ensemble_v1(res_list,weight = f1_np,
                                use_weight = args.use_weight, use_rule = args.use_rule, 
                                agg_method = args.agg_method, use_logit = args.use_logit,
                                topN = args.topN,id_question = id_question)
    elif args.version == 'v2':
        #根据nbest答案的排序为每个答案赋予分值
        pred_ans = ensemble_v2(res_list,  weight = f1_np,
                                use_weight = args.use_weight, use_rule = args.use_rule, 
                                agg_method = args.agg_method, use_decay = args.use_decay,
                                topN = args.topN, decay = args.decay_rate,id_question = id_question)
    else:
        input('version input error! version=', version)
    
    json.dump(pred_ans, open(args.output_path, "w"), ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    '''
    参数： group datapath export_datapath constant division
    '''
    
    #使用旧模型的epoch 3-4 结果
    Ls = ["14","17","21","22","23","25","26","27","28","29","33","34", "35","36", "37"] 
    group = {'output_data_join_utf8':[]}
    for i in range(3,5):
        for l in Ls:
            group['output_data_join_utf8'] += [l+"_"+str(i)]
   
    #使用新模型的epoch 2-3 结果
    #Ls = ["38","39","40","43","14","17","21","22","23","26","27","28","29","33","35","36", "37"]
    #group['output_data_join_utf8_2'] = []
    #for i in range(2,4):
    #    for l in Ls:
    #        group['output_data_join_utf8_2'] += [l+"_"+str(i)]
    
    
    hparams = Hparams()
    args = hparams.parser.parse_args()
    print (args) 
    
    # 获取所有模型名称列表，并显示。用于运行时检查是否有误
    model_name_list = []
    for name in group:
        for sub in group[name]:
            model_name_list.append(name+'/'+sub)
    model_name_list = sorted(model_name_list)
    print(model_name_list)

    #对结果进行ensemble
    ensemble(model_name_list, args)
    
