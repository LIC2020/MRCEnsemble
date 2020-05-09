# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
from collections import OrderedDict
import os
import itertools 
__author__ = "aitingliu@bupt.edu.cn"

path = '/home/ljp/data/lic2020/'

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
        }
}

v_dict = { 
    'output_data_join_utf8':
        {
            '14': 77.01477,
            '17': 77.51061,
            '22': 77.63886,
            '23': 77.44467,
            '25':78.18604,
            '26': 77.13503,
            '27':77.68519,
            '28':78.26716,
            '29':77.75663,
            '33':77.09937
        }
        }
dev_single_dict = {
    'output_data_join_utf8':
    {
        '17_5': 85.771,
        '17_4': 85.971,
        '17_3': 86.51,
        '17_2': 86.537,
        '17_1': 85.641,
        '22_5': 86.64,
        '22_4': 86.234,
        '22_3': 86.229,
        '22_2': 85.821,
        '22_1': 85.814,
        '14_5': 85.666,
        '14_4': 85.411,
        '14_3': 85.731,
        '14_2': 85.4,
        '14_1': 85.519,
        '23_5': 86.06,
        '23_4': 86.225,
        '23_3': 86.972,
        '23_2': 86.976,
        '23_1': 86.136,
        '25_5': 86.295,
        '25_4': 86.476,
        '25_3': 86.349,
        '25_2': 86.797,
        '25_1': 85.381,
        '26_5': 86.642,
        '26_4': 87.041,
        '26_3': 87.271,
        '26_2': 87.644,
        '26_1': 87.684,
        '27_5': 85.781,
        '27_4': 85.954,
        '27_3': 85.912,
        '27_2': 86.526,
        '27_1': 86.718,
        '28_5': 86.424,
        '28_4': 86.08,
        '28_3': 86.395,
        '28_2': 86.267,
        '28_1': 86.162,
        '29_5': 86.402,
        '29_4': 86.656,
        '29_3': 86.843,
        '29_2': 86.492,
        '29_1': 86.284,
        '33_5': 86.207,
        '33_4': 85.93,
        '33_3': 86.022,
        '33_2': 86.087,
        '33_1': 86.477,
        '34_5': 86.913,
        '34_4': 86.801,
        '34_3': 87.01,
        '34_2': 87.08,
        '34_1': 85.877,
        '35_5': 86.573,
        '35_4': 86.549,
        '35_3': 85.996,
        '35_2': 85.769,
        '35_1': 85.574,
        '36_5': 86.799,
        '36_4': 86.097,
        '36_3': 86.22,
        '36_2': 86.207,
        '36_1': 86.691,
        '37_5': 86.936,
        '37_4': 87.146,
        '37_3': 87.259,
        '37_2': 87.509,
        '37_1': 87.592,
    }
}
def ensemble_v1(pred_file, division, input_files):
    """
    text出现频次 和 probability 加权排序
    """
    if 8 in input_files:
        input('error! model 8 doesn\'t have the same key as other model! ')
    res_list = [json.load(open(path+"output_utf8/{}_{}_nbest_predictions_utf8.json".format(i, division))) for i in input_files]
    for item in res_list:
        print(len(item))
    res_json = {}

    for k in list(res_list[0].keys()):
        text_list = {}
        for i in range(len(res_list)):
            for j in range(len(res_list[i][k])):
                text = res_list[i][k][j]["text"]
                prob = res_list[i][k][j]["probability"]
                if not text_list.get(text):
                    # TODO(aitingliu): start_logit 和 end_logit也可以加进来，看看有没有效果增强
                    text_list[text] = 1 * prob  # 1 * prob
                else:
                    text_list[text] += 1 * prob  # 1 * prob
        # print(text_list)
        # print(sorted(text_list.items(), key=lambda d: d[1], reverse=True))
        res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]

    json.dump(res_json, open(pred_file, "w"), ensure_ascii=False, indent=4)

def ensemble_v4_multiple(res_list, pred_file, weight): 
    """
    res_list: 单模型输出的json
    对模型在test1上F1的权重w做下列计算作为输入的weight
    weight = (w-50) / (w.max())
    input_files = [2, 3, 6, 7]
    """
    res_json = {}
    try:
        for k in list(res_list[0].keys()):
            text_list = {}
            for i in range(len(res_list)):
                for j in range(len(res_list[i][k])):
                    text = res_list[i][k][j]["text"]
                    prob = res_list[i][k][j]["probability"]
                    if not text_list.get(text):
                        text_list[text] = weight[i] * prob  # 1 * prob
                    else:
                        text_list[text] += weight[i] * prob  # 1 * prob
            res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]
#         json.dump(res_json, open(pred_file, "w"), ensure_ascii=False, indent=4)
        return res_json
    except Exception as e1:
        print("err1: ", pred_file, e1)
        
def get_json_dict(division, model_lst=[]):
    
    if division == "dev":
        single_dict = dev_single_dict
    json_dict = {}
    if len(model_lst) != 0: # 只load指定文件
        for item in model_lst:
            model_type, param = item.split('/')
            obj = json.load(open(datapath+'{}_{}_nbest_predictions_utf8.json'.format(item, division)))
            try:
                json_dict[model_type+'/'+ param] = (obj, single_dict[model_type][param])
            except:
                json_dict[model_type+'/'+ param] = (obj, v_dict[model_type][param.split("_")[0]])
        return json_dict
    
    # load全部文件
    for model_type, param_dict in single_dict.items():
        for param, f1 in param_dict.items():
            obj = json.load(open(datapath+'{}/{}_{}_nbest_predictions_utf8.json'.format(model_type, param, division)))
            json_dict[model_type+'/'+ param] = (obj, f1)
    return json_dict


def ensemble_one(model_lst, division, version, weight_all=None):
    obj_dict = get_json_dict(division, model_lst)
    res_list = [obj_dict[key][0] for key in model_lst]
    if version == 'v4':
        f1_np = np.array([obj_dict[key][1] for key in model_lst])
        f1_np = (f1_np - weight_all) / f1_np.max()
    
    if version == 'v4':
        file_name = '_'.join(model_lst) + '_w' + str(weight_all) + '.json'
        pred_ans = ensemble_v4_multiple(res_list, file_name, f1_np)
    elif version == 'v1' and weight_all == None:
        file_name = '_'.join(model_lst) + '.json'
        pred_ans = ensemble_v1_multiple(res_list, file_name)
    else:
        input('version input error! version=', version)
    
    sim_name = []
    for item in model_lst:
        if 'xxlarge' in item:
            sim_name.append('xx'+item.split('/')[1])
        elif 'output_roberta_utf8' in item:
            sim_name.append('ro'+item.split('/')[1])
        elif 'output_data_join_utf8' in item:
            sim_name.append('dj'+item.split('/')[1].split("_")[0])
        else:
            sim_name.append(item.split('/')[1])
    sim_name = list(set(sim_name))
    if version == 'v4':
        file_name = '-'.join(sim_name) + '_w' + str(weight_all) + '.json'
    elif version == 'v1':
        file_name = '-'.join(sim_name) + '.json'
    else:
        input('version input error! version=', version)
    file_name = '{}_ensemble_{}_'.format(division, version) + file_name
    print('file_name: ', file_name)
    json.dump(pred_ans, open(os.path.join(export_datapath, file_name), "w"), ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    '''
    参数： group datapath export_datapath constant division
    '''
    #group = {'output_data_join_utf8': ["26_2", "14_3", "23_3", "26_5", "17_2", "23_4", "27_1", "27_2", "17_3", "23_5", "27_5", "22_5", "25_4", "28_5", "23_2", "26_1"]}  # 按照single_dict的模型类型，指定参与ensemble的模型名称。原始json文件名类似：14_3_test1_nbest_predictions_utf8.json
    Ls = ["14","17","22","23","25","26","27","28","29","33","34", "35","36", "37"]
    #L = ["14","17","22","23","25","26","27","28"]#之前看过的模型
    #L = ["17","23","25","26","27","28"] #大于78的模型
    for num in range(2,len(Ls) + 1):
        for L in itertools.combinations(Ls,num):
            group = {'output_data_join_utf8':[]}
            for i in range(4,6):
                for l in L:
                    group['output_data_join_utf8'] += [l+"_"+str(i)]
            datapath = './results/'  # 输出的ensemble json存放路径
            export_datapath = '/home_export/bzw/MRC/code/lic2020/results/ensemble_dev/EP4-5'# 新ensemble json保存目录
            constant = 71.5  # constant
            division = 'dev'  # test1/train/dev
            version = 'v4'  # ensemble version

            
            model_name_list = []
            for name in group:
                for sub in group[name]:
                    model_name_list.append(name+'/'+sub)
            model_name_list = sorted(model_name_list)
            print(model_name_list)
            ensemble_one(model_name_list, division, version, constant)
    

