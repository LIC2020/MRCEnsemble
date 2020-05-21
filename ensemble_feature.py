# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
from collections import OrderedDict
import sys
from tqdm import tqdm
import os
import re

puc = set(',.，。！？~-=? ')
left_adj_words = ["大于","接近","超过","不超过","最高","最低","约为","约","大约"]
right_adj_words = ["左右","上下","以内","内","以上"]

def check_overlap_QA(question,answer_tuple,topN):
	'''
		这个函数为了过滤到一些和问题重叠度太高的答案。具体可以参考下面"fun1"中的例子
		question: str
		answer_tuple: ((answer_text, leijia_prob),  (answer_text, leijia_prob) ...)
		答案按照leijia概率从高到低排序排序
	'''

	top1_answer  = answer_tuple[0][0]

	for i,(answer, prob)  in enumerate(answer_tuple[1:topN]):

		diff_words = set(top1_answer) - set(answer) 
		if len(diff_words & set(question)) > 0.7 * len(diff_words) and  len(diff_words & set(question)) > 2:
			return answer

def Num_adj(answer_tuple,question,topN=3):
	'''
		避免一些数字的修饰词被忽略的情况。比如"大于5cm" vs "5cm"; "10min左右" vs "10min"
		使用func2进行测试
		answer_tuple: ((answer_text, leijia_prob),  (answer_text, leijia_prob) ...)
		答案按照leijia概率从高到低排序排序
	'''
	def is_number(text):
		if re.findall(r"[1-9]",text):
			return True
		if re.findall(r"[几千百万亿十]",text):
			return True
		return False

	top1_answer  = answer_tuple[0][0]

	if not is_number(top1_answer):
		return None

	for i,(answer, prob)  in enumerate(answer_tuple[1:topN]):
		for w in left_adj_words:
			if w + top1_answer == answer and len(set(w) & set(question))<1:
				return answer
		for w in right_adj_words:
			if top1_answer + w == answer and len(set(w) & set(question))<1:
				return answer


def multi_sub_answer(answer_tuple,question,topN=3):
	
	top1_answer  = answer_tuple[0][0].lower()

	question = question.lower()

	best_answer = top1_answer
	best_answer_topkens = re.split(r"[;、，,\s]",best_answer)

	for i,(answer, prob)  in enumerate(answer_tuple[1:topN]):
		
		answer = answer.lower()

		if best_answer not in answer:
			continue
		else:
			answer_tokens = re.split(r"[;、，,]",answer)
			flag = True
			for t in best_answer_topkens:
				if t not in answer_tokens:
					flag = False
			if not flag:
				continue
			
			if len(answer_tokens) <= len(best_answer_topkens):
				continue
			for t in answer_tokens:
				if t >= '0' and t <= '9':
					break
				if "。" in t:
					break
				if len(t) > 5:
					break
				if len(t.strip()) == 0:
					break
				if len(t) - max([len(w) for w in best_answer_topkens]) >=2:
					break
				if t not in best_answer and len(set(t) & set(question)) > 1:
					break
			else:
				if len(set(answer_tokens) - set(best_answer_topkens)) > 0:
					best_answer = answer

	return best_answer if best_answer!=top1_answer else None

            
def some_cases(char): # 哪些
    return char in set('、或和')

def has_puc(char): # 短答案在长答案里面，然后其中有标点符号的不要
    return char in set('。，！~')

question_set = set()
def keep_answer(answers, question, debug, which=False):
    if len(answers) == 1:
        return answers[0]
    answer1 = answers[0]
    answer2 = answers[1]
    
    answer = answer1
    if answer1 == ''.join(answer2.split()) and not any(map(lambda c:'\u4e00' <= c <= '\u9fa5',answer1)) and not any(map(lambda c:'\u4e00' <= c <= '\u9fa5',answer1)):
        answer = answer2
        if debug and question not in question_set:
            print('答案是英文 并且只差空格，取有空格的\t', question)
            print('\t', answer1, '###', answer2)
        question_set.add(question)
        
    elif answer1 in answer2 and '哪些' in question and any(map(lambda x: some_cases(x), answer2)):
        answer = answer2
        if len(answers) >= 3:
            answer3 = answers[2]
            if answer2 in answer3 and '哪些' in question and any(map(lambda x: some_cases(x), answer3)):
                answer = answer3
        if debug and question not in question_set:
            print('question有哪些，要长答案\t', question)
            print('\t', answers[:3], ' \tfinal: ', answer)
        question_set.add(question)
        
    elif which and answer1 in answer2 and '哪几' in question and any(map(lambda x: some_cases(x), answer2)):
        answer = answer2
        if len(answers) >= 3:
            answer3 = answers[2]
            if answer2 in answer3 and '哪几' in question and any(map(lambda x: some_cases(x), answer3)):
                answer = answer3
        if debug and question not in question_set:
            print('question有 哪几 ，要长答案\t', question)
            print('\t', answers[:3], ' \tfinal: ', answer)
        question_set.add(question)
    elif len(set(answer1) -  set('，。,.！!?？、=-')) == 0:
        answer = answer2
        if debug and question not in question_set:
            print('只有标点的不要\t', question)
            print('\t', answer1, '###', answer2)
        question_set.add(question)
    else:
        if question not in question_set and debug:
            print('未处理\t', question)
            print('\t', answer1, '###', answer2)
#     
    return answer
    
def my_num_contain_feature(text_list, question):
    pred_answers_lst = sorted(text_list.items(), key=lambda d: d[1], reverse=True)
    pred_answers_text_lst = [item[0] for item in pred_answers_lst]
    ret2_answer = Num_adj(pred_answers_lst, question)
    if ret2_answer is not None:
        final_answer = ret2_answer
#                 print('不能忽略数字的修饰词\t', question)
#                 print('\ttop2: ', pred_answers_text_lst[:2], ' final: ', final_answer)
    else:
        ret3_answer = multi_sub_answer(pred_answers_lst, question)
        if ret3_answer is not None:
            final_answer = ret3_answer
#                     print('选长答案\t', question)
#                     print('\ttop3: ', pred_answers_text_lst[:3], ' final: ', final_answer)
        else:
            final_answer = keep_answer(pred_answers_text_lst, question, False)
    assert isinstance(final_answer, str)
    return final_answer
