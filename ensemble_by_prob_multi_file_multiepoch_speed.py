import json
import sys
import collections
from transformers.tokenization_bert import BasicTokenizer
import logging
import numpy as np
import os
from tqdm import tqdm 
import math
import glob
#logging = logging.getLogger(__name__)
from functools import partial
from multiprocessing import Pool, cpu_count
from numba import jit

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=True)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print ("=="*10)
            print (tok_text)
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        #return orig_text,0,len(orig_text)
        return pred_text.replace(" ",""),0,len(orig_text)
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        #return orig_text,0,len(orig_text)
        return pred_text.replace(" ",""),0,len(orig_text)

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        #return orig_text,0,len(orig_text)
        return pred_text.replace(" ",""),0,len(orig_text)
    
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        #return orig_text,0,len(orig_text)
        return pred_text.replace(" ",""),0,len(orig_text)


    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text,orig_start_position,orig_end_position + 1
def _get_best_start_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(logits.items(), key=lambda x: x[1][2], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _get_best_end_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(logits.items(), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def extract_answer(info,topk=20,max_answer_length=30):

    start_logits = info["start_logits"]
    end_logits = info["end_logits"]
    
    tokens = [start_logits[str(i)][1]  for i in range(len(start_logits))]

    start_indexes = _get_best_start_indexes(start_logits,topk)
    end_indexes = _get_best_end_indexes(end_logits, topk)

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )

    prelim_predictions = [] 

    for start_index in start_indexes:   
        for end_index in end_indexes:
            if int(end_index) < int(start_index):
                continue
            length = int(end_index) - int(start_index) + 1
            if length >max_answer_length:
                continue
            prelim_predictions.append(
                 _PrelimPrediction(
                start_index=start_index,
                end_index=end_index,
                start_logit=start_logits[start_index][2],
                end_logit=end_logits[end_index],
                )
            )
    
    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)    
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit","start_index", "end_index"]
        )
    nbest = []
    seen_predictions = {}
    final_text = ""
    for pred in prelim_predictions:
        if len(nbest) >= topk:                
            break
        tok_tokens = tokens[int(pred.start_index):int(pred.end_index)+1]
        orig_tokens = info["ori_tokens"][start_logits[pred.start_index][0]:start_logits[pred.end_index][0] + 1]
        
        tok_text = " ".join(tok_tokens)
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)
    
        final_text,start_index,end_index = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=False)

        if final_text in seen_predictions:
            continue
        seen_predictions[final_text] = True

        nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,start_index = start_index,end_index = end_index))
    
    if not nbest:
        nbest.append(_NbestPrediction(text="", start_logit=-1e6, end_logit=-1e6,start_index = 0, end_index = 0))

    total_scores = [] 
    best_non_null_entry = None 


    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = [] 
    for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index

            nbest_json.append(output)

    assert len(nbest_json) >= 1
    assert best_non_null_entry is not None
    
    return best_non_null_entry.text, nbest_json


@jit
def softmax(x):
    x_row_max = np.max(x,axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

@jit("f8(f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
def esum(z):
    return np.sum(np.exp(z))


@jit("f8[:](f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
def softmax_bak(z):
    num = np.exp(z)
    s = num / esum(z)
    return s


def ensemble_logits(file_list,filename):

    data_list = [json.load(open(os.path.join(f,filename))) for f in file_list]
    data_new = data_list[0]
    
    data_new = data_list[0]
    data_new["start_logits"] = softmax(np.array(data_new["start_logits"]))
    data_new["end_logits"] = softmax(np.array(data_new["end_logits"]))

    for i in range(1,len(file_list)):
            assert data_list[i]["ori_tokens"] == data_new["ori_tokens"] 
            data_new["start_logits"] +=   softmax(np.array(data_list[i]["start_logits"]))
            data_new["end_logits"][str(index)] +=   softmax(np.array(data_list[i]["end_logits"][str(index)]))
    return data_new

def deal_one_file(filename,ensemble_data_files):
    data = ensemble_logits(file_list = ensemble_data_files, filename = filename)
    answer,nbest = extract_answer(data)
    return filename,answer,nbest

def multiprocess(ensemble_data_files,filenames,threads=5):
    with Pool(threads) as p:
        annotate_ = partial(
            deal_one_file,
            ensemble_data_files=ensemble_data_files,
        )   
        features = list(
            tqdm(
                p.imap(annotate_, filenames, chunksize=4),
                total=len(filenames),
                desc="generate answer and nbest",
            )   
        ) 
    result = collections.OrderedDict()
    nbest_result =  collections.OrderedDict()
    for filename, answer, nbest in tqdm(features):
         result[filename] = answer
         nbest_result[filename] = nbest
    return result, nbest_result
if __name__ == "__main__":
    ensemble_data_files =  []
    versions = ["14","17","21","22","23","25","26","27","28","29","33","34","35","36","37","38","39"]
    #versions = ["14","17","21","22","23","25","26","27","28","29","33","34","35","36","37","38"]
    #versions = ["14"]
    _type = "test1" 
    logit_path = "results/test1_logit/"
    results_path = "results/ensemble_test1/"
    EP = "EP4-5"
    for version in versions:    
        for epoch in range(4,6):
            v = "{}_{}*{}*".format(version,epoch,_type)
            try:
                filepath = glob.glob(os.path.join(logit_path,v))[0]
                ensemble_data_files.append(filepath)
            except:    
                print ("cant find:", v)
                pass
    #exit()
    print ( ensemble_data_files)
    result = collections.OrderedDict()
    nbest_result =  collections.OrderedDict()
    for filename in tqdm(os.listdir(ensemble_data_files[0])):
        data = ensemble_logits(ensemble_data_files,filename)
        result[filename],nbest_result[filename] = extract_answer(data,topk=20,max_answer_length=30)
    #result, nbest_result = multiprocess(ensemble_data_files,os.listdir(ensemble_data_files[0])) 
    name = "_".join(versions)+"_"+EP+"_nbest20_len30.json"
    json.dump(result,open(os.path.join(results_path,name),"w"),ensure_ascii=False,indent=4)
    json.dump(nbest_result,open(os.path.join(results_path,name) + '.nbest',"w"),ensure_ascii=False,indent=4) 
