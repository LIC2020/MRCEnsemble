import json
import sys
import collections
from transformers.tokenization_bert import BasicTokenizer
import logging
#logging = logging.getLogger(__name__)

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
def extract_answer(info):

    start_logits = info["start_logits"]
    end_logits = info["end_logits"]
    
    tokens = [start_logits[str(i)][1]  for i in range(len(start_logits))]

    start_indexes = _get_best_start_indexes(start_logits, 10)
    end_indexes = _get_best_end_indexes(end_logits, 10)

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )

    prelim_predictions = [] 

    for start_index in start_indexes:   
        for end_index in end_indexes:
            if int(end_index) < int(start_index):
                continue
            length = int(end_index) - int(start_index) + 1
            if length >30:
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
        if len(nbest) >= 10:                
            break
        tok_tokens = tokens[int(pred.start_index):int(pred.end_index)+1]
        orig_tokens = info["ori_tokens"][start_logits[pred.start_index][0]:start_logits[pred.end_index][0] + 1]
        
        tok_text = " ".join(tok_tokens)
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)
    
        final_text,start_index,end_index = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=True)
        break

        if final_text in seen_predictions:
            continue
        seen_predictions[final_text] = True
        
    return final_text

def ensemble_logits(file_list):

    data_list = [json.load(open(f)) for f in file_list]
    data_new = data_list[0]
    
    for qid in data_new.keys():
        for index in range(len(data_new[qid]["end_logits"])):
            if type(data_new[qid]["end_logits"][str(index)]) in (tuple,list):
                data_new[qid]["end_logits"][str(index)] = data_new[qid]["end_logits"][str(index)][2]
            else:
                break

    for qid in data_list[0].keys():
        
        for i in range(1,len(file_list)):
            assert data_list[i][qid]["ori_tokens"] == data_new[qid]["ori_tokens"] 
            for index in range(len(data_new[qid]["start_logits"])):
                assert  data_new[qid]["start_logits"][str(index)][1] == data_list[i][qid]["start_logits"][str(index)][1]
                data_new[qid]["start_logits"][str(index)][2] +=   data_list[i][qid]["start_logits"][str(index)][2]
                try:
                    data_new[qid]["end_logits"][str(index)] +=   data_list[i][qid]["end_logits"][str(index)]
                except:
                    data_new[qid]["end_logits"][str(index)] +=   data_list[i][qid]["end_logits"][str(index)][2]
    return data_new

if __name__ == "__main__":
    ensemble_data_files =  sys.argv[1:-1]
    data = ensemble_logits(ensemble_data_files)
    
    result = collections.OrderedDict()
    for qid,logit in data.items():
        result[qid] = extract_answer(logit)

    json.dump(result,open(sys.argv[-1],"w"),ensure_ascii=False,indent=4)
 
