import sys
import json


source_data = json.load(open(sys.argv[1]))["data"]
nbest_data = json.load(open(sys.argv[2]))
def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    
    return False




def context_map(context):
    
    context_list = []

    index_map = []


    for i,c in enumerate(context):
        
        if _is_whitespace(c):
            prev_is_whitespace = True
        
        else:
            context_list.append(c)
            index_map.append(i)

    return "".join(context_list), index_map

for entry in source_data:
    for paragraph in entry["paragraphs"]:
        context_text = paragraph["context"]
        for qa in paragraph["qas"]:
            _id = qa["id"]
            nbest = nbest_data[_id]
            for result in nbest:
                start_index = result["start_index"]
                end_index = result["end_index"]

                if context_text[start_index:end_index] == result["text"]:
                    continue

                else:
                    prev_is_whitespace = True 

                    for i,c in enumerate(context_text):
                        if _is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if not prev_is_whitespace:
                                prev_is_whitespace = False
                                continue
                            if context_text[i:][start_index:end_index] == result["text"]:
                                start_index = i  + start_index
                                end_index = i  + end_index
                                assert context_text[start_index:end_index] == result["text"]
                                break
                            prev_is_whitespace = False

                if  context_text[start_index:end_index] == result["text"]:
                    result["start_index"] = start_index
                    result["end_index"] = end_index
                    
                    continue

                else:
                    prev_is_whitespace = True
                    answer = result["text"].replace(" ","")
                    for i,c in enumerate(context_text):
                        if _is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                context_new, index_map = context_map(context_text[i:])
                                if context_new[start_index:start_index + len(answer)] == answer:
                                    end_index = index_map[start_index + len(answer) - 1 ] + i + 1
                                    start_index = index_map[start_index] + i
                                    result["text"] = context_text[start_index:end_index]
                                    result["start_index"] = start_index
                                    result["end_index"] = end_index

                            prev_is_whitespace = False
                    
                    if context_text[start_index:end_index] == result["text"]:
                        continue
                    else:
                        print (result)

json.dump(nbest_data,open(sys.argv[3],"w"),ensure_ascii=False,indent = 4)
