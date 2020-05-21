import sys
import json
import os 
file_name = sys.argv[1]
new_directory = sys.argv[2]


def deal(data):
    
    f = data["id"]
    
    #start_logits = {}
    #end_logits = {}

    #tokens = data["tokens"].split()
    #assert len(tokens) == len(data["start_logits"])
    #for i,logit in enumerate(data["start_logits"]):
        
    #    start_logits[str(i)] = [data["span_id"][i],tokens[i],logit]
    #    end_logits[str(i)] = data["end_logits"][i]

    return f, {"start_logits":data["start_logits"],"end_logits":data["end_logits"],"ori_tokens":data["ori_tokens"],"tokens":data["tokens"],"span_id":data["span_id"]}

if __name__ == "__main__":
    directory = os.path.join(new_directory,file_name.split("/")[-1].split(".")[0])
    print (directory)
    try:    
        os.mkdir(directory)
    except:
        pass
    for line in open(file_name):
        line = line.strip()
        data = json.loads(line)
        f,data = deal(data)
        
        open(os.path.join(directory,f),"w").write(json.dumps(data))


