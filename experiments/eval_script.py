import re

from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
import torch
import pickle
import requests
import json
#from GENRE.genre_disamb import GenreDisamb
from prefix_trie import PrefixTrie
class EL_model():
    def __init__(self,model_path,max_seq_length):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",)
        self.max_seq_length=max_seq_length
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        #self.genre_disamb=GenreDisamb()
        self.trie=pickle.load(open("entity_trie_u.pkl","rb"))
    def _split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    def get_prefix_allowed_token_fn_ner(self,source):
        begin_seq=self.tokenizer.encode(" [START_ENT]")[:-1]
        end_seq = self.tokenizer.encode(" [END_ENT]")[:-1]
        source=[0]+self.tokenizer.encode(source)
        trigger_begin=self.tokenizer.encode(" [")[:-1][0]
        trigger_end = self.tokenizer.encode("]")[1]
        def prefix_allowed_token_fn(ar1, ar2):

                curr_tokens = ar2.tolist()
                if trigger_begin in curr_tokens:
                    last_an_start=len(curr_tokens) - 1 - curr_tokens[::-1].index(trigger_begin)
                else:
                    last_an_start=-1
                if trigger_end in curr_tokens:
                    last_an_end = len(curr_tokens) - 1 - curr_tokens[::-1].index(trigger_end)
                else:
                    last_an_end=-1
                # in_annotation_token
                if last_an_end<last_an_start:
                    seq_before=curr_tokens[0:last_an_start]
                    if trigger_begin in seq_before:
                        last_an_start_before = len(seq_before) - 1 - seq_before[::-1].index(trigger_begin)
                    else:
                        last_an_start_before = -1
                    if last_an_start_before==-1:
                        state="in_begin_sequence"
                    else:
                        annotation_seq_before=seq_before[last_an_start_before:last_an_end+1]
                        if annotation_seq_before==begin_seq:
                            state="in_end_sequence"
                        else:
                            state="in_begin_sequence"
                else:
                    state="copy"

                if state=="in_begin_sequence":
                    to_focus=curr_tokens[last_an_start:]
                    if len(to_focus)<len(begin_seq):
                        return begin_seq[len(to_focus)]
                    else:
                        state="copy"
                if state=="in_end_sequence":
                    to_focus=curr_tokens[last_an_start:]
                    if len(to_focus)<len(end_seq):
                        return end_seq[len(to_focus)]

                else:
                    ind_to_return=0
                    mod = "cp"
                    for i in range(len(curr_tokens)):
                        if curr_tokens[i]==trigger_begin:
                            mod="wait"
                        if mod=="cp":
                            ind_to_return+=1
                        if curr_tokens[i]==trigger_end:
                            mod="cp"
                    if ind_to_return>=len(source)-1:
                        return [1]
                    ret=source[ind_to_return]
                    return [ret,trigger_begin]

                    #return list(self.tokenizer.get_vocab().values())
        return prefix_allowed_token_fn

    def get_prefix_allowed_token_fn_candidates(self,mention_trie_map):
        trigger_begin=self.tokenizer.encode(" [")[:-1][0]
        trigger_end = self.tokenizer.encode("]")[1]
        def prefix_allowed_token_fn(ar1, ar2):

            curr_tokens = ar2.tolist()
            if trigger_begin in curr_tokens:
                last_an_start=len(curr_tokens) - 1 - curr_tokens[::-1].index(trigger_begin)

            else:
                last_an_start=-1
            if trigger_end in curr_tokens:
                last_an_end = len(curr_tokens) - 1 - curr_tokens[::-1].index(trigger_end)
            else:
                last_an_end=-1
            if last_an_end<last_an_start:
                curr_tokens_str = ",".join(str(el) for el in curr_tokens)
                begin_index=curr_tokens_str.rfind(str(trigger_begin))+len(str(trigger_begin))
                trie=None
                for key in mention_trie_map:
                    str_to_check=curr_tokens_str[begin_index+-len(key):begin_index]
                    if str_to_check==key:
                        ent_span=curr_tokens[last_an_start+1:]
                        trie=mention_trie_map[key]
                        seq=trie.get(ent_span)
                        return seq
            return list(self.tokenizer.get_vocab().values())

        return prefix_allowed_token_fn

    def get_pref_allowed_token_fn(self):
        target_seq = self.tokenizer.encode("[END_ENT] [")[:-1]
        end = ",".join(str(el) for el in self.tokenizer.encode(" ]")[:-1])
        find_token=784

        def prefix_allowed_token_fn(ar1, ar2):
            curr_state = ar2.tolist()
            print(curr_state)
            if find_token in curr_state:
                last_bk=len(curr_state) - 1 - curr_state[::-1].index(find_token)
                before=curr_state[last_bk-5:last_bk+1]

                if before==target_seq:
                    seq_to_check=curr_state[last_bk+1:]
                    string_sec=",".join(str(el)for el in seq_to_check)
                    if not end in string_sec:
                        return self.trie.get(seq_to_check)
                return list(self.tokenizer.get_vocab().values())
            else:
                return list(self.tokenizer.get_vocab().values())

        return prefix_allowed_token_fn
    def predict(self,seq,num_beams=6):
        i = self.tokenizer(seq, return_tensors="pt").input_ids
        out = self.model.generate(input_ids=i.to(self.device), max_length=512,num_beams=num_beams)
        out_sent = self.tokenizer.decode(out[0], skip_special_tokens=True)

        return out_sent
    def expand(self,ner_sequence):
        spans=re.findall(r"\[START_ENT\] ([^\]]+) \[END_ENT\]",ner_sequence)
        span_seq="\""+ ",".join(spans)+"\""
        span_to_exp={}
        for sp in spans:
            span_to_exp[sp]=None
        data = {"sentence": "\"" + ner_sequence + "\"","spans":span_seq}
        ans = requests.post("http://lm-test-dice.cs.upb.de:5000/annotate_ner/", data=json.dumps(data))
        jout=ans.content.decode()
        #jstring=str(json.loads(jout))
        jout=jout.replace("\n","")
        pattern=r'"span": "([^"]+)","entity_name": "([^"]+)"'
        result=re.search(pattern,jout)
        expansions=[]
        already_ext=set()
        while result is not None:
            #print(result)
            span=result.group(1)
            ent=result.group(2)
            if span!=result and not span in already_ext:
                already_ext.add(span)
                expansions.append({"span":span,"ent":ent,"annotation":result.group(0)})
            jout = jout.replace(result.group(0), "", 1)
            result=re.search(pattern,jout)
        for exp in expansions:
            span_to_exp[exp["span"]]=exp["ent"]
            #ner_sequence=ner_sequence.replace("[START_ENT] "+exp["span"],"[START_ENT] "+exp["ent"])
        #return ner_sequence,span_to_exp
        return span_to_exp
        '''
        try:
            span_map=ans.json()
            f
        except:
            print("failed to parse expansion json")
        '''
        #print(ans.content)

    def predict_ner(self,seq,join_outputs=True,num_beams=1):
        curr_seq_len = self.tokenizer(seq, return_tensors="pt").input_ids.size(1)
        num_splits = curr_seq_len // self.max_seq_length
        if num_splits > 0:
            sentences = seq.split(". ")
            splits = []
            chunks = list(self._split(sentences, num_splits + 1))
            for chunk in chunks:
                splits.append(". ".join(chunk))
            print(splits)
        else:
            splits = [seq]
        #Note: batching might be more efficient here instead of sending sequence one by one to the model
        predictions=[self.predict("Text to annotate: "+split+"[SEP]target_ner",num_beams=num_beams).replace("Text to annotate: ","") if split!=""else "" for split in splits]
        if join_outputs:
            return ". ".join(predictions)
        else:
            return predictions
    def replace_expansions(self,ner_sequence,out_sequence,spans):
        pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\] \[ ([^\]]+) \]"
        result = re.search(pattern, out_sequence)
        span_out_to_entitiy={}
        while result is not None:
            span_out_to_entitiy[result.group(1)]=result.group(2)
            out_sequence = out_sequence.replace(result.group(0), "", 1)
            result = re.search(pattern, out_sequence)
        '''
        for exp in expansions:
            out_sequence=out_sequence.replace("[START_ENT] "+exp["ent"],"[START_ENT] "+exp["span"])
        '''
        for span in spans.keys():
            if spans[span]!=None:
                s_key=spans[span]
            else:
                s_key=span
            if s_key in span_out_to_entitiy:
                ner_sequence=ner_sequence.replace("[START_ENT] "+span+" [END_ENT]","[START_ENT] "+span+" [END_ENT] [ "+span_out_to_entitiy[s_key]+" ]")
            else:
                ner_sequence = ner_sequence.replace("[START_ENT] " + span + " [END_ENT]",
                                                    span)
        return ner_sequence

    def predict_with_expansions(self,sequence,expansions,remove_expansions=True):
        mod_sequence=sequence
        for exp in expansions:
            if expansions[exp]!=None:
                mod_sequence=mod_sequence.replace("[START_ENT] "+exp,"[START_ENT] "+expansions[exp])
        output_mod=self.predict(mod_sequence + "[SEP]target_el")
        if remove_expansions:
            return self.replace_expansions(sequence,output_mod,expansions)
        else:
            return output_mod
    def predict_e2e(self,seq,expand=True):
        ner_out=self.predict_ner(seq,False,num_beams=4)
        if not expand:
            predictions = [self.predict(split + "[SEP]target_el") if split != "" else "" for split in ner_out]
        else:
            joint_ner=". ".join(ner_out)
            expansions=self.expand(joint_ner)
            predictions = [self.predict_with_expansions(split,expansions)  if split != "" else "" for split in ner_out]
        return ". ".join(predictions).replace("Text to annotate:","")
    def predict_disambiguation_only(self,seq,expand=True):
        curr_seq_len = self.tokenizer(seq, return_tensors="pt").input_ids.size(1)
        num_splits = curr_seq_len // self.max_seq_length
        if num_splits > 0:
            sentences = seq.split(". ")
            splits = []
            chunks = list(self._split(sentences, num_splits + 1))
            for chunk in chunks:
                splits.append(". ".join(chunk))
            print(splits)
        else:
            splits = [seq]
        if not expand:
            predictions = [self.predict(split + "[SEP]target_el") if split != "" else "" for split in splits]
            out=". ".join(predictions)
        else:
            expansions = self.expand(seq)
            predictions = [self.predict_with_expansions(split, expansions) if split != "" else "" for split in splits]
            output_mod= ". ".join(predictions)
            out=self.replace_expansions(seq,output_mod,expansions)
        return out

'''
import json

d=json.load(open("qald_10_resources.json","r"))
'''

test_str="Angelina her partner Brad and her father Jon"
EL_model=EL_model("../JointModel/aida-125ep",200)
res=EL_model.predict_e2e(test_str)
print(res)
#EL_model.expand("[START_ENT] Angelina [END_ENT] her partner [START_ENT] Brat [END_ENT] and her father [START_ENT] Jon [END_ENT]")

#EL_model.expand("[START_ENT] David [END_ENT] and [START_ENT] Victoria [END_ENT]")
#EL_model.expand("[START_ENT] Paderborn [END_ENT] was founded as a bishopric by [START_ENT] Charlemagne [END_ENT] in 795, although its official history began in 777 when [START_ENT] Charlemagne [END_ENT] built a castle near the [START_ENT] Paderborn [END_ENT] springs.(4) In 799 [START_ENT] Pope Leo III [END_ENT] fled his enemies in [START_ENT] Rome [END_ENT] and reached [START_ENT] Paderborn [END_ENT], where he met [START_ENT] Charlemagne [END_ENT], and stayed there for three months. It was during this time that it was decided that [START_ENT] Charlemagne [END_ENT] would be crowned emperor. [START_ENT] Charlemagne [END_ENT] reinstated [START_ENT] Leo [END_ENT] in [START_ENT] Rome [END_ENT] in 800 and was crowned as [START_ENT] Holy Roman Emperor [END_ENT] by [START_ENT] Leo [END_ENT] in return.")

