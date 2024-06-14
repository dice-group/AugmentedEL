import re

from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
import torch
import pickle
import requests
import json
from LLMservice import Local_service
#from GENRE.genre_disamb import GenreDisamb
class EL_model():
    def __init__(self,model_path,max_seq_length,apply_llm_ner=True):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",)
        self.max_seq_length=max_seq_length
        self.lm_ws=Local_service("llama2:70b")
        self.device="cuda:1" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.apply_llm_ner=apply_llm_ner
        #self.genre_disamb=GenreDisamb()
        #self.trie=pickle.load(open("../data/entity_trie_u.pkl","rb"))
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
    def expand(self,ner_sequence,expand_end_to_end=False):
        spans=re.findall(r"\[START_ENT\] ([^\]]+) \[END_ENT\]",ner_sequence)
        span_seq="\""+ ",".join(spans)+"\""
        span_to_exp={}
        for sp in spans:
            span_to_exp[sp]=None
        #data = {"sentence": "\"" + ner_sequence + "\"","spans":span_seq}
        #ans = requests.post("http://localhost:6000/annotate_ner/", data=json.dumps(data))
        jout = self.lm_ws.run_expansion_request(ner_sequence,span_seq)
        print(jout)
        #jout=ans.content.decode()
        #jstring=str(json.loads(jout))
        jout=jout.replace("\n","")
        #pattern=r'"span": "([^"]+)","entity_name": "([^"]+)"'
        print(jout)
        pattern = r'"span":\s?"([^"]+)",\s?"entity_name":\s?"([^"]+)"'
        result=re.search(pattern,jout)
        expansions=[]
        already_ext=set()
        while result is not None:
            #print(result)
            span=result.group(1)
            ent=result.group(2)
            print(span)
            print(ent)
            if span!=result and not span in already_ext:
                already_ext.add(span)
                expansions.append({"span":span,"ent":ent,"annotation":result.group(0)})
            jout = jout.replace(result.group(0), "", 1)
            result=re.search(pattern,jout)
        for exp in expansions:
            span_to_exp[exp["span"]]=exp["ent"]
            #ner_sequence=ner_sequence.replace("[START_ENT] "+exp["span"],"[START_ENT] "+exp["ent"])
        #return ner_sequence,span_to_exp
        print(span_to_exp)
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

        if self.apply_llm_ner:
            llm_out=self.lm_ws.run_ner_request(seq)
            pattern = r'"([^"]+)"[,\]]'
            result = re.search(pattern, llm_out)
            entities = []
            already_ext = set()
            while result is not None:
                # print(result)
                span = result.group(1)
                if span != result and not span in already_ext:
                    already_ext.add(span)
                    entities.append(span)
                llm_out = llm_out.replace(result.group(0), "", 1)
                result = re.search(pattern, llm_out)
            print(entities)
            entities.sort(key=len,reverse=True)
            predictions = [self.predict("Text to annotate: " + split + "[SEP]target_ner", num_beams=num_beams).replace(
                "Text to annotate: ", "") if split != "" else "" for split in splits]
            preditions_update=[]
            for pred_seq in predictions:
                pred_update=pred_seq
                last_ind=0
                if "[START_ENT]"in pred_update:
                    cur_str=pred_update[last_ind:pred_seq.index("[START_ENT]")]
                else:
                    cur_str = pred_update

                i=0
                updated=True
                while updated:
                    while i <len(entities):
                        if entities[i] in cur_str:
                            str_up=cur_str.replace(entities[i],"[START_ENT] "+entities[i]+" [END_ENT]")
                            pred_update=pred_update.replace(cur_str,str_up)
                            cur_str = pred_update[last_ind:pred_update.index("[START_ENT]",last_ind)]
                            i=0
                        else:
                            i=i+1
                    ind_up=last_ind+len(cur_str)
                    if "[END_ENT]" in pred_update[ind_up:]:
                        last_ind=pred_update.index("[END_ENT]",ind_up)+len("[END_ENT]")
                        if "[START_ENT]" in pred_update[last_ind:]:
                            cur_str = pred_update[last_ind:pred_update.index("[START_ENT]",last_ind)]
                        else:
                            cur_str = pred_update[last_ind:]
                    else:
                        updated=False
                    i=0
                preditions_update.append(pred_update)
            predictions=preditions_update
        else:
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
    def predict_mixed_e2e(self,seq,e2e_model,expand=True):
        ner_out=e2e_model.predict_el_e2e(seq,False)
        repl_pattern = r"\[END_ENT\] \[ ([^\]]+) \]"
        ner_out = [re.sub(repl_pattern, "[END_ENT]", el) for el in ner_out]
        if not expand:
            predictions = [self.predict(split + "[SEP]target_el") if split != "" else "" for split in ner_out]
        else:
            joint_ner=". ".join(ner_out)
            expansions=self.expand(joint_ner)
            predictions = [self.predict_with_expansions(split,expansions)  if split != "" else "" for split in ner_out]
        return ". ".join(predictions).replace("Text to annotate:","")
    def predict_el_e2e(self,seq,join_outputs=True,num_beams=1):
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
        predictions=[self.predict("Text to annotate: "+split+"[SEP]target_el",num_beams=num_beams).replace("Text to annotate: ","") if split!=""else "" for split in splits]
        if join_outputs:
            return ". ".join(predictions)
        return predictions
    def predict_el_e2e_exp(self,seq,join_outputs=True,num_beams=1):
        full_seq=self.predict_el_e2e(seq)

        repl_pattern=r"\[END_ENT\] \[ ([^\]]+) \]"
        ner_seq=re.sub(repl_pattern,"[END_ENT]",full_seq)
        exp_seq=""+ner_seq
        expansions=self.expand(exp_seq)
        pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\]"
        result = re.search(pattern, exp_seq)

        while result is not None:
            ent_span=result.group(1)
            if ent_span in expansions and expansions[ent_span] is not None:
                exp_seq = exp_seq.replace(result.group(0), expansions[ent_span], 1)
            else:
                exp_seq = exp_seq.replace(result.group(0), result.group(1), 1)
            result = re.search(pattern, exp_seq)
        print(exp_seq)
        print(ner_seq)
        annotated_expansion_sequence=self.predict_el_e2e(exp_seq)
        pattern = r"\[START_ENT\] ([^\]]+) \[END_ENT\] \[ ([^\]]+) \]"
        result = re.search(pattern, annotated_expansion_sequence)
        span_out_to_entity_exp={}
        while result is not None:
            span_out_to_entity_exp[result.group(1)]=result.group(2)
            annotated_expansion_sequence = annotated_expansion_sequence.replace(result.group(0), "", 1)
            result = re.search(pattern, annotated_expansion_sequence)
        span_out_to_entity = {}
        result = re.search(pattern, full_seq)
        while result is not None:
            span_out_to_entity[result.group(1)] = result.group(2)
            full_seq = full_seq.replace(result.group(0), "", 1)
            result = re.search(pattern, full_seq)


        for span in expansions.keys():
            if expansions[span]!=None:
                s_key=expansions[span]
            else:
                s_key=span
            if s_key in span_out_to_entity_exp:
                ner_seq=ner_seq.replace("[START_ENT] "+span+" [END_ENT]","[START_ENT] "+span+" [END_ENT] [ "+span_out_to_entity_exp[s_key]+" ]")
            elif span in span_out_to_entity:
                ner_seq=ner_seq.replace("[START_ENT] "+span+" [END_ENT]","[START_ENT] "+span+" [END_ENT] [ "+span_out_to_entity[span]+" ]")
        return ner_seq
    def predict_disambiguation_flair_with_ner_expansion(self,seq,original_seq,expand=True):
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

        llm_out = self.lm_ws.run_ner_request(original_seq)
        pattern = r'"([^"]+)"[,\]]'
        result = re.search(pattern, llm_out)
        entities = []
        already_ext = set()
        while result is not None:
            # print(result)
            span = result.group(1)
            if span != result and not span in already_ext:
                already_ext.add(span)
                entities.append(span)
            llm_out = llm_out.replace(result.group(0), "", 1)
            result = re.search(pattern, llm_out)
        print(entities)
        entities.sort(key=len, reverse=True)

        preditions_update = []
        for pred_seq in splits:
            pred_update = pred_seq
            last_ind = 0
            if "[START_ENT]" in pred_update:
                cur_str = pred_update[last_ind:pred_seq.index("[START_ENT]")]
            else:
                cur_str = pred_update

            i = 0
            updated = True
            while updated:
                while i < len(entities):
                    if entities[i] in cur_str:
                        str_up = cur_str.replace(entities[i], "[START_ENT] " + entities[i] + " [END_ENT]")
                        pred_update = pred_update.replace(cur_str, str_up)
                        cur_str = pred_update[last_ind:pred_update.index("[START_ENT]", last_ind)]
                        i = 0
                    else:
                        i = i + 1
                ind_up = last_ind + len(cur_str)
                if "[END_ENT]" in pred_update[ind_up:]:
                    last_ind = pred_update.index("[END_ENT]", ind_up) + len("[END_ENT]")
                    if "[START_ENT]" in pred_update[last_ind:]:
                        cur_str = pred_update[last_ind:pred_update.index("[START_ENT]", last_ind)]
                    else:
                        cur_str = pred_update[last_ind:]
                else:
                    updated = False
                i = 0
            preditions_update.append(pred_update)
        splits = preditions_update
        new_seq=". ".join(splits)
        if not expand:
            predictions = [self.predict(split + "[SEP]target_el") if split != "" else "" for split in splits]
            out=". ".join(predictions)
        else:
            expansions = self.expand(new_seq)
            predictions = [self.predict_with_expansions(split, expansions) if split != "" else "" for split in splits]
            out= ". ".join(predictions)
            #out=self.replace_expansions(seq,output_mod,expansions)
        return out
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
            out= ". ".join(predictions)
            #out=self.replace_expansions(seq,output_mod,expansions)
        return out

'''
import json

d=json.load(open("qald_10_resources.json","r"))
'''
'''
test_str="Barbara Walters stands by Rosie O'Donnell ‘View' host denies Trump's claim she wanted comedian off morning show NEW YORK - Barbara Walters is back from vacation — and she's standing by Rosie O'Donnell in her bitter battle of words with Donald Trump. Walters, creator of ABC's 'The View,' said Wednesday on the daytime chat show that she never told Trump she didn't want O'Donnell on the show, as he has claimed. 'Nothing could be further from the truth,' she said. 'She has brought a new vitality to this show and the ratings prove it,' Walters said of O'Donnell, who is on vacation this week. When she returns, Walters said, 'We will all welcome her back with open arms.' Walters also took a moment to smooth things over with The Donald, who got all riled up when O'Donnell said on 'The View' that he had been 'bankrupt so many times.' 'ABC has asked me to say this just to clarify things, and I will quote: ‘Donald Trump has never filed for personal bankruptcy. Several of his casino companies have filed for business bankruptcies. They are out of bankruptcy now,'' Walters said. O'Donnell and Trump have been feuding since he announced last month that Miss USA Tara Conner, whose title had been in jeopardy because of underage drinking, would keep her crown. Trump is the owner of the Miss Universe Organization, which includes Miss USA and Miss Teen USA. The 44-year-old outspoken moderator of 'The View,' who joined the show in September, said Trump's news conference with Conner had annoyed her 'on a multitude of levels' and that the twice-divorced real estate mogul had no right to be 'the moral compass for 20-year-olds in America.' Trump fired back, calling O'Donnell a 'loser' and a 'bully,' among other insults, in various media interviews. He is the host of NBC's 'The Apprentice"
EL_model=EL_model("../joint_model/aida-125ep",200,apply_llm_ner=True)
res=EL_model.predict_e2e(test_str)
print(res)
'''
'''
test_str="Costa Rica group CocoFunka power this week's Indiesent Exposure http://ht.ly/2G4nS by @fuseboxradio on @planetill"
EL_model=EL_model("../joint_model/e2e_aida",200,apply_llm_ner=True)
res=EL_model.predict_el_e2e_exp(test_str)
print(res)
'''
#EL_model.expand("[START_ENT] Angelina [END_ENT] her partner [START_ENT] Brat [END_ENT] and her father [START_ENT] Jon [END_ENT]")

#EL_model.expand("[START_ENT] David [END_ENT] and [START_ENT] Victoria [END_ENT]")
#EL_model.expand("[START_ENT] Paderborn [END_ENT] was founded as a bishopric by [START_ENT] Charlemagne [END_ENT] in 795, although its official history began in 777 when [START_ENT] Charlemagne [END_ENT] built a castle near the [START_ENT] Paderborn [END_ENT] springs.(4) In 799 [START_ENT] Pope Leo III [END_ENT] fled his enemies in [START_ENT] Rome [END_ENT] and reached [START_ENT] Paderborn [END_ENT], where he met [START_ENT] Charlemagne [END_ENT], and stayed there for three months. It was during this time that it was decided that [START_ENT] Charlemagne [END_ENT] would be crowned emperor. [START_ENT] Charlemagne [END_ENT] reinstated [START_ENT] Leo [END_ENT] in [START_ENT] Rome [END_ENT] in 800 and was crowned as [START_ENT] Holy Roman Emperor [END_ENT] by [START_ENT] Leo [END_ENT] in return.")

