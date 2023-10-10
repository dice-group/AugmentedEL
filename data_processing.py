import json
import torch

from torch.utils.data import Dataset
from transformers import T5PreTrainedModel
import  rdflib.plugins.sparql as sparql

from rdflib.plugins.sparql import algebra
import pickle

class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)

class Dataprocessor():
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args=args

    def read_ds_to_list(self,path_to_ds):
        return []
    def process_training_ds(self,data):
        samples = self.read_ds_to_list(data)
        dataset=ListDataset([])
        for sample in samples:
            dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def process_sample(self,input,label=None):
        pass
class Dataprocessor_test(Dataprocessor):

    def process_sample(self,input,label=None):
        encoding = self.tokenizer(text=input,text_target=label, return_tensors="pt",
                                  )
        return encoding

class Dataprocessor_KBQA_basic(Dataprocessor):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        return samples
    def process_sample(self,input,label=None):
        encoding = self.tokenizer.prepare_seq2seq_batch(src_texts=[input],text_target=[label], return_tensors="pt",
                                  max_length=self.args["max_target_length"],
                                  max_target_length=self.args["max_target_length"]

                                  )
        input=encoding.data["input_ids"]
        padded_input_tensor = self.tokenizer.pad_token_id * torch.ones(
            (input.shape[0], self.args["max_input_length"]), dtype=input.dtype, device=input.device
        )
        padded_input_tensor[:, : input.shape[-1]] = input
        encoding.data["input_ids"]=torch.flatten(padded_input_tensor)

        attention_mask = encoding.data["attention_mask"]
        padded_attention_mask = self.tokenizer.pad_token_id * torch.ones(
            (attention_mask.shape[0], self.args["max_input_length"]), dtype=attention_mask.dtype, device=attention_mask.device
        )
        padded_attention_mask[:, : attention_mask.shape[-1]] = attention_mask
        encoding.data["attention_mask"] = torch.flatten(padded_attention_mask)

        target = encoding.data["labels"]
        padded_target_tensor = self.tokenizer.pad_token_id * torch.ones(
            (target.shape[0], self.args["max_target_length"]), dtype=target.dtype, device=target.device
        )
        padded_target_tensor[:, : target.shape[-1]] = target
        encoding.data["labels"]=torch.flatten(padded_target_tensor)

        #out["decoder_input_ids"]=T5PreTrainedModel._shift_right(input_ids=out["labels"])
        return encoding.data


class LCqUAD_text_ent(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data:
            if "entities" in question and  question["question"] is not None:
                print(question)
                sample={}
                in_text=question["question"]
                in_text+="<sep> entities: "
                for ent in question["entities"]:
                    in_text+=ent["label"]+" : "+ent["uri"].replace("http://www.wikidata.org/entity/","")+" , "
                sample["input"]=in_text
                sample["label"]=question["sparql_wikidata"]
                samples.append(sample)
        return samples

class LCqUAD(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data:
            if "entities" in question and  question["question"] is not None:
                print(question)
                sample={}
                in_text=question["question"]
                sample["input"]=in_text
                sample["label"]=question["sparql_wikidata"]
                samples.append(sample)
        return samples

class LCqUAD_rep_vars(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self,path_to_ds):
        prefixes = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX wdv: <http://www.wikidata.org/value/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        """
        samples=[]
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data:
            if "entities" in question and  question["question"] is not None:
                sample={}
                in_text=question["question"]
                in_text += "<sep> entities: "
                parsed_query = sparql.parser.parseQuery(prefixes+question["sparql_wikidata"])

                en = algebra.translateQuery(parsed_query)
                for ent in question["entities"]:
                    in_text+=ent["label"]+" : "+ent["uri"].replace("http://www.wikidata.org/entity/","")+" , "
                res_vars=en.algebra["PV"]
                vars=en.algebra["_vars"]
                it = 0
                processed_Query=question["sparql_wikidata"]
                for el in res_vars:
                    processed_Query=processed_Query.replace("?"+el,"_result_"+str(it))
                    it+=1
                it=0
                for el in vars:
                    processed_Query=processed_Query.replace("?"+el,"_var_"+str(it))
                    it += 1
                processed_Query=processed_Query.replace("{","_cbo_")
                processed_Query=processed_Query.replace("}", "_cbc_")
                sample["input"]=in_text
                sample["label"]=processed_Query
                print(sample)
                samples.append(sample)
        return samples

class LCqUAD_rep_vars_triples(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self,path_to_ds):
        prefixes = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX wdv: <http://www.wikidata.org/value/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        """
        samples=[]
        entitylabels = pickle.load(open("../precomputed/wikidata_labels.sav", "rb"))
        relationlabels = pickle.load(open("../precomputed/relation_labels.sav", "rb"))
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data:
            if "triples" in question and len(question["triples"])>0 and  question["question"] is not None:
                sample={}
                in_text=question["question"]
                in_text += "<sep> entities: "
                parsed_query = sparql.parser.parseQuery(prefixes+question["sparql_wikidata"])

                en = algebra.translateQuery(parsed_query)
                triple_str=""
                entities=set()
                relations=set()
                for t in question["triples"]:
                     if t[0].replace("http://www.wikidata.org/entity/","") in entitylabels:
                         entities.add(t[0].replace("http://www.wikidata.org/entity/",""))
                     triple_str+=t[0].replace("http://www.wikidata.org/entity/","")+" "
                     if t[1].replace("http://www.wikidata.org/prop/direct/","") in relationlabels:
                         relations.add(t[1].replace("http://www.wikidata.org/prop/direct/",""))
                     triple_str+=t[1].replace("http://www.wikidata.org/prop/direct/","")+" "
                     if t[2].replace("http://www.wikidata.org/entity/","") in entitylabels:
                        entities.add(t[2].replace("http://www.wikidata.org/entity/",""))
                     triple_str += t[2].replace("http://www.wikidata.org/entity/","") + " ; "


                for ent in entities:
                    in_text+=entitylabels[ent]+" : "+ent+" , "
                in_text += "<sep> relations: "
                for rel in relations:
                    in_text+=relationlabels[rel]+" : "+rel+" , "
                in_text+="<sep> triples: "+triple_str
                res_vars=en.algebra["PV"]
                vars=en.algebra["_vars"]
                it = 0
                processed_Query=question["sparql_wikidata"]
                for el in res_vars:
                    processed_Query=processed_Query.replace("?"+el,"_result_"+str(it))
                    it+=1
                it=0
                for el in vars:
                    processed_Query=processed_Query.replace("?"+el,"_var_"+str(it))
                    it += 1
                processed_Query=processed_Query.replace("{","_cbo_")
                processed_Query=processed_Query.replace("}", "_cbc_")
                sample["input"]=in_text
                sample["label"]=processed_Query
                print(sample)
                samples.append(sample)
        return samples

class Dataprocessor_QALD(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data["questions"]:
            sample={}
            for lang in question["question"]:
                if lang["language"]=="en":
                    sample["input"]=lang["string"]
            sample["label"]=question["query"]["sparql"]
            samples.append(sample)
        return samples






