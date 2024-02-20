import json
import torch

from torch.utils.data import Dataset
from transformers import T5PreTrainedModel
import  rdflib.plugins.sparql as sparql
import re
from tqdm import tqdm
from nif import NIFDocument
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
        dataset= ListDataset([])
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

class Dataprocessor_basic(Dataprocessor):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        return samples
    def process_sample(self,input,label=None):
        encoding = self.tokenizer.prepare_seq2seq_batch(src_texts=input,text_target=label, return_tensors="pt",
                                  max_length=self.args["max_target_length"],
                                  max_target_length=self.args["max_target_length"]

                                  )
        '''
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
        '''
        #out["decoder_input_ids"]=T5PreTrainedModel._shift_right(input_ids=out["labels"])
        return encoding.data


class Kilt_joint_el(Dataprocessor_basic):
    def process_training_ds(self,data):
        samples = self.read_ds_to_list(data)
        dataset= ListDataset(samples)
        #for sample in samples:
        #    dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def __call__(self, features):
        return self.process_sample([el["input"]for el in features],[el["label"]for el in features])
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        input=open(path_to_ds+"data.source","r",encoding="utf-8")
        output=open(path_to_ds+"data.target","r",encoding="utf-8")
        for line in tqdm(input):
            src_ner=line+"[SEP]target_ner"
            target = output.readline()
            ents=re.findall(r"{ (.*?) } \[ (.*?) ]",target)
            target_ner=""+target
            target_el = "" + target
            for el in ents:
                target_ner=target_ner.replace("{ "+el[0]+" } [ "+el[1]+" ]","[START_ENT] "+el[0]+" [END_ENT]")
                target_el = target_el.replace("{ " + el[0] + " }","[START_ENT] " + el[0] + " [END_ENT]")
            samples.append({"input":src_ner,"label":target_ner})
            samples.append({"input": target_ner+"[SEP]target_el", "label": target_el})
        input.close()
        output.close()
        return samples

class Aida_joint_el(Dataprocessor_basic):
    def process_training_ds(self,data):
        samples = self.read_ds_to_list(data)
        dataset= ListDataset(samples)
        #for sample in samples:
        #    dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def __call__(self, features):
        return self.process_sample([el["input"]for el in features],[el["label"]for el in features])
    def load(self,path_to_ds):
        with open(path_to_ds, 'r', encoding='utf-8') as file:
            doc = NIFDocument.nifStringToNifDocument(file.read())
        return doc


    def groupNifDocumentByRefContext(self,ds: NIFDocument):
        refContextToDocument = {}
        for nifContent in ds.nifContent:
            if nifContent.reference_context is not None:
                docrefcontext = nifContent.reference_context
            else:
                docrefcontext = nifContent.uri
            if docrefcontext in refContextToDocument:
                refContextToDocument.get(docrefcontext).addContent(nifContent)
            else:
                doc = NIFDocument.NIFDocument()
                doc.addContent(nifContent)
                refContextToDocument[nifContent.uri] = doc
        #for el in refContextToDocument:
        #    refContextToDocument[el].nifContent.sort(key=lambda x: x.begin_index)
        return refContextToDocument

    def split(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    def read_ds_to_list(self,path_to_ds):
        # candidate_file=json.load(open("../candiatedata/"+dsName+"_candidates.json"))
        wikipedia_data = pickle.load(open("../wikipedia_data.pkl", "rb"))
        # labels=pickle.load(open("labels_full_updated.sav","rb"))
        documentmap = self.groupNifDocumentByRefContext(self.load(path_to_ds))
        samples = []
        seq_len = 300
        i = 0
        for el in documentmap:
            source = ""
            target = ""
            annotations = []
            for cont in documentmap.get(el).nifContent:
                if cont.is_string is not None:
                    target = cont.is_string
                    source = cont.is_string
                else:
                    annotations.append(cont)
            curr_seq_len = self.tokenizer(source, return_tensors="pt").input_ids.size(1)
            num_splits = curr_seq_len // seq_len

            if num_splits > 0:
                sentences = source.split(". ")
                splits = []
                chunks = list(self.split(sentences, num_splits + 1))
                for chunk in chunks:
                    splits.append(". ".join(chunk) + ". ")
                print(splits)
            else:
                splits = [source]
            annotations.sort(key=lambda x: int(x.begin_index))
            target_splits_el = splits.copy()
            # target_splits_ner = splits.copy()
            offsets = [0 for el in splits]
            starttag = "[START_ENT] "
            endtag = " [END_ENT]"
            # cands = set()
            elinks = set()

            for an in annotations:
                reduce = 0
                sp_id = 0
                for sp in splits:
                    if not int(an.begin_index) > len(sp) + reduce:
                        sp_id = splits.index(sp)
                        break
                    else:
                        reduce += len(sp)
                        # sp_id = splits.index(sp)

                if not "notInWiki" in an.taIdentRef:
                    # cands=cands.union(set(candidates[an.uri]))
                    target_splits_el[sp_id] = \
                        target_splits_el[sp_id][0:int(an.begin_index) - reduce + offsets[sp_id]] + starttag + \
                        target_splits_el[sp_id][int(an.begin_index) - reduce + offsets[sp_id]:
                                                len(target_splits_el[sp_id])]
                    offsets[sp_id] += len(starttag)
                    tag = " [ " + wikipedia_data[an.taIdentRef.replace("http://", "")]["title"] + " ]"
                    ellink = endtag + tag
                    target_splits_el[sp_id] = target_splits_el[sp_id][0:int(an.end_index) - reduce +
                                                                        offsets[sp_id]] + ellink + target_splits_el[
                                                                                                       sp_id][
                                                                                                   int(an.end_index) - reduce +
                                                                                                   offsets[sp_id]:len(
                                                                                                       target_splits_el[
                                                                                                           sp_id])]
                    offsets[sp_id] += len(ellink)

            target_splits_ner = splits.copy()
            offsets = [0 for el in splits]
            starttag = "[START_ENT] "
            endtag = " [END_ENT]"
            # cands = set()
            elinks = set()

            for an in annotations:
                reduce = 0
                sp_id = 0
                for sp in splits:
                    if not int(an.begin_index) > len(sp) + reduce:
                        sp_id = splits.index(sp)
                        break
                    else:
                        reduce += len(sp)
                        # sp_id = splits.index(sp)

                if not "notInWiki" in an.taIdentRef:
                    # cands=cands.union(set(candidates[an.uri]))
                    target_splits_ner[sp_id] = \
                        target_splits_ner[sp_id][0:int(an.begin_index) - reduce + offsets[sp_id]] + starttag + \
                        target_splits_ner[sp_id][int(an.begin_index) - reduce + offsets[sp_id]:
                                                 len(target_splits_ner[sp_id])]
                    offsets[sp_id] += len(starttag)
                    ellink = endtag
                    target_splits_ner[sp_id] = target_splits_ner[sp_id][0:int(an.end_index) - reduce +
                                                                          offsets[sp_id]] + ellink \
                                               + target_splits_ner[sp_id][int(an.end_index) - reduce + offsets[sp_id]:
                                                                          len(target_splits_ner[sp_id])]
                    offsets[sp_id] += len(ellink)
            for el in splits:
                sample = {"input": el+"[SEP]target_ner", "label": target_splits_ner[splits.index(el)]}
                samples.append(sample)
            for el in target_splits_ner:
                sample = {"input": el + "[SEP]target_el", "label": target_splits_el[target_splits_ner.index(el)]}
                samples.append(sample)

        return samples






