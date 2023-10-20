import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from getContextMentionIds import getmentioncontext
from nif import NIFDocument
import pickle
redirects=pickle.load(open("data/redirects.sav","rb"))
know_entities=pickle.load(open("data/known_entities.sav","rb"))
import torch
from torch.utils.data import Dataset
import requests
import json

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

    def process_training_ds_prebuild(self,filname):
        samples=pickle.load(open(filname,"rb"))
        dataset = ListDataset([])
        for sample in samples:
            dataset.examples.append(self.process_sample(sample["source"], sample["target"]))
        return dataset
    def process_training_ds(self,dsName,cand_file,path="data/nifData/"):
        samples = self.generate_seq_to_seq_sample_graph(dsName,cand_file,path)
        dataset=ListDataset([])
        for sample in samples:
            dataset.examples.append(self.process_sample(sample["source"],sample["target"]))
        return dataset
    def process_sample(self,input,label=None):
        encoding = self.tokenizer.prepare_seq2seq_batch(src_texts=[input],text_target=[label], return_tensors="pt",
                                  max_length=self.args["max_target_length"],
                                  max_target_length=self.args["max_target_length"]

                                  )
        issr=input
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

    def groupNifDocumentByRefContextKGE(self,ds: NIFDocument,KGE):
        refContextToDocument = {}
        for nifContent in ds.nifContent:
            if nifContent.reference_context is not None and not "notInWiki"in nifContent.taIdentRef and KGE.checkcontains(head_entity=[nifContent.taIdentRef]):
                docrefcontext = nifContent.reference_context
            else:
                docrefcontext = nifContent.uri
            if docrefcontext in refContextToDocument:
                refContextToDocument.get(docrefcontext).addContent(nifContent)
            else:
                doc = NIFDocument.NIFDocument()
                doc.addContent(nifContent)
                refContextToDocument[nifContent.uri] = doc
        return refContextToDocument

    def load(self,path_to_ds):
        with open(path_to_ds, 'r', encoding='utf-8') as file:
            doc = NIFDocument.nifStringToNifDocument(file.read())
        return doc

    def generate_seq_to_seq_sample(self,dsName,cand_file,path="data/nifData/"):
        candidates=pickle.load(open(cand_file,"rb"))
        labels=pickle.load(open("labels_full_updated.sav","rb"))
        documentmap = self.groupNifDocumentByRefContext(self.load(path + dsName))
        samples=[]
        seq_len=300
        for el in documentmap:
            #source=""
            target=""
            annotations=[]
            for cont in documentmap.get(el).nifContent:
                if cont.is_string is not None:
                    target = cont.is_string
                else:
                    annotations.append(cont)
            annotations.sort(key=lambda x: int(x.begin_index))
            offset=0
            starttag="[BENT]"
            endtag="[EENT]"
            #cands = set()
            elinks= {}
            for an in annotations:

                if not "notInWiki" in an.taIdentRef:
                    #cands=cands.union(set(candidates[an.uri]))
                    target=target[0:int(an.begin_index)+offset]+starttag+target[int(an.begin_index)+offset:len(target)]
                    offset+=len(starttag)
                    ellink=" link:"+an.taIdentRef.replace("http://dbpedia.org/resource","dbr:")+endtag
                    elinks[ellink]=candidates[an.uri]
                    target=target[0:int(an.end_index)+offset]+ellink+target[int(an.end_index)+offset:len(target)]
                    offset+=len(ellink)


            splits = []
            while len(target)>seq_len:
                next_end_ind = len(target)
                ind_sent=seq_len
                if "[EENT]" in target[seq_len:len(target)]:
                    next_end_ind=target[seq_len:len(target)].index("[EENT]")+seq_len+len("[EENT]")
                while not "."==target[ind_sent] and "." in target[seq_len:len(target)]:
                    ind_sent=ind_sent+1
                ind=min(next_end_ind,ind_sent)
                sent=target[0:ind+1]
                splits.append(sent)
                target=target[ind+1:len(target)]
            splits.append(target)
            for section in splits:
                source = section.replace("[BENT]","")
                candstr="[CANDIDATES]: "
                cands=set()
                for em in elinks:
                    if em in source:
                        cands=cands.union(set(elinks[em]))
                        source=source.replace(em,"")
                for cand in cands:
                    if not ""==cand:
                        candstr += cand.replace("http://dbpedia.org/resource/", "")+" , "
                print(source+candstr)
                samples.append({"source":source+candstr,"target":section})
            #source+=candstr
            #print(target)
            #samples.append({"source":target,"target":target})
        return samples

    def generate_seq_to_seq_sample_graph(self,dsName,cand_file,path="data/nifData/"):
        candidates=pickle.load(open(cand_file,"rb"))
        labels=pickle.load(open("labels_full_updated.sav","rb"))
        documentmap = self.groupNifDocumentByRefContext(self.load(path + dsName))
        samples=[]
        seq_len=300
        it=0
        for el in documentmap:
            #source=""
            target=""
            annotations=[]
            for cont in documentmap.get(el).nifContent:
                if cont.is_string is not None:
                    target = cont.is_string
                else:
                    annotations.append(cont)
            annotations.sort(key=lambda x: int(x.begin_index))
            offset=0
            starttag="[BENT]"
            endtag="[EENT]"
            #cands = set()
            elinks= {}
            all_cands=set()

            for an in annotations:

                if not "notInWiki" in an.taIdentRef:
                    all_cands=all_cands.union(set(candidates[an.uri]))
                    target=target[0:int(an.begin_index)+offset]+starttag+target[int(an.begin_index)+offset:len(target)]
                    offset+=len(starttag)
                    ellink=" link:"+an.taIdentRef.replace("http://dbpedia.org/resource","dbr:")+endtag
                    elinks[ellink]=candidates[an.uri]
                    target=target[0:int(an.end_index)+offset]+ellink+target[int(an.end_index)+offset:len(target)]
                    offset+=len(ellink)

            resp_all = requests.post("http://localhost:5000/get_subgraph_dp_1", data=json.dumps(
                list(all_cands))).json()
            splits = []


            while len(target)>seq_len:
                next_end_ind = len(target)
                ind_sent=seq_len
                if "[EENT]" in target[seq_len:len(target)]:
                    next_end_ind=target[seq_len:len(target)].index("[EENT]")+seq_len+len("[EENT]")
                while not "."==target[ind_sent] and "." in target[seq_len:len(target)]:
                    ind_sent=ind_sent+1
                ind=min(next_end_ind,ind_sent)
                sent=target[0:ind+1]
                splits.append(sent)
                target=target[ind+1:len(target)]
            splits.append(target)
            for section in splits:
                source = section.replace("[BENT]","")
                candstr="[LINKED]: "
                cands=set()
                for em in elinks:
                    if em in source:
                        cands=cands.union(set(elinks[em]))
                        source=source.replace(em,"")
                #resp = requests.post("http://localhost:5000/get_subgraph_dp_1", data=json.dumps(
                #    list(cands))).json()
                related=set()
                for tp in resp_all["direct_relations"]:
                    if tp[0]in cands and tp[2] in cands:
                        candstr+=tp[0].replace("http://dbpedia.org/resource/", "")
                        candstr += " , "+tp[1].replace("http://dbpedia.org/ontology/", "")
                        candstr += " , " + tp[2].replace("http://dbpedia.org/resource/", "")
                        related.add(tp[0])
                        related.add(tp[2])
                        candstr+="[SEP]"
                candstr +="[ONE_HOP]"
                for tp in resp_all["one_hop_connected"]:
                    if tp[0] in cands and tp[1] in cands:
                        candstr+=tp[0].replace("http://dbpedia.org/resource/", "")
                        candstr += " , "+tp[1].replace("http://dbpedia.org/resource/", "")+"[SEP]"
                        related.add(tp[0])
                        related.add(tp[1])
                candstr+="[UNLINKED]"
                for el in cands.difference(related):
                    candstr += el.replace("http://dbpedia.org/resource/", "")+"[SEP]"


                samples.append({"source":source+candstr,"target":section})
            print(str(it) + " processed Documents")
            it += 1
            #source+=candstr
            #print(target)
            #samples.append({"source":target,"target":target})
        pickle.dump(samples,open("graph_samples_"+dsName+".pkl","wb"))
        return samples

