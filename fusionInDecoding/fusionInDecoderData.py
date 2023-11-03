import json
import pickle
from nif import NIFDocument

class Dataprocessor():
    def __init__(self, tokenizer, args,input_prefix="Document",candidiate_prefix="Candidate",kb="../wikipedia_data.pkl"):
        self.tokenizer = tokenizer
        self.args = args
        self.kb = pickle.load(open(kb,"rb"))
        self.input_prefix=input_prefix
        self.candidate_prefix=candidiate_prefix

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

    def generate_seq_to_seq_sample(self,dsName,path="../data/wikipedia_nif/"):
        candidate_file=json.load(open("../candiatedata/"+dsName+"_candidates.json"))
        #labels=pickle.load(open("labels_full_updated.sav","rb"))
        documentmap = self.groupNifDocumentByRefContext(self.load(path + dsName))
        samples=[]
        seq_len=300
        i=0
        for el in documentmap:
            #source=""
            target=""
            annotations=[]
            for cont in documentmap.get(el).nifContent:
                if cont.is_string is not None:
                    target = cont.is_string
                else:
                    annotations.append(cont)
            candidate_doc=candidate_file[i]
            for paragraph in candidate_doc:
                ind=target.find(paragraph["text"])
                paragraph["start"]=ind
                paragraph["end"]=ind+len(paragraph["text"])
            annotations.sort(key=lambda x: int(x.begin_index))

            offset=0
            starttag="[BENT]"
            endtag="[EENT]"
            #cands = set()
            elinks= set()
            for an in annotations:

                if not "notInWiki" in an.taIdentRef:
                    #cands=cands.union(set(candidates[an.uri]))
                    target=target[0:int(an.begin_index)+offset]+starttag+target[int(an.begin_index)+offset:len(target)]
                    offset+=len(starttag)
                    ellink=" link:"+an.taIdentRef.replace("http://dbpedia.org/resource","dbr:")+endtag
                    elinks.add(ellink)
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
            curr_start_ind=0
            for section in splits:
                source = section.replace("[BENT]","")
                for em in elinks:
                    if em in source:
                        #cands=cands.union(set(elinks[em]))
                        source=source.replace(em,"")
                curr_end_ind=curr_start_ind+len(source)
                candidates=set()
                for paragraph in candidate_doc:
                    if curr_start_ind <= paragraph["start"] and curr_end_ind >paragraph["start"]:
                        candidates = candidates.union(set(paragraph["candidates"]))
                curr_start_ind=curr_end_ind
                samples.append({"source":source,"candidates":list(candidates),"target":section})
            #source+=candstr
            #print(target)
            #samples.append({"source":target,"target":target})
            i+=1
        return samples


    def read_to_list(self,datasetname):
        samples=[]
        data = json.load(open(datasetname))
        for doc in data:
            for passage in doc:
                candidates=[]
                for cand in passage["candidates"]:
                    cand_ent=self.kb[cand]
                    samples.append()
                    print(cand)

dataset_names = ["ACE2004"]
dataset_names.append("aida_testa")
dataset_names.append("aida_testb")
dataset_names.append("aida_train")
dataset_names.append("aida_complete")
dataset_names.append("AQUAINT")
dataset_names.append("spotlight")
dataset_names.append("iitb-fix")
dataset_names.append("KORE50")
dataset_names.append("MSNBC")
dataset_names.append("N3-Reuters-128")
dataset_names.append("N3-RSS-500")
dataset_names.append("oke-challenge-task1-eval")
dataset_names.append("oke-challenge-task1-example")
dataset_names.append("oke-challenge-task1-gs")

dp = Dataprocessor(None,[])
for ds in dataset_names:
    samples = dp.generate_seq_to_seq_sample(ds)
    pickle.dump(samples,open("data/"+ds+".pkl","wb"))