import json

from nifDataHandlers import DataprocessorRetrieval
from transformers import BertTokenizer
from entqa_dualencoder.Passageretriever import PassageRetriever

tokenizer=BertTokenizer.from_pretrained('bert-large-uncased')
ds=DataprocessorRetrieval(tokenizer,{})

retriever=PassageRetriever()

#pickle.load(open("../wikidata_to_wikipedia.pkl","rb"))

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
notfound=set()

for dataset in dataset_names:


    documents = ds.process_retrieval_ds(dataset)
    num_found=0
    num_not_found=0
    out_ds=[]
    for doc in documents:
        out_doc=[]
        retrievaldocuments=[]
        topic=doc[0]
        for passage in doc:
            retrievaldocuments.append({"text":passage["text"],"topic":topic["text"]})
            out_doc.append({"text":passage["text"],"topic":topic["text"],"gold_ents":list(passage["entities"])})
        results=retriever.search_for_passages(retrievaldocuments)
        for i in range(0,len(results)):
            out_doc[i]["candidates"]=results[i]
        out_ds.append(out_doc)
    json.dump(out_ds, open(dataset+"_candidates.json","w",encoding="utf-8"),indent=1)
