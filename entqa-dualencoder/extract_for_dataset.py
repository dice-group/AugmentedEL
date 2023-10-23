from nifDataHandlers import DataprocessorRetrieval
from transformers import BertTokenizer
from Passageretriever import PassageRetriever

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
    for doc in documents:
        retrievaldocuments=[]
        topic=doc[0]
        allresults=set()
        gold_ents=set()
        for passage in doc:
            retrievaldocuments.append({"text":passage["text"],"topic":topic["text"]})
            gold_ents=gold_ents.union(passage["entities"])
        results=retriever.search_for_passages(retrievaldocuments)
        for result_per_passage in results:
            allresults=allresults.union(set(result_per_passage))
        num_foundcurr=len(allresults.intersection(gold_ents))
        num_found+=num_foundcurr
        num_not_found+=len(gold_ents)-num_foundcurr
    print(dataset)
    print("found:"+str(num_found)+"notfound"+str(num_not_found))
