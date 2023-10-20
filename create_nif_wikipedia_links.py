import pickle


from nif import NIFDocument

redirects=pickle.load(open("data/redirects.sav","rb"))
wikipedialinks=pickle.load(open("titles_to_wikipedia.pkl","rb"))
wikipediawd=pickle.load(open("titles_to_wikipedia.pkl","rb"))
lab=pickle.load(open("labels_full_updated.sav","rb"))
def load(path_to_ds):
    with open("data/nifData/"+path_to_ds, 'r',encoding="utf-8") as file:
        doc = NIFDocument.nifStringToNifDocument(file.read())
    return doc
def loadwd(path_to_ds):
    with open("data/wikidata/"+path_to_ds, 'r',encoding="utf-8") as file:
        doc = NIFDocument.nifStringToNifDocument(file.read())
    return doc

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

for ds in dataset_names:
    data=load(ds)
    for cont in data.nifContent:

        if cont.taIdentRef is not None and not "notInWiki" in cont.taIdentRef:
            uri = cont.taIdentRef
            if uri in redirects:
                uri=redirects[uri]
            if not uri in lab:
                print(uri+" not in lab")
                notfound.add(cont.uri)
            else:
                label=lab[uri]
                if not label[0] in wikipedialinks:
                    notfound.add(cont.uri)

#pickle.dump(notfound,open("notfoundannotations","rb"))
wikidatalinks=pickle.load(open("wikidata_to_wikipedia.pkl","rb"))
nfound_wd=[]
for ds in dataset_names:
    data=loadwd(ds)
    for cont in data.nifContent:

        if cont.taIdentRef is not None and not "notInWiki" in cont.taIdentRef:
           if cont.uri in notfound:
               if not cont.taIdentRef.replace("http://www.wikidata.org/entity/","") in wikidatalinks:
                   print(cont.uri)
                   nfound_wd.append(cont.uri)
                   print(cont.taIdentRef)
print(len(nfound_wd))

