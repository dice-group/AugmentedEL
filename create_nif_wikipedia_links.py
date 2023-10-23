import pickle


from nif import NIFDocument

redirects=pickle.load(open("data/redirects.sav","rb"))
wikipedialinks=pickle.load(open("titles_to_wikipedia.pkl","rb"))
#wikipediawd=pickle.load(open("titles_to_wikipedia.pkl","rb"))
lab=pickle.load(open("labels_full_updated.sav","rb"))
wikidatalinks=pickle.load(open("wikidata_to_wikipedia.pkl","rb"))
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
    data_db=load(ds)
    data_wk=loadwd(ds)
    nif_cont=data_db.nifContent
    nif_cont_wikidata = data_wk.nifContent
    for i in range(0,len(nif_cont)):

        if nif_cont[i].taIdentRef is not None and not "notInWiki" in nif_cont[i].taIdentRef:
            uri = nif_cont[i].taIdentRef
            stop=False
            if uri in redirects:
                uri=redirects[uri]
            if not uri in lab:
                #print(uri+" not in lab")
                #notfound.add(nif_cont.uri)
                if  not "notInWiki" in nif_cont_wikidata[i].taIdentRef:
                    wikipedia_id=wikidatalinks[nif_cont_wikidata[i].taIdentRef.replace("http://www.wikidata.org/entity/","")]
                else:
                    stop=True
            else:
                label=lab[uri]
                if not label[0] in wikipedialinks:
                    if not "notInWiki" in nif_cont_wikidata[i].taIdentRef:
                        wikipedia_id=wikidatalinks[nif_cont_wikidata[i].taIdentRef.replace("http://www.wikidata.org/entity/","")]
                    else:
                        stop = True
                elif not stop:
                    wikipedia_id=wikipedialinks[label[0]]
            if not stop:
                print(wikipedia_id)
                nif_cont[i].taIdentRef="http://"+wikipedia_id
            else:
                nif_cont[i].taIdentRef = "http://notInWiki"+nif_cont[i].anchor_of.replace(" ","_")
    with open("data/wikipedia_nif/"+ds,"w",encoding="utf-8")as out:

        out.write(data_db.get_nif_string())

#pickle.dump(notfound,open("notfoundannotations","rb"))
'''
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
'''
