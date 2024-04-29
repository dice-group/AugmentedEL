from nif import NIFDocument,NIFContent
import pickle
import re
def load(path_to_ds):
    with open(path_to_ds, 'r', encoding='utf-8') as file:
        doc = NIFDocument.nifStringToNifDocument(file.read())
    return doc
wk_tiltels=pickle.load(open("../titles_to_wikipedia.pkl","rb"))

ds=load("extracted_der")
cont_rem=[]
for ct in ds.nifContent:
    if ct.taIdentRef is not None:
        lb=ct.taIdentRef.replace("http://","").replace("_"," ")
        if lb in wk_tiltels or re.match(r"http://[0-9]+",ct.taIdentRef):
            if lb in wk_tiltels:
                ct.taIdentRef="http://"+wk_tiltels[lb]
                cont_rem.append(ct)
            else:
                cont_rem.append(ct)
        else:
            print(ct.taIdentRef)
    else:
        cont_rem.append(ct)
ds.nifContent=cont_rem
with open("../data/nif_wikipedia_cleaned/der","w",encoding="utf-8")as out_file:
    out_file.write(ds.get_nif_string())
