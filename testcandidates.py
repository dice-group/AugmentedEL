import json

data= json.load(open("candiatedata/aida_train_candidates.json"))
for doc in data:
    #candsset=set()
    candmap={}
    gold_ents=set()
    for passage in doc:
        for cand in passage["candidates"]:
            if not cand in candmap:
                candmap[cand]=0
            candmap[cand]=candmap[cand]+1
            gold_ents=gold_ents.union(set(passage["gold_ents"]))
    print("passages",len(doc))
    d_num=0
    for k in candmap:
        if candmap[k]>(len(doc)/3)*2:
            d_num+=1
    print(d_num)
    '''
    for gold_ents in gold_ents:
        if gold_ents in candmap:
            print(gold_ents,candmap[gold_ents])
        else:
            print(gold_ents,0)
    '''