import pickle
d=pickle.load(open("wikipedia_data.pkl","rb"))
dbpedia_links={}
for key in d:
    ent=d[key]
    dbpedia_link=ent["title"]
    dbpedia_links[dbpedia_link]=key
pickle.dump(dbpedia_links,open("titles_to_wikipedia.pkl","wb"))
print(d)