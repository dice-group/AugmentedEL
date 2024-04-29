import json
import pickle
from tqdm import tqdm
from experiments.prefix_trie.PrefixTrie import Trie_not_recursive
import csv
from transformers import T5Tokenizer
import torch
import sys
sys.setrecursionlimit(1500)
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base",)
entities=pickle.load(open("../../wikipedia_data.pkl","rb"))

trie=Trie_not_recursive()
#entities=list(entities.keys())
#entities.append("relations: ")
entdict={}
for el in tqdm(list(entities.keys())):
    #print(entities[el].encode("utf-8").decode('unicode_escape'))
    #if el == "relations: ":
    #    stest=tokenizer.tokenize(el)
    #    seq=tokenizer.encode(el)[:-1]
    lab=" "+entities[el]["title"]+" ]"
    seq=tokenizer.encode(lab)
    #print(lab)
    #print(tokenizer.tokenize(lab))
    trie.add(seq)
pickle.dump(trie,open("../entity_trie_u.pkl","wb"))