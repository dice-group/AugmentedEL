from initialize_model import load_model
import torch
from transformers import BertTokenizer
import numpy
import faiss
import FaissApi as indexApi
import pickle

class PassageRetriever():
    def __init__(self,maxlen=128,batch_size=10,index_path="faiss-hswf-entqa",id_to_index="id-to-index-entqa.pkl"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model=load_model("retriever.pt",self.device)
        self.tokenizer=BertTokenizer.from_pretrained('bert-large-uncased')
        self.max_len=maxlen
        self.batch_size=batch_size
        faiss_index = faiss.read_index(index_path)
        self.index = indexApi.DenseHNSWFlatIndexer(1024)
        self.index.index = faiss_index

        self.index.index_id_to_db_id = pickle.load(open(id_to_index, "rb"))
        print("finished initializing")

    def encode_passage(self,passage,topic=None):
        passage_tokens = self.tokenizer.tokenize(passage)
        topic_tokens=[]
        if topic is not None:
            topic_tokens = self.tokenizer.tokenize(topic)
        window = ["[CLS]"]+passage_tokens[:self.max_len-2] + ["[SEP]"]
        if len(window)<self.max_len and topic_tokens:
            window+=topic_tokens
            window=window[:self.max_len]

        token_ids = self.tokenizer.convert_tokens_to_ids(window)
        passage_mask = [1] * len(token_ids)
        padding = [0] * (self.max_len - len(token_ids))
        passage_mask += padding
        token_ids += padding
        return token_ids,passage_mask

    def embedpassages(self,passages):
        passagestokens=[]
        masks = []
        embeddings=[]
        for passage in passages:
            if "topic" in passage:
                p,m=self.encode_passage(passage["text"],passage["topic"])
            else:
                p,m=self.encode_passage(passage["text"])
            passagestokens.append(p)
            masks.append(m)

            if len(passagestokens) == self.batch_size:
                em = self.model.encode(mention_token_ids=torch.tensor(passagestokens, device=self.device),
                                  mention_masks=torch.tensor(masks, device=self.device))
                embeddings.extend(em[0].tolist())
                passagestokens = []
                masks = []
        if len(passagestokens) > 0:
            em = self.model.encode(mention_token_ids=torch.tensor(passagestokens, device=self.device),
                                   mention_masks=torch.tensor(masks, device=self.device))
            embeddings.extend(em[0].tolist())
        return embeddings
    def search_index(self, candidate_encodings,k):
        candidates=numpy.array(candidate_encodings,dtype=numpy.float32)
        found_uris=[]
        D, I = self.index.search_knn(candidates, k)
        found=[]
        for el in numpy.nditer(I, order='C'):
            f = self.index.index_id_to_db_id[int(el)]
            found.append(f)
            if len(found) == k:
                found_uris.append(found)
                found = []
        return found_uris

    def search_for_passages(self,passages,k=60):
        passage_embeddings=self.embedpassages(passages)
        found=self.search_index(passage_embeddings,k)
        return found





