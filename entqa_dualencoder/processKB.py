from transformers import BertTokenizer
import json
import pickle
import numpy
import torch
import faiss
from tqdm import tqdm
from initialize_model import load_model
from FaissApi import DenseHNSWFlatIndexer
def get_entity_window(item, tokenizer, max_ent_len):
    title = item['title']
    text = item["abstract"]
    #text = ' '.join(text)
    #max_ent_len -= 2  # CLS, SEP
    ENT = '[unused2]'
    title_tokens = tokenizer.tokenize(title)
    text_tokens = tokenizer.tokenize(text)
    window = ["[CLS]"]+(title_tokens + [ENT] + text_tokens)[:max_ent_len-2]+["[SEP]"]
    question_ids = tokenizer.convert_tokens_to_ids(window)
    entity_mask=[1]*len(question_ids)
    padding = [0] * (max_ent_len - len(question_ids))
    entity_mask += padding
    question_ids += padding

    return question_ids, entity_mask


# process kilt knowledge base
def process_kilt_kb(args):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    fout = open(args.out_kb_path, 'w')
    with open(args.raw_kb_path, 'r') as f:
        for line in f:
            field = {}
            item = json.loads(line)
            window = get_entity_window(item, tokenizer, args.max_ent_len)
            entity_dict = tokenizer.encode_plus(window,
                                                add_special_tokens=True,
                                                max_ent_length=args.max_ent_len,
                                                pad_to_max_ent_length=True,
                                                truncation=True)
            field['wikipedia_id'] = item['wikipedia_id']
            field['title'] = item['wikipedia_title']
            field['text_ids'] = entity_dict['input_ids']
            field['text_masks'] = entity_dict['attention_mask']
            fout.write('%s\n' % json.dumps(field))

    fout.close()
def processKB(data,batch_size,model,device):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    entity_embeds = []
    masks=[]
    embeddings = []
    id_to_index = {}
    ind=0
    for key in tqdm(data):
        entembed,mask=get_entity_window(data[key],tokenizer,128)
        entity_embeds.append(entembed)
        masks.append(mask)
        id_to_index[ind]=key
        if len(entity_embeds)==batch_size:
            em = model.encode(entity_token_ids=torch.tensor(entity_embeds, device=device),
                              entity_masks=torch.tensor(masks, device=device))
            embeddings.extend(em[2].tolist())
            entity_embeds=[]
            masks=[]
        ind+=1

    if len(entity_embeds) > 0:
        em = model.encode(entity_token_ids=torch.tensor(entity_embeds, device=device),
                          entity_masks=torch.tensor(masks, device=device))
        embeddings.extend(em[2].tolist())
    x_dim = len(embeddings)
    y_dim = len(embeddings[0])
    vectors = numpy.zeros((x_dim, y_dim), dtype=numpy.float32)

    for i in range(0, len(embeddings)):
        vectors[i] = numpy.asarray(embeddings[i])
    print("start indexing")
    index = DenseHNSWFlatIndexer(vector_sz=y_dim)
    index.index_data(vectors)
    pickle.dump(id_to_index, open("id-to-index-entqa.pkl", 'wb'))
    print(index.index.ntotal)
    faiss.write_index(index.index, "faiss-hswf-entqa")



device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
ds=pickle.load(open("../wikipedia_data.pkl","rb"))
model=load_model("retriever.pt",device)
processKB(ds,10,model,device)
