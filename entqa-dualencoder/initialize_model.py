import json
from transformers import BertTokenizer, BertModel, AdamW, \
    get_linear_schedule_with_warmup, get_constant_schedule
from collections import OrderedDict
import torch
from retriever import DualEncoder

def load_model(model_path, device, type_loss="sum_log_nce",
               ):
    '''
    with open(config_path) as json_file:
        params = json.load(json_file)

    if blink:
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
    else:
        ctxt_bert = BertModel.from_pretrained('bert-large-uncased')
        cand_bert = BertModel.from_pretrained('bert-large-uncased')

    if is_init:
        if blink:
            ctxt_dict = OrderedDict()
            cand_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:26] == 'context_encoder.bert_model':
                    new_k = k[27:]
                    ctxt_dict[new_k] = v
                if k[:23] == 'cand_encoder.bert_model':
                    new_k = k[24:]
                    cand_dict[new_k] = v
            ctxt_bert.load_state_dict(ctxt_dict, strict=False)
            cand_bert.load_state_dict(cand_dict, strict=False)
        model = DualEncoder(ctxt_bert, cand_bert, type_loss)
    else:
    '''
    ctxt_bert = BertModel.from_pretrained('bert-large-uncased')
    cand_bert = BertModel.from_pretrained('bert-large-uncased')
    state_dict = torch.load(model_path) if device.type == 'cuda' else \
        torch.load(model_path, map_location=torch.device('cpu'))
    model = DualEncoder(ctxt_bert, cand_bert, type_loss)
    model.load_state_dict(state_dict['sd'],strict=False)
    return model.to(device)
#device = torch.device(
#            "cuda" if torch.cuda.is_available() else "cpu")
#model=load_model("retriever.pt",device)
#print("finished loading model")
