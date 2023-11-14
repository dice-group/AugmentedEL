import torch
from params import Fusion_In_Decoder_Parser
import transformers
import pickle

class Collator():
    def __init__(self, tokenizer, params,kb_data="../wikipedia_data.pkl"):
        self.tokenizer = tokenizer
        self.params=params
        self.data=pickle.load(open(kb_data,"rb"))


    def encode_passages(self,batch_text_passages):
        passage_ids, passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = self.tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.params["text_maxlength"],
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids, passage_masks.bool()

    def collate(self, batch):
        assert(batch[0]['target'] != None)
        #index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.params["answer_maxlength"],
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_candidates(example):
            if example['candidates'] is None or len(example['candidates'] )== 0:
                return [example['source']]
            return [example['source'] + " " + "title: "+self.data[cand]["title"]+" abstract: "
                    +self.data[cand]["abstract"] for cand in example['candidates']]
        text_passages = [append_candidates(example) for example in batch]
        passage_ids, passage_masks = self.encode_passages(text_passages)
        return (target_ids, target_mask, passage_ids, passage_masks,target)

'''
parser = Fusion_In_Decoder_Parser()
parser.add_reader_options()
parser.add_optim_options()
parser.add_eval_options()

# args = argparse.Namespace(**params)
args = parser.parse_args()
print(args)
params = args.__dict__
tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base")
samples=pickle.load(open("../fusionInDecoding/dataaida_train.pkl","rb"))
c=Collator(tokenizer,params)
c.__call__([samples[0]])
'''