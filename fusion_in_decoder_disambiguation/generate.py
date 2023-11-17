import torch
import transformers
from params import Fusion_In_Decoder_Parser
import model
import pickle
from Collator import Collator

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
parser = Fusion_In_Decoder_Parser()
parser.add_reader_options()
parser.add_optim_options()
parser.add_eval_options()

# args = argparse.Namespace(**params)
args = parser.parse_args()
print(args)
params = args.__dict__
model_name = 't5-' + params["model_size"]
model_class = model.FiDT5
model = model_class.from_pretrained(params["model_path"])
#t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
#model = model.FiDT5(t5.config)
#model.load_t5(t5.state_dict())
model.to(device)
test_samples = pickle.load(open("../fusionInDecoding/dataaida_testa.pkl", "rb"))
tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
collator=Collator(tokenizer, params)
for el in test_samples:
    labels, _, context_ids, context_mask, _=collator.collate([el])
    #print(context_ids)
    res = model.generate(context_ids.to(device),context_mask.to(device),params["answer_maxlength"])
    #print(res)
    print(tokenizer.decode(res[0], skip_special_tokens=True))