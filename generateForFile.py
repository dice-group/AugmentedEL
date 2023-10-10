from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
import nifDataHandlers
from parameters import ELParser
parser = ELParser(add_model_args=True,add_training_args=True)
parser.add_model_args()
import pickle

# args = argparse.Namespace(**params)
args = parser.parse_args()
print(args)
params = args.__dict__

tokenizer = AutoTokenizer.from_pretrained(
        "t5-small",
    )
#tokenizer = T5Tokenizer.from_pretrained("out-lcquad-triples")
#dp=nifDataHandlers.Dataprocessor(tokenizer,params)
data=pickle.load(open("graph_samples_core50.pkl","rb"))
model = T5ForConditionalGeneration.from_pretrained("out-graph-data/checkpoint-34000")

dg=nifDataHandlers.Dataprocessor(tokenizer, params)
samples = dg.process_training_ds_prebuild("graph_samples_core50.pkl")

for i in range(0,len(samples)):
    #input = sample["source"]

    # sample = dp.process_sample(input,labels)
    #encoding = tokenizer(text="test", text_target=None, return_tensors="pt",
    #                    )
    #i = encoding.input_ids
    # l=dp.process_sample(labels).input_ids
    # the forward function automatically creates the correct decoder_input_ids
    ins=(samples[i]["input_ids"]).view(1,400)
    out = model.generate(input_ids=ins, max_length=400)
    #print(input)
    print(data[i]["source"])
    print(data[i]["target"])
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print("\n")
