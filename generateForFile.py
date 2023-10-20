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
model = T5ForConditionalGeneration.from_pretrained("out-graph-data/checkpoint-34000")

dg=nifDataHandlers.Dataprocessor(tokenizer, params)

#tokenizer = T5Tokenizer.from_pretrained("out-lcquad-triples")
#dp=nifDataHandlers.Dataprocessor(tokenizer,params)
dataset_names = ["ACE2004"]
dataset_names.append("aida_testa")
dataset_names.append("aida_testb")
dataset_names.append("aida_train")
dataset_names.append("aida_complete")
dataset_names.append("AQUAINT")
dataset_names.append("spotlight")
dataset_names.append("iitb-fix")
dataset_names.append("KORE50")
dataset_names.append("MSNBC")
dataset_names.append("N3-Reuters-128")
dataset_names.append("N3-RSS-500")
dataset_names.append("oke-challenge-task1-eval")
dataset_names.append("oke-challenge-task1-example")
dataset_names.append("oke-challenge-task1-gs")

for ds in dataset_names:
    data=pickle.load(open("graph-data/graph_samples_"+ds+".pkl","rb"))
    samples = dg.process_training_ds_prebuild("graph-data/graph_samples_"+ds+".pkl")

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
