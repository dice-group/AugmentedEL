
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
import nifDataHandlers
from parameters import ELParser
parser = ELParser(add_model_args=True,add_training_args=True)
parser.add_model_args()
args = parser.parse_args()
print(args)
params = args.__dict__
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

tokenizer = AutoTokenizer.from_pretrained(
        "t5-small",
    )
dg=nifDataHandlers.Dataprocessor(tokenizer, params)
for ds in dataset_names:
    samples = dg.process_training_ds(ds,"candidateds/candidatest"+ds+".pkl")