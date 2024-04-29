from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
import torch
model = T5ForConditionalGeneration.from_pretrained("../JointModel/aida-50ep")
tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",)
max_seq_length=200
seq=tokenizer.tokenize("[START_ENT] Paderborn [END_ENT] [ Paderborn ] was founded as a bishopric by [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] in 795, although its official history began in 777 when [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] built a castle near the [START_ENT] Paderborn [END_ENT] [ Paderborn ] springs.[4] In 799 [START_ENT] Pope Leo III [END_ENT] [ Pope Leo III ] fled his enemies in [START_ENT] Rome [END_ENT] [ Rome ] and reached [START_ENT] Paderborn [END_ENT] [ Paderborn ], where he met [START_ENT] Charlemagne [END_ENT] [ Charlemagne ], and stayed there for three months. It was during this time that it was decided that [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] would be crowned emperor. [START_ENT] Charlemagne [END_ENT] [ Charlemagne ] reinstated [START_ENT] Leo [END_ENT] [ Pope Leo I ] in [START_ENT] Rome [END_ENT] [ Rome ] in 800 and was crowned as [START_ENT] Holy Roman Emperor [END_ENT] [ Holy Roman Emperor ] by [START_ENT] Leo [END_ENT] [ Pope Leo I ] in return.")
test=tokenizer.tokenize("[START_ENT]")
test_en=tokenizer.encode("[START_ENT]")
device="cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)


