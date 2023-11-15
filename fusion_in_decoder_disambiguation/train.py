import torch
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import util
import torch.multiprocessing as mp
import model
import transformers
from Collator import Collator
from params import Fusion_In_Decoder_Parser
import regex
import string
import numpy as np
import pickle
from tqdm import tqdm
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])
def evaluate(model, dataset, tokenizer, collator, params,device):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=params["per_gpu_batch_size"],
        drop_last=False,
        collate_fn=collator.collate
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (_, _, context_ids, context_mask,target) = batch

            outputs = model.generate(
                input_ids=context_ids.to(device),
                attention_mask=context_mask.to(device),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = target
                score = ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = util.weighted_average(np.mean(exactmatch), total, params)
    return exactmatch

def train(rank,model, optimizer, scheduler, step, train_dataset, eval_dataset, params, collator, best_dev_em, checkpoint_path):
    model.to(device)
    #if params["is_main"]:

    tb_logger = True
    #else:
    #    tb_logger = False
    rank=device
    torch.manual_seed(0 + params["seed"]) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=params["per_gpu_batch_size"],
        drop_last=True,
        #num_workers=10,
        collate_fn=collator.collate
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < params["total_steps"]:
        epoch += 1
        for batch in tqdm(train_dataloader):
            step += 1
            (labels, _, context_ids, context_mask,_) = batch

            train_loss = model(
                input_ids=context_ids.to(device),
                attention_mask=context_mask.to(device),
                labels=labels.to(device),
                return_dict=False
            )[0]

            train_loss.backward()

            if step % params["accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip"])
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = util.average_main(train_loss, params)
            curr_loss += train_loss.item()

            if step % params["eval_freq"] == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, params,device)
                model.train()
                if dev_em > best_dev_em:
                    best_dev_em = dev_em
                    util.save(model, optimizer, scheduler, step, best_dev_em,
                                params, checkpoint_path, 'best_dev')
                log = f"{step} / {params['total_steps']} |"
                log += f"train: {curr_loss/params['eval_freq']:.3f} |"
                log += f"evaluation: {100*dev_em:.2f}EM |"
                log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                print(log)
                if tb_logger:
                    print("Evaluation", dev_em, step)
                    print("Training", curr_loss / (params["eval_freq"]), step)
                curr_loss = 0.
            if step % params["save_freq"] == 0:
            #if params["is_main"] and step % params["save_freq"] == 0:
                util.save(model, optimizer, scheduler, step, best_dev_em,
                          params, checkpoint_path, f"step-{step}")
            if step >params["total_steps"]:
                break

if __name__ == '__main__':
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
    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    #model = model.to(opt.local_rank)
    optimizer, scheduler = util.set_optim(params, model)
    step, best_dev_em = 0, 0.0
    # load data
    train_samples=pickle.load(open("../fusionInDecoding/dataaida_train.pkl","rb"))
    test_samples = pickle.load(open("../fusionInDecoding/dataaida_testa.pkl", "rb"))
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    world_size = torch.cuda.device_count()
    print(world_size)
    collator=Collator(tokenizer, params)
    train(0,model,optimizer,scheduler,step,train_samples,test_samples,params,collator,best_dev_em,params["checkpoint_dir"])
    '''
    mp.spawn(
        train,
        args=(model,optimizer,scheduler,step,train_samples,test_samples,params,collator,best_dev_em,params["checkpoint_dir"]),
        nprocs=world_size
    )
    '''