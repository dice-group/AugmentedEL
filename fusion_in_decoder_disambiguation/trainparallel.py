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
import torch.distributed as dist
from tqdm import tqdm
import os

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

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
    #sampler = SequentialSampler(dataset)
    sampler=DistributedSampler(dataset, num_replicas=world_size, rank=LOCAL_RANK, shuffle=True, drop_last=False)
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
        for batch in tqdm(dataloader):
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

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, params, collator, best_dev_em, checkpoint_path):
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    model.to(device)
    #if params["is_main"]:

    tb_logger = True
    #else:
    #    tb_logger = False
    torch.manual_seed(0 + params["seed"]) #different seed for different sampling depending on global_rank
    train_sampler=DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=True, drop_last=False)
    #train_sampler = RandomSampler(train_dataset)
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

                dist.barrier()
                if WORLD_RANK == 0:
                    dev_em_values = [None for _ in range(WORLD_SIZE)]
                    dist.all_gather_object(dev_em_values, dev_em)
                    print("dev_em_values: "+str(dev_em_values))
                    avg_dev_em=sum(dev_em_values)/WORLD_SIZE
                    print("avg_dev_em: " + str(avg_dev_em))
                    model.train()
                    if avg_dev_em > best_dev_em:
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
                dist.barrier()
                #if params["is_main"] and step % params["save_freq"] == 0:
                if WORLD_RANK == 0:
                    util.save(model, optimizer, scheduler, step, best_dev_em,
                          params, checkpoint_path, f"step-{step}")
            if step >params["total_steps"]:
                dist.barrier()
                break

if __name__ == '__main__':
    '''
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    '''
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
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    train(0,model,optimizer,scheduler,step,train_samples,test_samples,params,collator,best_dev_em,params["checkpoint_dir"])
    '''
    mp.spawn(
        train,
        args=(model,optimizer,scheduler,step,train_samples,test_samples,params,collator,best_dev_em,params["checkpoint_dir"]),
        nprocs=world_size
    )
    '''