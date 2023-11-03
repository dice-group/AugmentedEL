import torch
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import util
import torch.multiprocessing as mp
import model
import transformers
from params import Fusion_In_Decoder_Parser
def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, params, collator, best_dev_em, checkpoint_path,rank,world_size):
    model.to(rank)
    if params["is_main"]:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(params["checkpoint_dir"])/params["name"])
        except:
            tb_logger = None
            print('Tensorboard is not available.')

    torch.manual_seed(params["global_rank"].global_rank + params["seed"]) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < params["total_steps"]:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
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
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, params)
                model.train()
                if params["is_main"]:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        util.save(model, optimizer, scheduler, step, best_dev_em,
                                params, checkpoint_path, 'best_dev')
                    log = f"{step} / {params['total_steps']} |"
                    log += f"train: {curr_loss/params['eval_freq']:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    print(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (params["eval_freq"]), step)
                    curr_loss = 0.

            if params["is_main"] and step % params["save_freq"] == 0:
                util.save(model, optimizer, scheduler, step, best_dev_em,
                          params, checkpoint_path, f"step-{step}")
            if step >params["total_steps"]:
                break

if __name__ == '__main__':
    parser = Fusion_In_Decoder_Parser(add_model_args=True, add_training_args=True)
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
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    # suppose we have 3 gpus
    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(
        train,
        args=(world_size,10),
        nprocs=world_size
    )