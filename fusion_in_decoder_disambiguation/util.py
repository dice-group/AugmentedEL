# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist
import csv

logger = logging.getLogger(__name__)



def get_checkpoint_path(params):
    checkpoint_path = Path(params["checkpoint_dir"]) / params["opt.name"]
    checkpoint_exists = checkpoint_path.exists()
    if params["is_distributed"]:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def save(model, optimizer, scheduler, step, best_eval_metric, params, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": params,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def load(model_class, dir_path, params, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path)
    model = model.to(params["device"])
    logger.info("loading checkpoint %s" %optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=params["device"])
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    else:
        best_eval_metric = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(params, model)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(params, model):
    if params["optim"] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    elif params["optim"] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    if params["scheduler"] == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif params["scheduler"] == 'linear':
        if params["scheduler_steps"] is None:
            scheduler_steps = params["total_steps"]
        else:
            scheduler_steps = params["scheduler_steps"]
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=params["warmup_steps"], scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=params["fixed_lr"])
    return optimizer, scheduler


def average_main(x, params):
    if not params["is_distributed"]:
        return x
    if params["world_size"] > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if params["is_main"]:
            x = x / params["world_size"]
    return x


def sum_main(x, params):
    if not params["is_distributed"]:
        return x
    if params["world_size"] > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, params):
    if not params["is_distributed"]:
        return x, count
    t_loss = torch.tensor([x * count], device=params["device"])
    t_total = torch.tensor([count], device=params["device"])
    t_loss = sum_main(t_loss, params)
    t_total = sum_main(t_total, params)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, params):
    dir_path = Path(params["checkpoint_dir"]) / params["name"]
    write_path = dir_path / 'tmp_dir'
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f'{params["global_rank"]}.json'
    with open(tmp_path, 'w') as fw:
        json.dump(data, fw)
    if params["is_distributed"]:
        torch.distributed.barrier()
    if params["is_main"]:
        final_path = dir_path / 'dataset_wscores.json'
        logger.info(f'Writing dataset with scores at {final_path}')
        glob_path = write_path / '*'
        results_path = write_path.glob('*.json')
        alldata = []
        for path in results_path:
            with open(path, 'r') as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, 'w') as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()
