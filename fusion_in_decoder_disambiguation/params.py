import argparse
import os

class Fusion_In_Decoder_Parser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.
    """

    def __init__(
        self, add_optim_options=False, add_eval_options=False,add_reader_options=False,
        description=' ',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        #os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}
        self.initialize_parser()
        if add_optim_options:
            self.add_optim_options()
        if add_eval_options:
            self.add_eval_options()
        if add_reader_options:
            self.add_reader_options()

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

    def add_optim_options(self):
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--total_steps', type=int, default=1000)
        parser.add_argument('--scheduler_steps', type=int, default=None,
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        parser.add_argument('--accumulation_steps', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument('--optim', type=str, default='adam')
        parser.add_argument('--scheduler', type=str, default='fixed')
        parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument('--write_results', action='store_true', help='save results')
        parser.add_argument('--write_crossattention_scores', action='store_true',
                        help='save dataset with cross-attention scores')

    def add_reader_options(self):
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        parser.add_argument('--model_size', type=str, default='base')
        parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        parser.add_argument('--text_maxlength', type=int, default=200,
                        help='maximum number of tokens in text segments (question+passage)')
        parser.add_argument('--answer_maxlength', type=int, default=-1,
                        help='maximum number of tokens used to train the model, no truncation if -1')
        parser.add_argument('--no_title', action='store_true',
                        help='article titles not included in passages')
        parser.add_argument('--n_context', type=int, default=1)



    def initialize_parser(self):
        parser = self.add_argument_group("Model Arguments")
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        parser.add_argument('--model_path', type=str, default='none', help='path for retraining')

        # dataset parameters
        parser.add_argument("--per_gpu_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
        parser.add_argument('--maxload', type=int, default=-1)

        parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        parser.add_argument('--eval_freq', type=int, default=500,
                        help='evaluate model every <eval_freq> steps during training')
        parser.add_argument('--save_freq', type=int, default=5000,
                        help='save model every <save_freq> steps during training')
        parser.add_argument('--eval_print_freq', type=int, default=1000,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')
