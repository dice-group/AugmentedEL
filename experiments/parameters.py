import argparse
import os

class ELParser(argparse.ArgumentParser):


    def __init__(
        self,
        description=' ',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
        )
        self.add_arg = self.add_argument

        self.overridable = {}


        self.add_model_args()

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")

        parser.add_argument(
            "--llama_model",
            default="llama3:70b",
            type=str,
            help="Which ollama model to use for the augmentation",
        )
        parser.add_argument(
            "--e2e_model_path",
            default='../joint_model/e2e_aida',
            type=str,
            help="Path to the end to end foundational model",
        )
        parser.add_argument(
            "--joint_model_path",
            default='../joint_model/aida-125ep',
            type=str,
            help="path to the joint foundational model",
        )
        parser.add_argument(
            "--wikipedia_dictionary",
            default="../data/wikidata_uris.pkl",
            type=str,
            help="dictionary for wikipedia dictionary",
        )

        parser.add_argument(
            "--model_name",
            default="t5-small",
            type=str,
            help="path or name of model",
        )
        parser.add_argument(
            "--eval_beams",
            default=None,
            type=int,
            help="# num_beams to use for evaluation.",
        )
