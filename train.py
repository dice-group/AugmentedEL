import nifDataHandlers
from parameters import ELParser
from typing import Callable, Dict, List, Optional, Tuple, Iterable
import numpy as np
import os

from data_processing import ListDataset,LCqUAD, LCqUAD_rep_vars,LCqUAD_rep_vars_triples
from transformers import Trainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartTokenizer,
    TrainingArguments,
    T5Tokenizer,
    set_seed,
    EvalPrediction,
    AutoModelForSeq2SeqLM
)
def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = ELParser(add_model_args=True,add_training_args=True)
    parser.add_model_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)
    params = args.__dict__


    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        params["config_name"] if params["config_name"] is not None else params["model_name"],
        cache_dir=params["cache_dir"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        params["tokenizer_name"] if params["tokenizer_name"] is not None else params["model_name"],
        cache_dir=params["cache_dir"],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        params["model_name"],
        from_tf=".ckpt" in params["model_name"],
        config=config,
        cache_dir=params["cache_dir"],
    )

    # use task specific params
    # use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if params["eval_beams"] is not None:
        model.config.num_beams = params["eval_beams"]
    assert model.config.num_beams >= 1, f"got eval_beams={model.config.num_beams}. Need an integer >= 1"

    # set max length for generation
    model.config.max_generate_length = params["max_target_length"]

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def non_pad_len(tokens: np.ndarray) -> int:
            return np.count_nonzero(tokens != tokenizer.pad_token_id)

        def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
            pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
            return pred_str, label_str

        # with decoding
        def _exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            pred_str, label_str = decode_pred(pred)
            ex = sum([a == b for (a, b) in zip(pred_str, label_str)]) / len(pred_str)
            result = {'ex': ex}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result

        # without decoding
        def exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            # pred_str, label_str = decode_pred(pred)
            ex = np.sum(np.all(pred.label_ids == pred.predictions, axis=1)) / pred.label_ids.shape[0]
            # for a, b in zip(pred.label_ids, pred.predictions):
            #     print(a)
            #     print(b)
            # exit()
            result = {'ex': ex, 'num_total': pred.label_ids.shape[0]}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result

        # compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
        compute_metrics_fn = exact_match_metrics
        return compute_metrics_fn
    dg=nifDataHandlers.Dataprocessor(tokenizer, params)

    # Get datasets
    if params["train_model"]:
        train_dataset = ListDataset(dg.process_training_ds_prebuild("graph_samples_aida.pkl"))

    else:
        train_dataset = ListDataset(dg.process_training_ds("aida_testa","canidatestest.pkl"))
    eval_dataset = ListDataset(dg.process_training_ds_prebuild("graph_samples_aida_test.pkl"))
    '''
    if training_args.do_eval:
        eval_dataset = ListDataset(load_and_cache_examples(model_args, tokenizer, evaluate=True))
    else:
        eval_dataset = ListDataset([])
    '''
    # Training
    if params["train_model"]:
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(warmup_ratio=params["warmup_ratio"],label_smoothing_factor=params["label_smoothing"],output_dir=params["output_dir"],num_train_epochs=50),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(),
        )

        trainer.train(
            model_path=params["model_name"] if os.path.isdir(params["model_name"]) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(params["output_dir"])

    # prediction
    eval_results = {}
    '''
    if training_args.do_eval:
        logging.info("*** Test ***")

        result = run_prediction(training_args, eval_dataset, model, tokenizer, output_prediction=True)
        # if trainer.is_world_process_zero():
        logger.info("***** Test results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)

        eval_results.update(result)
    return eval_results
    '''


if __name__ == "__main__":
    main()