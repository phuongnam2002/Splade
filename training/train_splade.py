import os
import math
import torch
import wandb
import argparse

from transformers import set_seed

from trainer.train_splade import SpladeTrainer
from components.loaders.dataloaders import OnlineDataset
from utils.utils import MODEL_CLASSES, MODEL_PATH_MAP, load_tokenizer, logger


def main(args):
    logger.info("Args={}".format(str(args)))

    set_seed(args.seed)

    # Pre Setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = load_tokenizer(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]

    if args.pretrained:
        print("Loading model ....")
        model = model_class.from_pretrained(
            args.pretrained_path,
            tokenizer=tokenizer,
            args=args,
        )
    else:
        model_config = config_class.from_pretrained(
            args.model_name_or_path, finetuning_task=args.token_level
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=model_config,
            tokenizer=tokenizer,
            args=args,
        )

    model.to(args.device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(model)
    logger.info(model.dtype)
    logger.info("Vocab size: {}".format(len(tokenizer)))

    # Load data
    train_dataset = OnlineDataset(args, tokenizer, "train")

    trainer = SpladeTrainer(
        args=args,
        model=model.to(args.device),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    import statistics

    results = trainer.evaluate_on_benchmark()
    for k, v in results.items():
        results[k] = statistics.mean(v)

    print(results)

    if args.do_train:
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )

        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default=None,
        required=True,
        type=str,
        help="Path to save, load model",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir",
    )
    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--model_type",
        default="phobert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=200,
        help="Number of steps between each logging update.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between each model evaluation.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Number of steps between each checkpoint saving.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="semantic-similarity",
        help="Name of the Weight and Bias project.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="test-source",
        help="Name of the run for Weight and Bias.",
    )
    parser.add_argument(
        "--wandb_watch",
        type=str,
        default="false",
        help="Whether to enable tracking of gradients and model topology in Weight and Bias.",
    )
    parser.add_argument(
        "--wandb_log_model",
        type=str,
        default="false",
        help="Whether to enable model versioning in Weight and Bias.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Flag indicating whether to run the training process.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to initialize the model from a pretrained base model.",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Path to the pretrained model.",
    )

    # CUDA Configuration
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Flag indicating whether to avoid using CUDA when available.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="ID of the GPU to be used for computation.",
    )

    # Hyperparameters for training
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for initialization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Batch size used for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--k_tokens",
        default=256,
        type=int,
        help="Top_k maximum tokens after Splade",
    )
    parser.add_argument(
        "--threshold_flops",
        default=10.0,
        type=float,
    )
    parser.add_argument(
        "--sparse_loss_weight",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--flops_loss_weight",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--dataloader_drop_last",
        type=bool,
        default=True,
        help="Toggle whether to drop the last incomplete batch in the dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=bool,
        default=True,
        help="Toggle whether to use pinned memory in the dataloader.",
    )
    # Tokenizer Configuration
    parser.add_argument(
        "--max_seq_len_query",
        default=64,
        type=int,
        help="The maximum total input sequence length for query after tokenization.",
    )
    parser.add_argument(
        "--max_seq_len_document",
        default=256,
        type=int,
        help="The maximum total input sequence length for document after tokenization.",
    )
    parser.add_argument(
        "--max_seq_len_response",
        default=64,
        type=int,
        help="The maximum total input sequence length for response after tokenization.",
    )
    # Optimizer Configuration
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage during training. "
             "When this flag is set, intermediate activations are recomputed during "
             "backward pass, which can be memory-efficient but might increase "
             "training time.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        type=str,
        help="Type of learning rate scheduler to use. Available options are: 'cosine', 'step', 'plateau'. "
             "The default is 'cosine', which uses a cosine annealing schedule. ",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )
    parser.add_argument(
        "--tuning_metric",
        default="loss",
        type=str,
        help="Metrics to tune when training",
    )

    # Model Configuration
    parser.add_argument(
        "--compute_dtype",
        type=torch.dtype,
        default=torch.float,
        help="Used in quantization configs. Do not specify this argument manually.",
    )

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    # Check if parameter passed or if set within environ
    args.use_wandb = len(args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    args.benchmark_dir = [
        [
            "/home/black/saturn/data/benchmark/bm_history_v400.json",
            "/home/black/saturn/data/benchmark/corpus_history.json",
        ],
        # [
        #     "/home/black/saturn/data/benchmark/bm_history_v200.json",
        #     "/home/black/saturn/data/benchmark/corpus_history.json",
        # ],
        # [
        #     "/home/black/saturn/data/benchmark/bm_history_cttgt2.json",
        #     "/home/black/saturn/data/benchmark/corpus_history.json",
        # ],
        # [
        #     "/home/black/saturn/data/benchmark/bm_history_fqa.json",
        #     "/home/black/saturn/data/benchmark/corpus_history.json",
        # ],
        # [
        #     "/home/black/saturn/data/benchmark/bm_visquad_uit.json",
        #     "/home/black/saturn/data/benchmark/corpus_visquad.json"
        # ],
        # [
        #     "/home/black/saturn/data/benchmark/bm_geography_v100.json",
        #     "/home/black/saturn/data/benchmark/corpus_geography.json"
        # ]
    ]
    main(args)
