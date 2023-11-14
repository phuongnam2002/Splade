import torch
import wandb
import statistics
from typing import Optional
from tqdm import tqdm, trange
from lion_pytorch import Lion
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from utils.io import load_json
from utils.utils import logger
from utils.metrics import recall
from utils.normalize import preprocessing
from components.losses.flops import Flops
from components.models.splade import Splade
from utils.sparse_scores import sparse_scores
from utils.early_stopping import EarlyStopping
from retriever.splade_retriever import SpladeRetriever
from components.losses.ranking_loss import RankingLoss


class SpladeTrainer:
    def __init__(
            self,
            args,
            train_dataset: Optional[Dataset] = None,
            model: Optional[Splade] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.args = args

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

        self.flops = Flops()
        self.ranking_loss = RankingLoss()

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )

        if self.args.max_steps > 0:
            total = self.args.max_steps
            self.args.num_train_epochs = (
                    self.args.max_steps
                    // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                    + 1
            )
        else:
            total = (
                    len(train_dataloader)
                    // self.args.gradient_accumulation_steps
                    * self.args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = self.get_optimizer()

        scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        # Automatic Mixed Precision
        scaler = torch.cuda.amp.GradScaler()

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, position=0, leave=True)
            logger.info(f"Epoch {_}")

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)

                anchors = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1]
                }
                positives = {
                    "input_ids": batch[2],
                    "attention_mask": batch[3]
                }

                anchor_activations = self.model(**anchors)
                positive_activations = self.model(**positives)

                with torch.cuda.amp.autocast():
                    scores = sparse_scores(
                        anchor_activations=anchor_activations["sparse_activations"],
                        positive_activations=positive_activations["sparse_activations"],
                    )

                    sparse_loss = self.ranking_loss(**scores)
                    flop_loss = self.flops(
                        anchor_activations=anchor_activations["sparse_activations"],
                        positive_activations=positive_activations["sparse_activations"],
                        threshold=self.args.threshold_flops,
                    )

                    loss = self.args.sparse_loss_weight * sparse_loss + self.args.flops_loss_weight * flop_loss

                wandb.log({"Train Loss": loss.item()})
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()

                    self.model.zero_grad()
                    global_step += 1

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    logger.info(f"Tuning metrics: {self.args.tuning_metric}")
                    results = self.evaluate_on_benchmark()
                    for k, v in results.items():
                        results[k] = statistics.mean(v)

                    wandb.log({"Eval": results})
                    early_stopping(results[self.args.tuning_metric], self.model, self.args)
                    if early_stopping.early_stop:
                        logger.info('Early stopping')
                        break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break

        return

    def evaluate_on_benchmark(
            self,
            top_k_results=None
    ):
        if top_k_results is None:
            top_k_results = [5, 10, 20, 50, 100]

        results = {}

        self.model.eval()

        for paths in self.args.benchmark_dir:
            (benchmark_path, corpus_path) = paths
            name_benchmark = benchmark_path.split('/')[-1].split('.')[0]
            benchmark = load_json(benchmark_path)
            corpus = load_json(corpus_path)

            retriever = SpladeRetriever(key="meta", on="text", model=self.model, args=self.args)
            retriever.add(documents=corpus, k_tokens=self.args.k_tokens, bath_size=self.args.eval_batch_size)

            for id, metadata in enumerate(benchmark):
                query = preprocessing(metadata['query'])
                gts = metadata['gt']

                result = retriever(query, k_tokens=self.args.k_tokens, batch_size=self.args.eval_batch_size)

                pred_ids = [pred['id'] for pred in result[0]]

                for k in top_k_results:
                    if f"recall_{name_benchmark}_{k}" not in results:
                        results[f"recall_{name_benchmark}_{k}"] = []

                    results[f"recall_{name_benchmark}_{k}"].append(recall(pred_ids[:k], gts))

        return results

    def get_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Lion(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate
        )
        return optimizer
