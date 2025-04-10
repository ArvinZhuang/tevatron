
import logging
import os
import sys
from typing import Dict
from tevatron.retriever.modeling.encoder import EncoderOutput
from torch import Tensor
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.retriever.modeling import SpladeModel
from tevatron.retriever.trainer import TevatronTrainer
from typing import List

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator

from tevatron.retriever.trainer import TevatronTrainer as Trainer

logger = logging.getLogger(__name__)

class SpladeModelTopK(SpladeModel):
    def __init__(self, topks: List[int] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topks = topks
        self.weights = None

    def set_topks(self, topks: List[int]):
        self.topks = topks
        # sort topks in descending order
        self.topks.sort(reverse=True)
        # set weights for each topk, decreasing, sum to 1
        self.weights = list(range(len(self.topks) + 1, 0, -1))
        self.weights = [w / sum(self.weights) for w in self.weights]


    def _topk_token_mask(self, reps, topk_indices):
        mask = torch.zeros_like(reps, dtype=torch.bool)

        # Scatter True at the top-k indices
        mask.scatter_(1, topk_indices, 1)
        reps = reps * mask.float()
        return reps

    def encode_query(self, qry):
        qry_out = self.encoder(**qry, return_dict=True).logits
        # aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(qry_out)) * qry['attention_mask'].unsqueeze(-1),
        #                                   dim=1)
        aggregated_psg_out, _ = torch.max(torch.log(1 + F.softplus(qry_out)) * qry['attention_mask'].unsqueeze(-1),
                                          dim=1)
        return aggregated_psg_out

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if self.topks is not None:
            max_topk = self.topks[0]
            if q_reps is not None:
                _, max_q_topk_indices = torch.topk(q_reps, max_topk, dim=1)
                q_reps = self._topk_token_mask(q_reps, max_q_topk_indices)
            if p_reps is not None:
                _, max_p_topk_indices = torch.topk(p_reps, max_topk, dim=1)
                p_reps = self._topk_token_mask(p_reps, max_p_topk_indices)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )


        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            loss = 0
            for weight, topk in zip(self.weights, self.topks):
                q_reps = self._topk_token_mask(q_reps, max_q_topk_indices[:, :topk])
                p_reps = self._topk_token_mask(p_reps, max_p_topk_indices[:, :topk])
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))

                loss += self.compute_loss(scores / self.temperature, target) * weight

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

@dataclass
class SpladeTrainingArguments(TevatronTrainingArguments):
    q_flops_loss_factor: float = field(default=4)
    p_flops_loss_factor: float = field(default=32)
    topks: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])


class SpladeTrainer(TevatronTrainer):
    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage = inputs
        output = model(query=query, passage=passage)
        q_reps = output.q_reps
        p_reps = output.p_reps
        loss = output.loss
        q_flops_loss = self.args.q_flops_loss_factor*self._flops(q_reps) if self.args.q_flops_loss_factor != 0 else 0
        p_flops_loss = self.args.p_flops_loss_factor*self._flops(p_reps) if self.args.p_flops_loss_factor != 0 else 0
        if self.is_ddp:
            q_flops_loss *= self._dist_loss_scale_factor
            p_flops_loss *= self._dist_loss_scale_factor
        return loss + q_flops_loss + p_flops_loss


TrainingArguments = SpladeTrainingArguments

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    model = SpladeModelTopK.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
    )
    model.set_topks(training_args.topks)

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)

    trainer = SpladeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
