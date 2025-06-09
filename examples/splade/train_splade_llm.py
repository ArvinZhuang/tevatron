import logging
import os
import sys
from typing import Dict, Optional, List
from uu import encode
import yaml


import torch
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import (
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.retriever.modeling import SpladeModel
from tevatron.retriever.modeling.encoder import EncoderOutput
from tevatron.retriever.trainer import TevatronTrainer


from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import TrainDataset, MultiTrainDataset
from tevatron.retriever.collator import TrainCollator


from tevatron.retriever.trainer import TevatronTrainer as Trainer

logger = logging.getLogger(__name__)


class SpladeLlmModel(SpladeModel):
    TRANSFORMER_CLS = AutoModelForCausalLM

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


    def encode_query(self, qry):
        # reps = torch.zeros((qry['input_ids'].shape[0], self.encoder.vocab_size),
        #                    dtype=self.encoder.dtype,
        #                    device=qry['input_ids'].device)
        # sequence_lengths = qry['attention_mask'].sum(dim=1)
        # for i, length in enumerate(sequence_lengths):
        #     reps[i][qry['input_ids'][i][:length]] = 1
        # return reps
        return self.encode_passage(qry)


    def encode_passage(self, passage):
        passage_out = self.encoder(**passage, return_dict=True).logits

        if self.pooling in ('eos', 'last'):
            left_padding = (passage['attention_mask'][:, -1].sum() == passage['attention_mask'].shape[0])
            if left_padding:
                reps = passage_out[:, -1]
            else:
                sequence_lengths = passage['attention_mask'].sum(dim=1) - 1
                batch_size = passage_out.shape[0]
                reps = passage_out[torch.arange(batch_size, device=passage_out.device), sequence_lengths]
            reps = torch.log(1 + torch.relu(reps))
        elif self.pooling == 'max':
            passage_out = self.encoder(**passage, return_dict=True).logits
            reps, _ = torch.max(torch.log(1 + torch.relu(passage_out)) * passage['attention_mask'].unsqueeze(-1),
                                              dim=1)
        else:
            raise ValueError(f"Pooling method {self.pooling} not supported")
        return reps


    def _topk_token_mask(self, reps, topk_indices):
        mask = torch.zeros_like(reps, dtype=torch.bool)

        # Scatter True at the top-k indices
        mask.scatter_(1, topk_indices, 1)
        reps = reps * mask.float()
        return reps

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            reps = []
            if q_reps is not None:
                reps.append(q_reps.clone())
            if p_reps is not None:
                reps.append(p_reps.clone())

            if self.topks is not None:
                max_topk = self.topks[0]
                if q_reps is not None:
                    _, max_q_topk_indices = torch.topk(q_reps, max_topk, dim=1)
                    q_reps = self._topk_token_mask(q_reps, max_q_topk_indices)
                if p_reps is not None:
                    _, max_p_topk_indices = torch.topk(p_reps, max_topk, dim=1)
                    p_reps = self._topk_token_mask(p_reps, max_p_topk_indices)

                for topk in self.topks:
                    if q_reps is not None:
                        q_reps = self._topk_token_mask(q_reps, max_q_topk_indices[:, :topk])
                        reps.append(q_reps.clone())
                    if p_reps is not None:
                        p_reps = self._topk_token_mask(p_reps, max_p_topk_indices[:, :topk])
                        reps.append(p_reps.clone())

            q_reps = reps if q_reps is not None else None
            p_reps = reps if p_reps is not None else None

            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )


        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)

            if self.topks is not None:
                max_topk = self.topks[0]
                if q_reps is not None:
                    _, max_q_topk_indices = torch.topk(q_reps, max_topk, dim=1)
                    q_reps = self._topk_token_mask(q_reps, max_q_topk_indices)
                if p_reps is not None:
                    _, max_p_topk_indices = torch.topk(p_reps, max_topk, dim=1)
                    p_reps = self._topk_token_mask(p_reps, max_p_topk_indices)

                for weight, topk in zip(self.weights, self.topks):
                    q_reps = self._topk_token_mask(q_reps, max_q_topk_indices[:, :topk])
                    p_reps = self._topk_token_mask(p_reps, max_p_topk_indices[:, :topk])
                    scores = self.compute_similarity(q_reps, p_reps)
                    scores = scores.view(q_reps.size(0), -1)

                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    target = target * (p_reps.size(0) // q_reps.size(0))
                    topk_loss = self.compute_loss(scores / self.temperature, target) * weight
                    loss = loss + topk_loss

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
class SpladeTrainCollator(TrainCollator):
    def __call__(self, features):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        all_queries = [q[0].replace('\\n', '\n') for q in all_queries]
        all_passages = [p[0].replace('\\n', '\n') for p in all_passages]
        q_collated = self.tokenizer(
            all_queries,
            padding=False,
            truncation=True,
            max_length=self.data_args.query_max_len - 1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=False,
            truncation=True,
            max_length=self.data_args.passage_max_len - 1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.query_suffix:
            query_suffix = self.data_args.query_suffix.replace("\\n", "\n")
            query_suffix_ids = self.tokenizer.encode(query_suffix, add_special_tokens=False)
            q_collated['input_ids'] = [q + query_suffix_ids for q in q_collated['input_ids']]

        if self.data_args.passage_suffix:
            passage_suffix = self.data_args.passage_suffix.replace("\\n", "\n")
            passage_suffix_ids = self.tokenizer.encode(passage_suffix, add_special_tokens=False)
            d_collated['input_ids'] = [d + passage_suffix_ids for d in d_collated['input_ids']]

        if self.data_args.append_eos_token:
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]

        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return q_collated, d_collated

@dataclass
class SpladeTrainingArguments(TevatronTrainingArguments):
    q_flops_loss_factor: float = field(default=4)
    p_flops_loss_factor: float = field(default=32)
    topks: List[int] = field(default=None)

@dataclass
class SpladeDataArguments(DataArguments):
    query_suffix: str = field(
        default='', metadata={"help": "suffix or instruction for query"}
    )

    passage_suffix: str = field(
        default='', metadata={"help": "suffix or instruction for passage"}
    )


class SpladeTrainer(TevatronTrainer):
    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # if self.state.global_step / self.state.max_steps > 0.1 and model.TOPK is None:
        #     model.TOPK = 256

        query, passage = inputs
        output = model(query=query, passage=passage)
        q_reps = output.q_reps
        p_reps = output.p_reps
        loss = output.loss


        # q_flops_loss_factor_coeff = self.state.global_step / 4000 if self.state.global_step < 4000 else 1
        # p_flops_loss_factor_coeff = self.state.global_step / 4000 if self.state.global_step < 4000 else 1

        q_flops_loss = self.args.q_flops_loss_factor*self._flops(q_reps) if self.args.q_flops_loss_factor != 0 else 0
        p_flops_loss = self.args.p_flops_loss_factor*self._flops(p_reps) if self.args.p_flops_loss_factor != 0 else 0

        # q_flops_loss = q_flops_loss * q_flops_loss_factor_coeff
        # p_flops_loss = p_flops_loss * p_flops_loss_factor_coeff

        if self.is_ddp:
            q_flops_loss *= self._dist_loss_scale_factor
            p_flops_loss *= self._dist_loss_scale_factor
        return loss + q_flops_loss + p_flops_loss


TrainingArguments = SpladeTrainingArguments

def main():
    parser = HfArgumentParser((ModelArguments, SpladeDataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: SpladeDataArguments
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

    model = SpladeLlmModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
    )

    if training_args.topks is not None:
        model.set_topks(training_args.topks)

    if data_args.train_yaml is not None:
        with open(data_args.train_yaml, 'r') as f:
            train_yaml = yaml.safe_load(f)
        dataset_list = train_yaml['train']
        corpus_list = train_yaml['corpus']

    train_dataset = MultiTrainDataset(data_args, dataset_list, corpus_list) if data_args.train_yaml is not None else TrainDataset(data_args)
    collator = SpladeTrainCollator(data_args, tokenizer)

    trainer = SpladeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )

    train_dataset.set_trainer(trainer)

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
