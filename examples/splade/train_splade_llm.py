import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import (
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.retriever.modeling import SpladeModel
from tevatron.retriever.trainer import TevatronTrainer


from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator

from tevatron.retriever.trainer import TevatronTrainer as Trainer

logger = logging.getLogger(__name__)


class SpladeLlmModel(SpladeModel):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def encode_query(self, qry):
        qry_out = self.encoder(**qry, return_dict=True).logits

        left_padding = (qry['attention_mask'][:, -1].sum() == qry['attention_mask'].shape[0])
        if left_padding:
            reps = qry_out[:, -1]
        else:
            sequence_lengths = qry['attention_mask'].sum(dim=1) - 1
            batch_size = qry_out.shape[0]
            reps = qry_out[torch.arange(batch_size, device=qry_out.device), sequence_lengths]
        reps = torch.log(1 + torch.relu(reps))

        return reps


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
            q_collated['input_ids'] = [q + self.tokenizer.encode(query_suffix) for q in q_collated['input_ids']]

        if self.data_args.passage_suffix:
            passage_suffix = self.data_args.passage_suffix.replace("\\n", "\n")
            d_collated['input_ids'] = [d + self.tokenizer.encode(passage_suffix) for d in d_collated['input_ids']]


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
        query, passage = inputs
        output = model(query=query, passage=passage)
        q_reps = output.q_reps
        p_reps = output.p_reps
        loss = output.loss
        q_flops_loss = self.args.q_flops_loss_factor*self._flops(q_reps)
        p_flops_loss = self.args.p_flops_loss_factor*self._flops(p_reps)
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

    train_dataset = TrainDataset(data_args)
    collator = SpladeTrainCollator(data_args, tokenizer)

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
