import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import json

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from dataclasses import dataclass, field
from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import EncodeCollator
from tevatron.retriever.modeling import EncoderOutput
from train_splade_llm import SpladeDataArguments, SpladeLlmModel, SpladeTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class SpladeEncodeCollator(EncodeCollator):

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        content_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[2] for x in features] # this will be ignored
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_inputs = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.query_suffix:
            query_suffix = self.data_args.query_suffix.replace("\\n", "\n")
            collated_inputs['input_ids'] = [x + self.tokenizer.encode(query_suffix, add_special_tokens=False) for x in collated_inputs['input_ids']]

        if self.data_args.passage_suffix:
            collated_inputs['input_ids'] = [x + self.tokenizer.encode(query_suffix, add_special_tokens=False) for x in collated_inputs['input_ids']]

        if self.data_args.append_eos_token and not self.data_args.encode_is_query:
            collated_inputs['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_inputs['input_ids']]

        collated_inputs = self.tokenizer.pad(
            collated_inputs,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return content_ids, collated_inputs


def main():
    parser = HfArgumentParser((ModelArguments, SpladeDataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: SpladeDataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    model = SpladeLlmModel.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
    )
    model.TOPK = training_args.topk
    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = SpladeEncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    collection_file = open(data_args.encode_output_path, "w")
    unk_token_id = 0
    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    model_output: EncoderOutput = model(query=batch)
                    reps = model_output.q_reps.cpu().detach().numpy()
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    reps = model_output.p_reps.cpu().detach().numpy()
                for rep, id_ in zip(reps, batch_ids):
                    idx = np.nonzero(rep)
                    # then extract values:
                    data = rep[idx]
                    data = np.rint(data * 100).astype(int) #if not data_args.encode_is_query else np.rint(data * 1).astype(int)
                    dict_splade = dict()
                    for id_token, value_token in zip(idx[0],data):
                        if value_token > 0:
                            if id_token not in vocab_dict:
                                print("unknown token id =>", id_token)
                                vocab_dict[id_token] = f'<|unknown_token_{unk_token_id}|>'
                                unk_token_id += 1
                            real_token = vocab_dict[id_token]
                            dict_splade[real_token] = int(value_token)
                    if len(dict_splade.keys()) == 0:
                        print("empty input =>", id_)
                        dict_splade['nan'] = 1  # in case of empty doc we fill with "[unused993]" token (just to fill
                        # and avoid issues with anserini), in practice happens just a few times ...
                    if not data_args.encode_is_query:
                        dict_ = dict(id=id_, content="", vector=dict_splade)
                        json_dict = json.dumps(dict_)  
                        collection_file.write(json_dict + "\n")
                    else:
                        string_splade = " ".join(
                            [" ".join([str(real_token)] * freq) for real_token, freq in dict_splade.items()])
                        collection_file.write(str(id_) + "\t" + string_splade + "\n")
    collection_file.close()




if __name__ == "__main__":
    main()
