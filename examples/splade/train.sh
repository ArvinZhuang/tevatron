python3 train_splade_llm.py \
  --output_dir checkpoints/Qwen/Qwen2.5-1.5B-Instruct \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --save_steps 5000 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --passage_prefix "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nUse one word to represent this passage: " \
  --passage_suffix "<|im_end|>\n<|im_start|>assistant\nThe word is: \"" \
  --query_prefix "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nUse one word to represent this query: " \
  --query_suffix "<|im_end|>\n<|im_start|>assistant\nThe word is: \""  \
  --bf16 \
  --per_device_train_batch_size 4 \
  --train_group_size 8 \
  --learning_rate 5e-6 \
  --query_max_len 128 \
  --passage_max_len 256 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 1 \
  --dataloader_num_workers 8 \
  --num_proc 8 \
  --logging_steps 100 \
  --report_to wandb \
  --attn_implementation sdpa \
  --overwrite_output_dir


python3 train_splade_llm.py \
  --output_dir checkpoints/meta-llama/Llama-3.2-1B \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --save_steps 5000 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --passage_prefix "You are a helpful assistant.\nUser: Use one word to represent this passage: \"" \
  --passage_suffix "\"\n\nAssistant: The word is: \"" \
  --query_prefix "You are a helpful assistant.\nUser: Use one word to represent this query: \"" \
  --query_suffix "\"\n\nAssistant: The word is: \"" \
  --bf16 \
  --per_device_train_batch_size 4 \
  --train_group_size 8 \
  --learning_rate 5e-6 \
  --query_max_len 128 \
  --passage_max_len 256 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 1 \
  --dataloader_num_workers 8 \
  --num_proc 8 \
  --logging_steps 100 \
  --report_to wandb \
  --attn_implementation sdpa \
  --overwrite_output_dir