CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29001 examples/reranker/reranker_train.py \
  --output_dir reranker_msmarco \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 16 \
  --train_group_size 8 \
  --learning_rate 5e-6 \
  --rerank_max_len 256 \
  --num_train_epochs 5 \
  --logging_steps 500 \
  --dataloader_num_workers 2
