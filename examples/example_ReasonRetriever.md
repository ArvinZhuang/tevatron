## Train
```bash
deepspeed --include localhost:0,1 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-qwen3-8b-reasonrank \
  --model_name_or_path Qwen/Qwen3-Embedding-8B \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 100 \
  --dataset_name ArvinZhuang/reasonrank-data-hn \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
  --passage_prefix "" \
  --bf16 \
  --pooling last \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 8 \
  --learning_rate 1e-4 \
  --query_max_len 256 \
  --passage_max_len 512 \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2
```

## BrowseComp-Plus
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-8B \
  --dataset_path data/browsecomp_plus_decrypted.jsonl \
  --lora \
  --lora_name_or_path retriever-qwen3-8b-reasonrank/checkpoint-100 \
  --encode_output_path embeddings/query.pkl \
  --query_max_len 512 \
  --encode_is_query \
  --normalize \
  --pooling eos \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
  --per_device_eval_batch_size 156 \
  --fp16

for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=${s} python -m tevatron.retriever.driver.encode \
  --model_name_or_path Qwen/Qwen3-Embedding-8B \
  --lora \
  --lora_name_or_path retriever-qwen3-8b-reasonrank/checkpoint-100 \
  --dataset_name Tevatron/browsecomp-plus-corpus \
  --encode_output_path embeddings/corpus-${s}.pkl \
  --passage_max_len 512 \
  --normalize \
  --pooling eos \
  --passage_prefix "" \
  --per_device_eval_batch_size 32 \
  --fp16 \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} &
done


python -m tevatron.retriever.driver.search --query_reps embeddings/query.pkl \
--passage_reps embeddings/'corpus-*.pkl' \
--depth 1000 \
--batch_size 128 \
--save_text \
--save_ranking_to runs/qwen3-8b_top1000.txt

python -m tevatron.utils.format.convert_result_to_trec --input runs/qwen3-8b_top1000.txt \
                                                       --output runs/qwen3-8b_top1000.trec
echo "Retrieval Results (Evidence):"
python -m pyserini.eval.trec_eval  -c -m recall.5,100,1000  -m ndcg_cut.10   topics-qrels/qrel_evidence.txt  runs/qwen3-8b_top1000.trec

echo "Retrieval Results (Gold):"
python -m pyserini.eval.trec_eval  -c -m recall.5,100,1000  -m ndcg_cut.10   topics-qrels/qrel_golds.txt  runs/qwen3-8b_top1000.trec
```



## Bright
```bash
for dataset in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_theorems theoremqa_questions; do
    embedding_path=bright_embeddings/${dataset}/reasonrank-reasonir
    mkdir -p ${embedding_path}
    CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
      --output_dir=temp \
      --model_name_or_path Qwen/Qwen3-Embedding-8B \
      --lora \
      --lora_name_or_path /scratch3/zhu042/tevatron/retriever-qwen3-8b-reasonrank-reasonir/checkpoint-400 \
      --dataset_name Tevatron/bright \
      --dataset_config ${dataset} \
      --dataset_split test \
      --encode_output_path ${embedding_path}/queries.pkl \
      --query_max_len 512 \
      --encode_is_query \
      --normalize \
      --pooling eos \
      --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
      --per_device_eval_batch_size 156 \
      --fp16 &
      
      CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode  \
      --output_dir=temp \
      --model_name_or_path Qwen/Qwen3-Embedding-8B \
      --lora \
      --lora_name_or_path /scratch3/zhu042/tevatron/retriever-qwen3-8b-reasonrank-reasonir/checkpoint-400 \
      --dataset_name ArvinZhuang/bright-gpt4_reason \
      --dataset_config ${dataset} \
      --dataset_split test \
      --encode_output_path ${embedding_path}/queries-gpt4.pkl \
      --query_max_len 512 \
      --encode_is_query \
      --normalize \
      --pooling eos \
      --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
      --per_device_eval_batch_size 32 \
      --fp16
    wait
    
    for s in 0 1 2 3;
    do
    CUDA_VISIBLE_DEVICES=${s} python -m tevatron.retriever.driver.encode \
      --model_name_or_path Qwen/Qwen3-Embedding-8B \
      --lora \
      --lora_name_or_path /scratch3/zhu042/tevatron/retriever-qwen3-8b-reasonrank-reasonir/checkpoint-400 \
      --dataset_name Tevatron/bright-corpus \
      --dataset_config ${dataset} \
      --dataset_split train \
      --encode_output_path ${embedding_path}/corpus-${s}.pkl \
      --passage_max_len 512 \
      --normalize \
      --pooling eos \
      --passage_prefix "" \
      --per_device_eval_batch_size 32 \
      --fp16 \
      --dataset_number_of_shards 4 \
      --dataset_shard_index ${s} &
    done
    wait
    
  
done
```


```bash
for dataset in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems; do
    res_path=bright_results/${dataset}/reasonrank
    embedding_path=bright_embeddings/${dataset}/reasonrank
    
    
#    python -m tevatron.retriever.driver.search \
#    --query_reps ${embedding_path}/queries-gpt4.pkl \
#    --passage_reps ${embedding_path}/'corpus-*.pkl' \
#    --depth 100 \
#    --batch_size 64 \
#    --save_text \
#    --save_ranking_to ${res_path}/rank-gpt4.txt
#    
#    
#    python -m tevatron.utils.format.convert_result_to_trec --input ${res_path}/rank-gpt4.txt \
#                                                           --output ${res_path}/rank-gpt4.trec
    
    echo "dataset: ${dataset}"
    python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 bright_qrels/${dataset}.tsv ${res_path}/rank-gpt4.trec

done
```

```bash
for dataset in biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems; do
    
    res_path=bright_results/${dataset}/reasonrank-reasonir
    embedding_path=bright_embeddings/${dataset}/reasonrank-reasonir
    mkdir -p ${res_path}
    python -m tevatron.retriever.driver.search \
    --query_reps ${embedding_path}/queries.pkl \
    --passage_reps ${embedding_path}/'corpus-*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${res_path}/rank.txt
    
    
    python -m tevatron.utils.format.convert_result_to_trec --input ${res_path}/rank.txt \
                                                           --output ${res_path}/rank.trec
    
    echo "dataset: ${dataset}"
    python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 bright_qrels/${dataset}.tsv ${res_path}/rank.trec
done
```