cd /home/fanz2/STARC/evaluation/LongBench/

CUDA_VISIBLE_DEVICES=0,1 python ruler_eval.py \
  --model Meta-Llama-3.1-8B-Instruct \
  --model_template_type llama-3 \
  --datalen 32768 \
  --data_root /home/fanz2/STARC/ruler/data \
  --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multivalue,ruler/niah_multiquery,ruler/vt,ruler/fwe,ruler/cwe,ruler/qa_1,ruler/qa_2" \
  --max_new_tokens 128 \
  --starc --token_budget 1024 --chunk_size 16 \
  --task ruler
