cd /home/fanz2/STARC/evaluation/LongBench/

model="Mistral-7B-Instruct-v0.3"
for task in "2wikimqa" "multifieldqa_en" "hotpotqa" "lcc" "musique" "narrativeqa" "passage_retrieval_en" "qasper" "samsum" "triviaqa" "repobench-p" "gov_report" "trec" "multi_news" "passage_count" "qmsum"
do

    for budget in 1024
    do
        CUDA_VISIBLE_DEVICES=0,1 python -u pred.py \
            --model $model --task $task \
            --starc --token_budget $budget --chunk_size 16
        
    done

    # Original model
    CUDA_VISIBLE_DEVICES=0,1 python -u pred.py \
        --model $model --task $task
done

CUDA_VISIBLE_DEVICES=0,1 python -u eval.py --model $model
