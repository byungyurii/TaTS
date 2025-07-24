all_models=("iTransformer" "PatchTST" "Crossformer" "DLinear" "FEDformer" "FiLM" "Autoformer" "Informer" "Transformer")

GPU=3

root_path=./data

seeds=(2025)
note="original"

datasets=("Agriculture" "Climate" )
# datasets=("Health")
current_dir=$(pwd)

prior_weight=0.0
prompt_weight=1.0
text_emb=12
# pred_lengths=(12 24 36 48)
pred_lengths=(6 8 10 12)

for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=$(basename ${root_path})

            for pred_len in "${pred_lengths[@]}"
            do
                echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
                CUDA_VISIBLE_DEVICES=${GPU} python -u run2.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path $root_path \
                    --data_path $data_path \
                    --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel} \
                    --model $model_name \
                    --data custom \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --des Exp \
                    --seed $seed \
                    --prior_weight $prior_weight \
                    --prompt_weight $prompt_weight \
                    --save_name single_test.csv \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 100 \
                    --patience 60 \
                    --note $note 
            done
        done
    done
done