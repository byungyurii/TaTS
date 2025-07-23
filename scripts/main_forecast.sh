all_models=("iTransformer")

GPU=3

root_path=./data

seeds=(2025)

datasets=("Environment")
# datasets=("Economy")
current_dir=$(pwd)

prior_weight=0.0
text_emb=12
pred_lengths=(48 96 192 336)
# pred_lengths=(6 8 10 12)

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
                CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
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
                    --prompt_weight 1.0 \
                    --save_name result_traffic_itransformer_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 150 \
                    --patience 60 
            done
        done
    done
done