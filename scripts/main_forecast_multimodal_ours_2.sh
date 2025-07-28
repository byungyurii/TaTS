all_models=("iTransformer" "Crossformer" "DLinear" "FEDformer" "FiLM" "Autoformer" "Informer" "Transformer")
#all_models=("DLinear" "FiLM")
GPU=3

root_path=./data

seeds=(2025)

datasets=("Economy")

current_dir=$(pwd)

prior_weight=0.5
prompt_weight=1     #unimodal 돌릴 때 0, multimodal 1 
text_emb=12
nce_weight=0.1
#pred_lengths=(12 24 36 48)
pred_lengths=(6 8 10 12)
#pred_lengths=(48 96 192 336)

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
                    --data_name ${dataset} \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --des Exp \
                    --seed $seed \
                    --prior_weight $prior_weight \
                    --save_name multimodal_0728_ours_infoNCEloss_economy_gpt2_pw0.5_nw0.1 \
                    --prompt_weight $prompt_weight \
                    --llm_model GPT2 \
                    --huggingface_token NA \
                    --train_epochs 100 \
                    --patience 20 \
                    --features S \
                    --use_fullmodel 0 \
                    --nce_weight ${nce_weight} 
            done
        done
    done
done
