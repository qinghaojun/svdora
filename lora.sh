export CUDA_VISIBLE_DEVICES=0,1
BASE_MODEL="/home/mw/work/yqh/svdora/model/gemma"
OUTPUT="/home/mw/work/yqh/svdora/output/lora-gemma-r128" 
DATA_PATH="/home/mw/work/yqh/svdora/traindata"
export PATH="/home/mw/work/yqh/svdora-env/bin:$PATH"
 /home/mw/work/yqh/env/svdora/bin/python svdora.py \
     --model_name_or_path $BASE_MODEL \
     --output_dir $OUTPUT \
     --lora_r 128 \
     --data_path $DATA_PATH  \
     --dataset_split "train[:100000]"\
     --dataset_field query response \
     --num_train_epochs 1 \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 128 \
     --save_strategy "steps" \
     --save_steps 100 \
     --save_total_limit 1 \
     --learning_rate 2e-5 \
     --weight_decay 0. \
     --warmup_ratio 0.03 \
     --lr_scheduler_type "cosine" \
     --logging_steps 1 \
     --bf16 True \
     --tf32 True \
     --report_to none

 python merge_adapter_to_base_model.py --base_mode $BASE_MODEL --adapter $OUTPUT/ft/ --output_path $OUTPUT
 python inference/gsm8k_inference.py --model $OUTPUT
 python inference/MATH_inference.py --model $OUTPUT