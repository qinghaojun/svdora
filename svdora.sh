export CUDA_VISIBLE_DEVICES=0,1
RESIDUAL_MODEL="/home/mw/work/yqh/PiSSA/model/gemma"
OUTPUT="/home/mw/work/yqh/PiSSA/output/lora-gemma-r128" #忘记改成pissa-xx了
DATA_PATH="/home/mw/work/yqh/PiSSA/traindata"

export PATH="/home/mw/work/yqh/env/pissa/bin:$PATH"
 /home/mw/work/yqh/env/pissa/bin/python svdora.py \
     --model_name_or_path $RESIDUAL_MODEL \
     --output_dir $OUTPUT \
     --adapter_name_or_path svdora_init \
     --init_lora_weights svdora  \
     --data_path $DATA_PATH \
     --dataset_split "train[:100000]"\
     --dataset_field query response \
     --num_train_epochs 1 \
     --per_device_train_batch_size 1 \
     --gradient_accumulation_steps 128 \
     --save_strategy "steps" \
     --save_steps 200 \
     --save_total_limit 1 \
     --learning_rate 2e-5 \
     --lora_r 128 \
     --weight_decay 0. \
     --warmup_ratio 0.03 \
     --lr_scheduler_type "cosine" \
     --logging_steps 1 \
     --bf16 True \
     --tf32 True \
     --report_to none \
#     --use_dora True \
   # --_n_gpu 2
CUDA_VISIBLE_DEVICES=0,1 /home/mw/work/yqh/env/pissa/bin/python merge_adapter_to_base_model.py --base_mode $RESIDUAL_MODEL --adapter $OUTPUT/ft/ --output_path $OUTPUT
CUDA_VISIBLE_DEVICES=0,1 /home/mw/work/yqh/pissa-env/bin/python inference/gsm8k_inference.py --model $OUTPUT --gpu_memory_utilization 0.9
CUDA_VISIBLE_DEVICES=0,1 /home/mw/work/yqh/pissa-env/bin/python inference/MATH_inference.py --model $OUTPUT --gpu_memory_utilization 0.9
  # --adapter_name_or_path $ADAPTER_PATH \
      #--data_path $DATA_PATH \
      #CUDA_VISIBLE_DEVICES=1 /home/mw/work/yqh/env/pissa/bin/python inference/MATH_inference.py --model $OUTPUT