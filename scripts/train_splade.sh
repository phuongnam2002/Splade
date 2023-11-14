timestamp=`date "+%Y%0m%0d_%T"`
model_dir="models"
data_dir="/home/black/saturn/data"
wandb_run_name="namdp_T10"
s="123"
lr="5e-5"
CUDA_VISIBLE_DEVICES=0 python training/train_splade.py \
        --model_dir $model_dir \
        --data_dir $data_dir\
        --token_level word-level \
        --model_type unsim-cse-vietnamese\
        --logging_steps 50 \
        --save_steps 50 \
        --wandb_run_name $wandb_run_name \
        --gpu_id 0 \
        --seed $s \
        --num_train_epochs 10 \
        --train_batch_size 150 \
        --eval_batch_size 312 \
        --max_seq_len_query 64 \
        --max_seq_len_document 256 \
        --learning_rate $lr \
        --tuning_metric recall_bm_history_v400_20 \
        --early_stopping 25 \
        --gradient_checkpointing \
        --do_train \
        --k_tokens 768 \
        --threshold_flops 10 \
        --sparse_loss_weight 1.0 \
        --flops_loss_weight 1.0