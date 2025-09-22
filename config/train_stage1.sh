llamafactory-cli train \
--stage sft \
--do_train True \
--model_name_or_path Models/Qwen2.5-3B-Instruct \
--preprocessing_num_workers 16 \
--finetuning_type full \
# Template to use: qwen, llama3, or default(starcoder)
--template qwen \
--flash_attn auto \
--dataset_dir json_data \
# Select the training dataset
--dataset monochrome_1_easy,monochrome_2_easy,monochrome_1,monochrome_2 \
--cutoff_len 8000 \
--learning_rate 4e-05 \
--num_train_epochs 1 \
--max_samples 1000000 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 5 \
--save_steps 3000 \
--warmup_steps 0 \
--packing False \
--report_to none \
# Set the model save path
--output_dir saves/Qwen2.5-3B-Instruct/full/monochrome \
--bf16 True \
--plot_loss True \
--trust_remote_code True \
--ddp_timeout 180000000 \
--include_num_input_tokens_seen True \
--optim adamw_torch \
--val_size 0.05 \
--eval_strategy steps \
--eval_steps 3000 \
--per_device_eval_batch_size 2 \
--deepspeed cache/ds_z2_config.json