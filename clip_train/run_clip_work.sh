echo "WTF"

wandb offline

accelerate launch --multi_gpu run_clip_work.py \
    --cache_dir /share/test/songtianwei/huggingface \
    --output_dir /share/ckpt/songtianwei \
    --model_name_or_path /share/test/songtianwei/model_save \
    --data_dir /share/test/songtianwei/data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  \
    --do_eval \
    --report_to="wandb" \
    --learning_rate="5e-5" \
    --warmup_steps="0" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --max_seq_length="77" \
    --max_steps=100000 \
    --num_train_epochs=10 \
    --per_device_train_batch_size="6" \
    --per_device_eval_batch_size="6" \
    --if_clip_train="True" \
    --if_clip_pretrained="True" \
    --if_add_noise="True" \
    --if_generator_train="True" \
    --if_use_8bit_adam="True" \
    --max_train_samples=50000 \
    --max_eval_samples=10000 \
# lr 5e-5

# if if_clip_pretrained set true, unable to converage



python run_clip_work.py \
    --cache_dir /share/test/songtianwei/huggingface \
    --output_dir /share/ckpt/songtianwei \
    --model_name_or_path /share/test/songtianwei/model_save \
    --data_dir /share/test/songtianwei/data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  \
    --do_eval \
    --report_to="wandb" \
    --learning_rate="5e-5" \
    --warmup_steps="0" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --max_seq_length="77" \
    --max_steps=100000 \
    --num_train_epochs=10 \
    --per_device_train_batch_size="6" \
    --per_device_eval_batch_size="6" \
    --if_clip_train="True" \
    --if_clip_pretrained="True" \
    --if_add_noise="True" \
    --if_generator_train="True" \
    --if_use_8bit_adam="True" \
    --max_train_samples=50000 \
    --max_eval_samples=10000 \
