echo "WTF"

python run_clip_work.py \
    --output_dir ./clip-train \
    --model_name_or_path ../clip-roberta \
    --data_dir /remote-home/songtianwei/research/diffusion_model_my/data \
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
    --per_device_train_batch_size="256" \
    --per_device_eval_batch_size="256" \
    --max_train_samples=50000 \
    --max_eval_samples=10000 \
    --if_clip_train="True" \
    --if_add_noise="False" \
    --if_generator_train="False" \
    --if_use_8bit_adam="False" \
    --resume_from_checkpoint="latest" \

# lr 5e-5