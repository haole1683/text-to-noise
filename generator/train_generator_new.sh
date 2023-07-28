
# pretrained model name
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# finetune dataset name
# export dataset_name="lambdalabs/pokemon-blip-captions"
export dataset_name="ydshieh/coco_dataset_script"
export data_dir="/remote-home/songtianwei/research/diffusion_model_my/data"
export dataset_config_name="2017"

# the path of the pretrained model
export output_dir="my_train_results/$dataset_name/$(date +'%Y-%m-%d')/$('%H-%M-%S')"

accelerate launch --mixed_precision="fp16"  train_generator_new.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --data_dir=$data_dir \
  --use_8bit_adam\
  --resolution=224 --center_crop --random_flip \
  --train_batch_size=6 \
  --eval_batch_size=6 \
  --gradient_checkpointing \
  --max_train_steps=350 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$output_dir \
  --report_to=wandb \
  --dataset_config_name=$dataset_config_name \
  --image_column image_path \
  --caption_column caption \
  --do_eval \
  --do_train \
  --max_train_samples=1000 \
  --max_eval_samples=100
  # --resume_from_checkpoint 