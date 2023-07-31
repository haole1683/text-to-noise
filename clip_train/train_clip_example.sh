# model used to generate clip-min noise 
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export DATASET_NAME="ydshieh/coco_dataset_script"
export DATASET_SHORT_NAME="COCO"
export DATASET_NOISE_TYPE="random"   # random, none, clip_min_noise
export MAX_TRAIN_SAMPLES=1000

# concatenate the dataset name and noise type to the output dir
export CURRENT_TIME=$(date +'%Y-%m-%d_%H-%M-%S')
export OUTPUT_DIR="./my_train_results/$DATASET_SHORT_NAME/$DATASET_NOISE_TYPE/$MAX_TRAIN_SAMPLES/$CURRENT_TIME"

python run_clip_my.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path ../clip-roberta \
    --data_dir /remote-home/songtianwei/research/diffusion_model_my/data \
    --dataset_name $DATASET_NAME \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="64" \
    --learning_rate "5e-5" \
    --warmup_steps="0" \
    --weight_decay=0.1 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --dataset_normalize_flag False \
    --dataset_noise_type $DATASET_NOISE_TYPE \
    # --learning_rate="5e-5" \
    # --preprocessing_num_workers 8 \
    # --overwrite_output_dir \
    # --do_train \
    # --max_eval_samples=100 \
    # --overwrite_cache=True \
# noise type : None (default), random, clip_min_noise
# dataset normalize flag
# larger batch_size !!