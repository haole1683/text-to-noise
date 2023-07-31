# model used to generate clip-min noise 
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# Define the options for DATASET_NOISE_TYPE and MAX_TRAIN_SAMPLES
# NOISE_TYPES_CHOICEs=("random" "none" "clip_min_noise")
NOISE_TYPES_CHOICEs=("clip_min_noise")
# MAX_TRAIN_SAMPLE_CHOICEs=(10000 50000 100000)
MAX_TRAIN_SAMPLE_CHOICEs=(50000)
DATASET_NAMES_CHOICEs=("ydshieh/coco_dataset_script")
DATASET_SHORT_NAME_CHOICEs=("COCO")
POISON_RATIO_CHOICEs=(1.0)



# Loop through the dataset names
for DATASET_NAME in "${DATASET_NAMES_CHOICEs[@]}"
    # Loop through the noise types
    do
    for MAX_TRAIN_SAMPLES in "${MAX_TRAIN_SAMPLE_CHOICEs[@]}"
        # Loop through the noise types
        do
            for DATASET_NOISE_TYPE in "${NOISE_TYPES_CHOICEs[@]}"
            # Loop through the train samples
            do
                # concatenate the dataset name and noise type to the output dir
                export CURRENT_TIME=$(date +'%Y-%m-%d_%H-%M-%S')
                export OUTPUT_DIR="./my_train_results/$DATASET_SHORT_NAME/$DATASET_NOISE_TYPE/$MAX_TRAIN_SAMPLES/$CURRENT_TIME"
                export DATASET_SHORT_NAME="COCO"

                export Learning_Rate="5e-5"  # 5e-5 

                echo "Launch task config as follows:"
                echo "DATASET_NAME: $DATASET_NAME"
                echo "DATASET_NOISE_TYPE: $DATASET_NOISE_TYPE"
                echo "MAX_TRAIN_SAMPLES: $MAX_TRAIN_SAMPLES"
                echo "OUTPUT_DIR: $OUTPUT_DIR"
                echo "Learning_Rate: $Learning_Rate"

                # Start your task using the current noise type and train samples count
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
                    --learning_rate $Learning_Rate \
                    --warmup_steps="0" \
                    --weight_decay=0.1 \
                    --max_train_samples $MAX_TRAIN_SAMPLES \
                    --dataset_normalize_flag=False \
                    --dataset_noise_type $DATASET_NOISE_TYPE \
                    # --preprocessing_num_workers 2 \
                    # --overwrite_output_dir \
                    # --max_eval_samples=100 \
                    # --overwrite_cache=True \
        done
    done
done




