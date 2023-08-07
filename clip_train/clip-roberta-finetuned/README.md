---
base_model: ../clip-roberta
tags:
- generated_from_trainer
datasets:
- ydshieh/coco_dataset_script
model-index:
- name: clip-roberta-finetuned
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# clip-roberta-finetuned

This model is a fine-tuned version of [../clip-roberta](https://huggingface.co/../clip-roberta) on the ydshieh/coco_dataset_script 2017 dataset.
It achieves the following results on the evaluation set:
- eval_loss: 5.6186
- eval_runtime: 1463.7184
- eval_samples_per_second: 6.832
- eval_steps_per_second: 0.027
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 256
- eval_batch_size: 256
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.32.0.dev0
- Pytorch 2.0.1+cu117
- Datasets 2.12.0
- Tokenizers 0.13.3
