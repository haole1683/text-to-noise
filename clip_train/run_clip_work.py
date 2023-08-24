# import ptvsd
# print("waiting for attaching")
# ptvsd.enable_attach(address = ('127.0.0.1', 5678))
# ptvsd.wait_for_attach()
# print("attached")


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import math
from tqdm import tqdm

import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import read_image

import torchvision.transforms as transforms
import accelerate
import datasets
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
from transformers import CLIPTokenizer, CLIPConfig, CLIPModel
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import diffusers

from diffusers.utils import is_wandb_available
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

if is_wandb_available():
    import wandb
    
from utils import Transform, normalize_fn, collate_fn
from generator import generatorDcGan, generatorDDPM


logger = get_logger(__name__, log_level="INFO")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.32.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input testing data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    # Noise type, default is none, other noise is "random" and "clip_min_noise"
    dataset_noise_type: Optional[str] = field(
        default=None,
        metadata={"help": "The type of noise to add to the dataset."},
    )
    
    dataset_normalize_flag: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to normalize the dataset."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


@dataclass
class ExperimentArguments:
    """
    Hyperparameters for the experiment.
    """
    if_clip_pretrained: bool = field(
        default=False, metadata={"help": "Whether to use the pretrained clip model."}
    )
    if_clip_train: bool = field(
        default=True, metadata={"help": "Whether to train the clip model."}
    )
    if_add_noise : bool = field(
        default=False, metadata={"help": "Whether to add noise to the dataset."}
    )
    if_generator_train: bool = field(
        default=False, metadata={"help": "Whether to generate the training dataset."}
    )
    if_normalize: bool = field(
        default=False, metadata={"help": "Whether to normalize the dataset."}
    )
    if_use_clip_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use the clip tokenizer."}
    )
    if_use_8bit_adam: bool = field(
        default=False, metadata={"help": "Whether to use the 8bit adam."}
    )
    if_trainloader_shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the trainloader."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of lr scheduler."},
        # help=(
        #     'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        #     ' "constant", "constant_with_warmup"]'
        # ),
    )


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


def main():
    # print("*"*100)
    # print(torch.cuda.device_count())
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ExperimentArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, experiment_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        print("1")
        model_args, data_args, training_args, experiment_args = parser.parse_args_into_dataclasses()

    if experiment_args.if_generator_train and not experiment_args.if_add_noise:
        raise ValueError("if_generator_train is True, but if_add_noise is False")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip", model_args, data_args)
    
    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    accelerator_project_config = ProjectConfiguration(total_limit=training_args.save_total_limit)

    # reference from https://github.com/huggingface/accelerate/issues/497
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="no",
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_scaler],
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    
    if experiment_args.if_clip_pretrained:
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
            )
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
    else:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    clip_config = CLIPConfig()
    clip_config.vision_config.image_size = 64
    clip_pretrained = experiment_args.if_clip_pretrained
    if clip_pretrained:
        clip_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
        )
    else:
        clip_model = AutoModel.from_config(clip_config)
        
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        logging.info("Freezing vision model")
        _freeze_params(clip_model.vision_model)

    if model_args.freeze_text_model:
        logging.info("Freezing text model")
        _freeze_params(clip_model.text_model)

    if training_args.seed is not None:
        set_seed(training_args.seed)
        
    if experiment_args.if_add_noise:
        if hasattr(clip_model.text_model.embeddings,"word_embeddings"):
            text_embedding_dim = clip_model.text_model.embeddings.word_embeddings.embedding_dim
        elif hasattr(clip_model.text_model.embeddings,"token_embedding"):
            text_embedding_dim = clip_model.text_model.embeddings.token_embedding.embedding_dim
        else:
            pass
        image_channel = clip_config.vision_config.num_channels
        image_shape = clip_config.vision_config.image_size
        generator = generatorDDPM(image_channel, image_shape)
    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        
    

    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        clip_config.vision_config.image_size
    )
    
    # image_transformations = torch.jit.script(image_transformations)
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    
    def transform_images(examples):
        if isinstance(examples[image_column][0],str):
            # For coco dataset, the images are loaded as path
            # images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
            images = [Image.open(image_file).convert("RGB") for image_file in examples[image_column]]
        else:
            # lambdalabs/pokemon-blip-captions
            images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples
    
    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open(image_file).convert("RGB") 
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    if training_args.do_train:
        with accelerator.main_process_first():
            if "train" not in dataset:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = dataset["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            # print(len(train_dataset))
            train_dataset = train_dataset.filter(
                filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
            )
            
            train_dataset = train_dataset.map(
                function=tokenize_captions,
                batched=True,
                remove_columns=[col for col in column_names if col != image_column],
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        
            # Transform images on the fly as doing it on the whole dataset takes too much time.
            train_dataset.set_transform(transform_images)

    # train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=experiment_args.if_trainloader_shuffle,
        collate_fn=collate_fn,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        drop_last=True,
    )
    
        
    if training_args.do_eval:
        with accelerator.main_process_first():
            if "validation" not in dataset:
                raise ValueError("--do_eval requires a train validation")
            eval_dataset = dataset["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
        
            eval_dataset = eval_dataset.filter(
                filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
            )
            eval_dataset = eval_dataset.map(
                function=tokenize_captions,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[col for col in column_names if col != image_column],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        
            # Transform images on the fly as doing it on the whole dataset takes too much time.
            eval_dataset.set_transform(transform_images)
    
    # evaluation dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=training_args.per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        drop_last=True,
    )


    if training_args.do_predict:
        with accelerator.main_process_first():
            if "test" not in dataset:
                raise ValueError("--do_predict requires a test dataset")
            test_dataset = dataset["test"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
                test_dataset = test_dataset.select(range(max_eval_samples))
        
            test_dataset = test_dataset.filter(
                filter_corrupt_images, batched=True, num_proc=data_args.preprocessing_num_workers
            )
            test_dataset = test_dataset.map(
                function=tokenize_captions,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[col for col in column_names if col != image_column],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
        
            # Transform images on the fly as doing it on the whole dataset takes too much time.
            test_dataset.set_transform(transform_images)

    # Initialize the optimizer
    use_8bit_adam = experiment_args.if_use_8bit_adam
    
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    decay_parameters = get_parameter_names(clip_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in clip_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in clip_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
    adam_kwargs = {
        "lr": training_args.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    generator_lr_scale = 1
    optimizer = optimizer_cls(optimizer_grouped_parameters, **adam_kwargs)
    if experiment_args.if_add_noise and experiment_args.if_generator_train:
        adam_kwargs["lr"] = training_args.learning_rate * generator_lr_scale
        generator_parameters = generator.parameters()
        optimizer_generator = optimizer_cls(generator_parameters, **adam_kwargs)
    
    lr_scheduler = get_scheduler(
        experiment_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps= 500,
        num_training_steps= training_args.max_steps * training_args.gradient_accumulation_steps,
    )
    lr_scheduler_generator = get_scheduler(
        experiment_args.lr_scheduler,
        optimizer=optimizer_generator,
        num_warmup_steps= 500,
        num_training_steps= training_args.max_steps * training_args.gradient_accumulation_steps,
    )
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    
    
    # Accelerator
    # For model
    clip_model = accelerator.prepare(clip_model)
    if experiment_args.if_add_noise:
        generator = accelerator.prepare(generator)
    # For optimizer and scheduler
    optimizer = accelerator.prepare(optimizer)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if experiment_args.if_add_noise and experiment_args.if_generator_train:
        optimizer_generator = accelerator.prepare(optimizer_generator)
        lr_scheduler_generator = accelerator.prepare(lr_scheduler_generator)    
    
    train_dataloader = accelerator.prepare(train_dataloader)
    eval_dataloader = accelerator.prepare(eval_dataloader)

    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None or training_args.max_steps <= 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    training_args.max_steps = (int)(training_args.max_steps)
    training_args.num_train_epochs = (int)(training_args.num_train_epochs)
    
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_project_name = "poison_clip"
        accelerator.init_trackers(tracker_project_name)
        
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if training_args.do_train:
        logger.info(f"  Training num examples = {len(train_dataset)}")
    if training_args.do_eval:
        logger.info(f"  Evaluation num examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint != "latest":
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(training_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{training_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            training_args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(training_args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * training_args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * training_args.gradient_accumulation_steps)
            
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, training_args.max_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")
    
    accelerator.free_memory()
    
    
    if_add_noise = experiment_args.if_add_noise
    if_generator_train = experiment_args.if_generator_train
    if_clip_train = experiment_args.if_clip_train
    
    if if_add_noise:
        if if_generator_train:
            generator.train()
            generator.requires_grad_(True)
            generator.zero_grad()
        else:
            generator.eval()
            generator.requires_grad_(False)
    else:
        pass
        
    if if_clip_train:
        clip_model.train()
        clip_model.requires_grad_(True)
    else:
        clip_model.eval()
        clip_model.requires_grad_(False)
    clip_model.zero_grad()
        
    if_normalize = experiment_args.if_normalize
    
    logger.info("Accelerator.device {}".format(accelerator.device))
    logger.info("GPU_NUM: {}".format(torch.cuda.device_count()))
    logger.info("clip_train: {}, generator_train: {}".format(if_clip_train, if_generator_train))
    logger.info("add_noise: {}, use_normailize:{}".format(if_add_noise,if_normalize))
    wandb.init()
    
    for epoch in range(first_epoch, training_args.num_train_epochs):
        print("THE DEVICE",accelerator.device)
        if training_args.do_train:
            if accelerator.is_main_process:
                logging.info("*"*50)
                logging.info("Doing Training")
                logging.info("*"*50)
                progress_bar.set_description("Training Steps")
            
            
            train_loss = 0.0
            generator_step_M = 1
            clip_step_N = 1
            train_target_list = ["generator"]*generator_step_M + ["clip"]*clip_step_N
            cur_index = 0
            
            clip_model.train()
            if if_generator_train:
                generator.train()
            
            # wait for everyone to be ready before starting to train.
            # accelerator.wait_for_everyone()
            
            for step, batch in enumerate(train_dataloader):
                batch_pixel_values = batch["pixel_values"]  # [6,3,224,224]
                batch_input_ids = batch["input_ids"]
                batch_attention_mask = batch["attention_mask"]
                
                train_target = train_target_list[cur_index]
                if train_target == "generator":
                    generator.requires_grad_(True)
                else:
                    generator.requires_grad_(False)
                
                if if_add_noise:
                    # if this is not distrubution training
                    if accelerator.use_distributed:
                        text_encoder = clip_model.module.text_model
                    else:
                        text_encoder = clip_model.text_model
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch_input_ids,batch_attention_mask)[0]               

                    generator_output = generator(batch_pixel_values, encoder_hidden_states)
                    image_ = generator_output
                    noise = image_ - batch_pixel_values
                    # noise = generator_output
                    
                    # limit the norm of the noise
                    norm_type = 'l2'
                    epsilon = 16
                    if norm_type == 'l2':
                        temp = torch.norm(noise.view(noise.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                        noise = noise * epsilon / temp
                    else:
                        noise = torch.clamp(noise, -epsilon / 255, epsilon / 255)
                    image = batch_pixel_values + noise 
                    image = torch.clamp(image, -1, 1)
                else:
                    image = batch_pixel_values
                image = batch_pixel_values  
                
                if if_normalize:
                    image = normalize_fn(image, mean=image_processor.image_mean, std=image_processor.image_std)
                    
                batch_data_input = {
                    "input_ids":batch_input_ids,
                    "pixel_values" : image,
                    "attention_mask":batch_attention_mask,
                    "return_loss": True
                }
        
                output = clip_model(**batch_data_input)
                logits_per_image = output.logits_per_image   # for training , image_logits is the same as logits text
                logits_per_text = output.logits_per_text
                loss = output.loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(training_args.per_device_train_batch_size)).mean()
                train_loss += avg_loss.item()/training_args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if if_generator_train:
                        accelerator.clip_grad_norm_(generator.parameters(), training_args.max_grad_norm)
                    if if_clip_train:
                        accelerator.clip_grad_norm_(clip_model.parameters(), training_args.max_grad_norm)
                
                # Update optimizer and scheduler
                if if_add_noise and if_generator_train:
                    if train_target == "generator":
                        optimizer_generator.step()
                        lr_scheduler_generator.step()
                    elif train_target == "clip":
                        optimizer.step()
                        lr_scheduler.step()
                else:
                    optimizer.step()
                    lr_scheduler.step()
                
                # update train target 
                cur_index = (cur_index + 1) % len(train_target_list)
                    
                
                # zero the grad of model and optimizer
                clip_model.zero_grad()
                optimizer.zero_grad()
                if if_add_noise and if_generator_train:
                    generator.zero_grad()
                    optimizer_generator.zero_grad()
                
                # update progress bar and save state
                if accelerator.is_main_process:
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0

                        checkpointing_steps = 5000
                        if global_step % checkpointing_steps == 0 or global_step == 10:
                            logging.info("Epoch : {} ; Step : {} ; Save checkpoint to {}".format(epoch, global_step, training_args.output_dir))
                            save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            accelerator.save_model(clip_model, os.path.join(save_path, "clip_model"))
                            if if_add_noise and if_generator_train:
                                accelerator.save_model(generator, os.path.join(save_path, "generator"))
                            logger.info(f"Saved state to {save_path}")
                        
                    record = {
                            "epoch": epoch,
                            "step": step,
                            "global_step":global_step,
                            "train_loss": loss.detach().item(),
                            "avg_train_loss": avg_loss.detach().item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            }
                    wandb.log(record)  
                    progress_bar.set_postfix(**record)
                    
                    if avg_loss.detach().item() < 0.9:
                        # end the training if the loss is too small        
                        accelerator.end_training()
                    if global_step >= training_args.max_steps:
                        break
        
        # evaluation on the eval dataset
        if training_args.do_eval and accelerator.is_main_process:
            logging.info("*"*50)
            logging.info("Doing Evaluation")
            logging.info("*"*50)
            progress_bar.set_description("Evaluation Steps")
            
            if if_add_noise:
                generator.eval()    
            clip_model.eval()
            
            eval_losses = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    batch_pixel_values = batch["pixel_values"]  # [6,3,224,224]
                    batch_input_ids = batch["input_ids"]
                    batch_attention_mask = batch["attention_mask"]

                    image = batch_pixel_values

                    if if_normalize:
                        image = normalize_fn(image, mean=image_processor.image_mean, std=image_processor.image_std)
                        
                    batch_data_input = {
                        "input_ids":batch_input_ids,
                        "pixel_values" : image,
                        "attention_mask":batch_attention_mask,
                        "return_loss": True
                    }
                    output = clip_model(**batch_data_input)
                    logits_per_image = output.logits_per_image   # for training , image_logits is the same as logits text
                    logits_per_text = output.logits_per_text
                    
                    loss = output.loss
                    logs = {"step" : step,  "eval_loss": loss.detach().item(),}
                    progress_bar.set_postfix(**logs)
                    eval_losses.append(loss.detach().item())
                
            eval_mean_loss = np.mean(eval_losses)
            eval_record = {
                        "epoch": epoch,
                        "global_step":global_step,
                        "eval_avg_loss": eval_mean_loss,
                        }
            wandb.log(eval_record)  

    accelerator.end_training()
    

if __name__ == "__main__":
    main()