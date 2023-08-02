
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
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
import torch.nn as nn
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry,ContextManagers
from transformers.utils.versions import require_version
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

if is_wandb_available():
    import wandb

import argparse

args = argparse.Namespace(
    output_dir='./clip-roberta-finetuned',
    model_name_or_path='../clip-roberta',
    data_dir='/remote-home/songtianwei/research/diffusion_model_my/data',
    dataset_name='ydshieh/coco_dataset_script',
    dataset_config_name='2017',
    image_column='image_path',
    caption_column='caption',
    remove_unused_columns=False,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size='8',
    per_device_eval_batch_size='8',
    learning_rate='5e-5',
    warmup_steps='0',
    weight_decay=0.1,
    overwrite_output_dir=True,
    input_perturbation=0.1,
    dataset_noise_type='clip_min_noise',
    dataset_normalize_flag=False,
    max_train_samples=10000
)

args_list = [
    '--output_dir', './clip-roberta-finetuned',
    '--model_name_or_path', '/remote-home/songtianwei/research/diffusion_model_my/clip-roberta',
    '--data_dir', '/remote-home/songtianwei/research/diffusion_model_my/data',
    '--dataset_name', 'ydshieh/coco_dataset_script',
    '--dataset_config_name', '2017',
    '--image_column', 'image_path',
    '--caption_column', 'caption',
    '--remove_unused_columns', 'False',
    '--do_train',
    '--do_eval',
    '--per_device_train_batch_size', '8',
    '--per_device_eval_batch_size', '8',
    '--learning_rate', '5e-5',
    '--warmup_steps', '0',
    '--weight_decay', '0.1',
    '--overwrite_output_dir',
    '--dataset_noise_type','clip_min_noise',
    '--dataset_normalize_flag','False',
    '--max_train_samples','100000',
    '--report_to','wandb'
]
    
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

dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC,antialias=None),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x
    
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.unet_config = {
            "act_fn": "silu",
            "attention_head_dim": 8,
            "block_out_channels": [
                320,
                640,
                1280,
                1280
            ],
            "center_input_sample": False,
            "cross_attention_dim": 768,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "out_channels": 4,
            "sample_size": 224,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ]
        }
        self.unet = UNet2DConditionModel(**self.unet_config)
        self.vae_config = {
            'in_channels': 3,
            'out_channels': 3,
            'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            'block_out_channels': [128, 256, 512, 512],
            'layers_per_block': 2,
            'act_fn': 'silu',
            'latent_channels': 4,
            'norm_num_groups': 32,
            'sample_size': 512,
            'scaling_factor': 0.18215,
        }
        self.vae = AutoencoderKL(**self.vae_config)
        
    def forward(self, img_pixel_values, encoder_hidden_states):
        latent = self.vae.encode(img_pixel_values).latent_dist.sample()
        timesteps = torch.randint(0, 1000, (1,),device=latent.device)
        timesteps = timesteps.long()  #  6
        unet_pred = self.unet(latent, timesteps, encoder_hidden_states).sample
        vae_decoding = self.vae.decoder(unet_pred)
        return vae_decoding
    
    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()
        self.vae.enable_xformers_memory_efficient_attention()

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        print("1")
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)

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

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="no",
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
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
            
        

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            use_auth_token=True if model_args.use_auth_token else None,
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
        
    revision = None
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    
    use_clip_tokenizer = True
    if use_clip_tokenizer:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
        )
        
    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    clip_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config = clip_model.config
    
    clip_model_config = clip_model.config
    clip_pretrained = False
    
    # if clip_pretrained:
    #     clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
    # else:
    #     clip_model_config = AutoModel.from_pretrained("openai/clip-vit-base-patch32").config
    #     clip_model = AutoModel.from_config(clip_model_config)
        
    
    clip_train = True
    logger.info(f"clip_train: {clip_train}")
    if clip_train:
        clip_model.train()
        clip_model.requires_grad_(True)
    else:
        clip_model.eval()
        clip_model.requires_grad_(False)
        # _freeze_params(clip_model)
        
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(clip_model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(clip_model.text_model)

    if training_args.seed is not None:
        set_seed(training_args.seed)
        
    generator_train=True
    generator = Generator()
    logger.info(f"generator_train: {generator_train}")
    if generator_train:
        generator.train()
        generator.requires_grad_(True)
    else:
        generator.eval()
        generator.requires_grad_(False)
        
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
        )
    
    # text_encoder
    text_encoder.requires_grad_(False)
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
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
        config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)
    
    # data_args.max_seq_length = 77
    
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    
    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples
    
    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open(image_file)
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
        shuffle=False,  # here change to False to check the order of the images
        collate_fn=collate_fn,
        batch_size=training_args.train_batch_size,
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
        batch_size=training_args.eval_batch_size,
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
            
    def normalize_fn(x, mean, std):
        return transforms.Normalize(mean=mean, std=std)(x)
    
    # Initialize the optimizer
    use_8bit_adam = False
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
        "lr": 5e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    optimizer = optimizer_cls(optimizer_grouped_parameters, **adam_kwargs)
   
    # lr_scheduler = 'linear'
    # lr_warmup_steps = 0
    
    # lr_scheduler = get_scheduler(
    #     lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=lr_warmup_steps * training_args.gradient_accumulation_steps,
    #     num_training_steps=training_args.max_steps * training_args.gradient_accumulation_steps,
    # )
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    optimizer = accelerator.prepare(optimizer)
    # lr_scheduler = accelerator.prepare(lr_scheduler)
    generator = accelerator.prepare(generator)
    
    # For model
    clip_model = accelerator.prepare(clip_model)
    if generator_train:
        generator = accelerator.prepare(generator)
        
    train_dataloader = accelerator.prepare(train_dataloader)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None or training_args.max_steps <= 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    training_args.max_steps = (int)(training_args.max_steps)
    training_args.num_train_epochs = (int)(training_args.num_train_epochs)
    
    tracker_project_name = "text2image-fine-tune"
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tracker_config.pop("validation_prompts")
        accelerator.init_trackers(tracker_project_name, tracker_config)
        
    total_batch_size = training_args.train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if args.do_train:
        logger.info(f"  Training num examples = {len(train_dataset)}")
    if args.do_eval:
        logger.info(f"  Evaluation num examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.train_batch_size}")
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
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, training_args.max_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")
    
    accelerator.free_memory()
    clip_model.zero_grad()
                
    for epoch in range(first_epoch, training_args.num_train_epochs):
        if training_args.do_train:
            logging.info("*"*50)
            logging.info("Doing Training")
            logging.info("*"*50)
            if generator_train:
                generator.train()
            else:
                generator.eval()
                
            if clip_train:
                clip_model.train()
            else:
                clip_model.eval()
                
            progress_bar.set_description("Training Steps")
            train_loss = 0.0

            generator_step_M = 1
            clip_step_N = 1
            train_target_list = ["generator"]*generator_step_M + ["clip"]*clip_step_N
            cur_index = 0
            for step, batch in enumerate(train_dataloader):
                clip_model.train()
                # Skip steps until we reach the resumed step
                if training_args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % training_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                # which to train
                train_target = train_target_list[cur_index]
                cur_index = (cur_index + 1) % len(train_target_list)
                if train_target == "generator":
                    pass

                # Convert images to latent space
                img_pixel_values = batch["pixel_values"]  # [6,3,224,224]
                # Get the text tokens for conditioning
                batch_token_ids = batch["input_ids"]
                
                add_noise = False
                generator_train = False
                if add_noise:
                    encoder_hidden_states = text_encoder(batch_token_ids)[0]  # [6,77,768]                
                    noise = generator(img_pixel_values, encoder_hidden_states)
                    
                    # limit the norm of the noise
                    norm_type = 'l2'
                    epsilon = 16
                    if norm_type == 'l2':
                        temp = torch.norm(noise.view(noise.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                        noise = noise * epsilon / temp
                    else:
                        noise = torch.clamp(noise, -epsilon / 255, epsilon / 255)
                    image = img_pixel_values + noise 
                    image = torch.clamp(image, -1, 1)
                else:
                    image = img_pixel_values
                     
                use_normailize = False
                if use_normailize:
                    image = normalize_fn(image)
                    
                batch_data_input = {
                    "input_ids":batch_token_ids,
                    "pixel_values" : image,
                    "attention_mask":batch["attention_mask"],
                    "return_loss": True
                }
                output = clip_model(**batch)
                # logits_per_image = output.logits_per_image   # for training , image_logits is the same as logits text
                # logits_per_text = output.logits_per_text
                
                loss = output.loss
                print("loss-",loss)
                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(loss.repeat(training_args.train_batch_size)).mean()
                # train_loss += avg_loss.item() / training_args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if generator_train:
                        accelerator.clip_grad_norm_(generator.parameters(), training_args.max_grad_norm)
                    elif clip_train:
                        accelerator.clip_grad_norm_(clip_model.parameters(), training_args.max_grad_norm)
                
                # Update optimizer
                optimizer.step()
                # lr_scheduler.step()
                
                clip_model.zero_grad()
                generator.zero_grad()
                # optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                #     global_step += 1
                #     accelerator.log({"train_loss": train_loss}, step=global_step)
                #     train_loss = 0.0

                #     checkpointing_steps = 100
                #     if global_step % checkpointing_steps == 0:
                #         logging.info("Epoch : {} ; Step : {} ; Save checkpoint to {}".format(epoch, global_step, training_args.output_dir))
                #         if accelerator.is_main_process:
                #             save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                #             accelerator.save_state(save_path)
                #             logger.info(f"Saved state to {save_path}")
                
                record = {
                        "epoch": epoch,
                        "step": step,
                        "global_step":global_step,
                        "train_loss": loss.detach().item(),
                        # "lr": lr_scheduler.get_last_lr()[0],
                        "lr": optimizer.param_groups[0]["lr"],
                        }
                wandb.log(record)  
                progress_bar.set_postfix(**record)

                # if global_step >= training_args.max_steps:
                #     break

        # evaluation on the eval dataset
        # training_args.do_eval = False
        # if training_args.do_eval and accelerator.is_main_process:
            
        #     logging.info("*"*50)
        #     logging.info("Doing Evaluation")
        #     logging.info("*"*50)
        #     progress_bar.set_description("Evaluation Steps")
            
        #     generator.eval()
        #     clip_model.eval()
            
        #     eval_losses = []
        #     for step, batch in enumerate(tqdm(eval_dataloader)):
        #         with torch.no_grad():
        #             # Convert images to latent space
        #             img_pixel_values = batch["pixel_values"].to(weight_dtype)  # [6,3,224,224]

        #             # Get the text embedding for conditioning
        #             batch_token_ids = batch["input_ids"]
        #             if generator_train:
        #                 encoder_hidden_states = text_encoder(batch_token_ids)[0]  # [6,77,768]                
        #                 noise = generator.forward(img_pixel_values, encoder_hidden_states)
                        
        #                 # limit the norm of the noise
        #                 norm_type = 'l2'
        #                 epsilon = 16
        #                 if norm_type == 'l2':
        #                     temp = torch.norm(noise.view(noise.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        #                     noise = noise * epsilon / temp
        #                 else:
        #                     noise = torch.clamp(noise, -epsilon / 255, epsilon / 255)
                            
        #                 add_noise = False
        #                 if add_noise:
        #                     image = img_pixel_values + noise
        #                 else:
        #                     image = img_pixel_values + noise * torch.tensor(0.0).to(noise.device)
        #             else:
        #                 image = img_pixel_values 
        #             image = torch.clamp(image, -1, 1)
                    
        #             use_normailize = False
        #             if use_normailize:
        #                 image = normalize_fn(image)
                    
        #             data_input = {
        #                 "input_ids":batch_token_ids,
        #                 "pixel_values" : image
        #             }
        #             output = clip_model(**data_input, return_loss=True)
        #             logits_per_image = output.logits_per_image   # for training , image_logits is the same as logits text
        #             logits_per_text = output.logits_per_text
                    
        #             loss = output.loss
                    
        #             # END MY CODE
        #         eval_losses.append(loss.detach().item())
        #         # logs = {"step" : step,  "lr": lr_scheduler.get_last_lr()[0],"eval_loss": loss.detach().item(),}
        #         # progress_bar.set_postfix(**logs)
        #     eval_mean_loss = np.mean(eval_losses)
        #     eval_record = {
        #                 "epoch": epoch,
        #                 "global_step":global_step,
        #                 "eval_avg_loss": eval_mean_loss,
        #                 }
        #     wandb.log(eval_record)  

    accelerator.end_training()
    

if __name__ == "__main__":
    main()