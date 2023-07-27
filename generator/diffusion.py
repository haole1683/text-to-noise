import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

result_path = "/remote-home/songtianwei/research/diffusion_model_my/sh/text-to-image/train_finetuned/my_train_results"
pretrained_path = result_path

# pipe = StableDiffusionPipeline.from_pretrained(result_path)
# pipe = pipe.to("cuda")    
# pipe.safety_checker = None
# pipe.requires_safety_checker = False

from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet")

# another Scheduler
from diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler.from_pretrained(pretrained_path, subfolder="scheduler")


# set device
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# set model to eval mode
vae.eval()
text_encoder.eval()
unet.eval()

# grad set to false
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

text_prompt = "A photo of a dog"

prompt = [text_prompt]*10
height = 224  # default height of Stable Diffusion
width = 224 # default width of Stable Diffusion
num_inference_steps = 100  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
batch_size = len(prompt)

# generate noise
def generate_noise1(prompts,type="text"):
    
    
    batch_size = len(prompts)
    if type == "text":
        text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")  # [bs:1,seq_len:77]
    else:
        text_input = prompts
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]  # [bs:1,seq_len:77,embedding_dim:768]

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),  # here not use the recommended `config`, goes wrong
        generator=generator,
    )
    latents = latents.to(torch_device)  
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents])
        # print(latent_model_input.shape)  # [bs:2, channel:4, latent_h: 64, latent_w: 64]
    
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # print(latent_model_input.shape)  # same shape of latent_model_input
    
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # print(noise_pred.shape) # same shape of latent_model_input
            
        # perform guidance
        noise_pred_text = noise_pred.chunk(2)
    
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        break

    noise_source = noise_pred
    with torch.no_grad():
        vae_decoding = vae.decode(noise_source).sample
    norm_type = 'l2'
    epsilon = 16
    if norm_type == 'l2':
        temp = torch.norm(vae_decoding.view(vae_decoding.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        vae_decoding = vae_decoding * epsilon / temp
    else:
        vae_decoding = torch.clamp(vae_decoding, -epsilon / 255, epsilon / 255)
    return vae_decoding


def generate_noise(prompts,type="text"):
    batch_size = len(prompts)
    if type == "text":
        text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")  # [bs:1,seq_len:77]
    else:
        text_input = prompts
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]  # [bs:1,seq_len:77,embedding_dim:768]

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),  # here not use the recommended `config`, goes wrong
        generator=generator,
    )
    latents = latents.to(torch_device)  
    # latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents])
        # print(latent_model_input.shape)  # [bs:2, channel:4, latent_h: 64, latent_w: 64]
    
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # print(latent_model_input.shape)  # same shape of latent_model_input
    
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # print(noise_pred.shape) # same shape of latent_model_input
            
        # perform guidance
        noise_pred_text = noise_pred.chunk(2)
    
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        break

    noise_source = noise_pred
    with torch.no_grad():
        vae_decoding = vae.decode(noise_source).sample
    norm_type = 'l2'
    epsilon = 16
    if norm_type == 'l2':
        temp = torch.norm(vae_decoding.view(vae_decoding.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        vae_decoding = vae_decoding * epsilon / temp
    else:
        vae_decoding = torch.clamp(vae_decoding, -epsilon / 255, epsilon / 255)
    return vae_decoding


if __name__ == "__main__":
    # generate noise
    print("generating noise...")
    print(generate_noise(prompt).shape)

