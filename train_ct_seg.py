import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
import PIL
import requests
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, ControlNetModel, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers.image_processor import VaeImageProcessor

import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import glob
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="2"



if __name__ == "__main__":

    accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision='fp16',
            log_with='wandb',
            # logging_dir='logs',
        )

    pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2'
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision='fp16',
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder='tokenizer'
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision="fp16",
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision='fp16',
    )

    controlnet = ControlNetModel.from_unet(unet)

    in_channels = 8
    out_channels = controlnet.conv_in.out_channels
    controlnet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, controlnet.conv_in.kernel_size, controlnet.conv_in.stride, controlnet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(controlnet.conv_in.weight)
        controlnet.conv_in = new_conv_in

    

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    optimizer = torch.optim.AdamW(
            controlnet.parameters(),
            lr=0.0001,
    )


    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    class GenMask2CT(Dataset):
        def __init__(self, path):
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            self.input_image_preprocessor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            self.crontrol_image_preprocessor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            self.class_dict_BTCV = {
                    0:(0, 0, 0),
                    1:(255, 60, 0),
                    2:(255, 60, 232),
                    3:(134, 79, 117),
                    4:(125, 0, 190),
                    5:(117, 200, 191),
                    6:(230, 91, 101),
                    7:(255, 0, 155),
                    8:(75, 205, 155),
                    9:(100, 37, 200)
            }

            self.class_dict = {
                    0:"background",
                    1:"aorta",
                    2:"kidney_left",
                    3:"liver",
                    4:"postcava",
                    5:"stomach",
                    6:"gall_bladder",
                    7:"kidney_right",
                    8:"pancreas",
                    9:"spleen"
            }

            self.path = path

            self.images = os.listdir(self.path+"/images")
            self.labels = os.listdir(self.path+"/labels")

            print(len(self.images), len(self.labels))


        def __len__(self,):
            return len(self.images)
        
        def __getitem__(self, idx):

            name = self.images[idx]
            image = Image.open(self.path+f"/labels/{name}").convert("RGB").resize((256, 256))
            label = Image.open(self.path+f"/images/{name}").convert("RGB").resize((256, 256))

            l = cv2.imread(self.path+f"/labels/{name}")
            l = cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
            l = self.rgb_to_onehot(l, self.class_dict_BTCV).argmax(-1)
            unique_ids = np.unique(l)

            image = self.input_image_preprocessor.preprocess(image)

            label = self.input_image_preprocessor.preprocess(label)

            condn = self.crontrol_image_preprocessor.preprocess(image)

            text_ids = tokenizer(
                'CT image containg '+" ".join([self.class_dict[i] for i in unique_ids]),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )


            return {
                'o_pixel_values':image.squeeze(0), # corrosponding-mask
                'g_pixel_values':label.squeeze(0), # ct-image
                'control_image':condn.squeeze(0), # if requried extra controable condition image
                'input_ids':text_ids['input_ids'].squeeze(0) # text ids will be fixed
            }

        def onehot_to_rgb(self, onehot, color_dict):
            onehot = np.int64(onehot)
            output = np.zeros( onehot.shape[:2]+(3,) )
            for k in color_dict.keys():
                output[onehot==k] = color_dict[k]
            return np.uint8(output)

        def rgb_to_onehot(self, rgb_arr, color_dict):
            num_classes = len(color_dict)
            shape = rgb_arr.shape[:2]+(num_classes,)
            arr = np.zeros( shape, dtype=np.int8 )
            for i, cls in enumerate(color_dict):
                arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
            return arr


    train_ds = GenMask2CT("/data2/onkar/altasSlices")

    train_dataloader = DataLoader(train_ds, batch_size=8, num_workers=2, shuffle=True)

    epochs=150
    num_step_loader = len(train_dataloader)
    lr_scheduler = get_scheduler(
            'cosine',
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps = epochs*num_step_loader
        )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    print(accelerator.device)
    max_train_step = 10000
    epochs = 150


    for epoch in range(epochs):
        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                
                _,_,h,w = batch["g_pixel_values"].shape
                latents = vae.encode(batch["g_pixel_values"].to(weight_dtype)).latent_dist.sample()
                o_latent = vae.encode(batch['o_pixel_values'].to(weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
                o_latent = o_latent * 0.18215

                condn = batch['control_image'].to(weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                control_noisy_latents = torch.cat([noisy_latents, o_latent], axis=1)


                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                        control_noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=condn,
                        return_dict=False,
                    )
                
                model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), max_norm=4.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            # if accelerator.sync_gradients:
            #     progress_bar.update(1)
            #     global_step += 1
            #     accelerator.log({"train_loss": train_loss}, step=global_step)
            #     train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            print(logs)

    # Create the pipeline using the trained modules and save it.
        if epoch % 10 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                controlnet = accelerator.unwrap_model(controlnet)
                p = StableDiffusionControlNetInpaintPipeline(
                        vae= vae,
                        text_encoder= text_encoder,
                        tokenizer= tokenizer,
                        unet= unet,
                        scheduler=noise_scheduler,
                        controlnet = controlnet,
                        safety_checker=None,
                        feature_extractor=None
                )
                p.save_pretrained('/data2/onkar/altasSlices/Mask2CT/')

    accelerator.end_training()
