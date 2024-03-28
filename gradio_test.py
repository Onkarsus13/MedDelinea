from diffusers import (
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetSceneTextErasingPipeline,
    )
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os



os.environ["CUDA_VISIBLE_DEVICES"]="0"

pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained('/datadrive/control_DiT_CTSeg/')



pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(torch.device('cuda:0'))

pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda:1").manual_seed(1)

image = np.load("/home/ec2-user/tts2/BTCV/data/BTCV/train_npz/case0010_slice142.npz", mmap_mode='r')
im = Image.fromarray(np.uint8(image['image']*255)).convert('RGB')


image = pipe(
    im,
    [im],
    num_inference_steps=20,
    generator=generator,
    controlnet_conditioning_scale=1.0,
    guidance_scale=1.0
).images[0]



image.save('resut.png')



# def inf(image, mask_image):
    
#     image = Image.fromarray(image).resize((512, 512))
#     mask_image = Image.fromarray(mask_image).resize((512, 512))

#     image = pipe(
#         image,
#         mask_image,
#         [mask_image],
#         num_inference_steps=20,
#         generator=generator,
#         controlnet_conditioning_scale=1.0,
#         guidance_scale=1.0
#     ).images[0]

#     return np.array(image)



# if __name__ == "__main__":

#     demo = gr.Interface(
#     inf, 
#     inputs=[gr.Image(), gr.Image()], 
#     outputs="image",
#     title="Scene Text Erasing, IIT-Jodhpur",
#     )
#     demo.launch(share=True)