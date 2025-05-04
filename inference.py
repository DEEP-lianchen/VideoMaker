import torch
import numpy as np
import random
import os
from PIL import Image
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
import sys
import torch
import pandas as pd
from model.pipline import VideoMakerAnimateDiffPipeline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, load_image, export_to_video
import cv2
# gloal variable and function
import argparse


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-s", "--seed", type=int, nargs='+',default=[42], help="seed for seed_everything")
    parser.add_argument("-p", "--prompt", type=str, default='A person wearing a Superman outfit.', help="prompt txt path or prompt string")
    parser.add_argument("--image_path", type=str, default='./examples/barack_obama.png', help="reference image path or reference image dir path")
    parser.add_argument("--weight_path", type=str, default='./pretrain_model/VideoMaker/human_pytorch_model.bin', help="VideoMaker Pretrained Weight path")
    parser.add_argument("-o","--output", type=str, default='outputs', help="output dir")
    parser.add_argument("--name", type=str, default='VideoMaker', help="output name")
    parser.add_argument("-n", "--num_steps", type=int, default=30, help="number of steps")
    parser.add_argument("--cfg", type=int, default=8, help="number of steps")
    parser.add_argument("--size",type=int, default=512, help="size of image")
    return parser

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

parser = get_parser()
args = parser.parse_args()
base_model_path = './pretrain_model/Realistic_Vision_V5.1_noVAE'
device = "cuda"
adapter = MotionAdapter.from_pretrained("./pretrain_model/animatediff-motion-adapter-v1-5-3")
scheduler = DDIMScheduler.from_pretrained(
    base_model_path,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = VideoMakerAnimateDiffPipeline.from_pretrained(
    base_model_path,
    motion_adapter=adapter,
    scheduler=scheduler,
).to("cuda")
print("load unet from ",args.weight_path)
pipe.set_fusion_model(unet_path=args.weight_path)
print("over")
# define and show the input ID images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

if isimage(args.image_path):
    image_path_list = [args.image_path]
elif os.path.isdir(args.image_path):
    image_path_list = [os.path.join(args.image_path, x) for x in os.listdir(args.image_path) if isimage(x)]
else:
    raise ValueError("Invalid image path")
input_id_images=[]
for image_path in image_path_list:
    print(image_path)
    input_id_images.append(load_image(image_path).resize((512,512)))

    # print(f"type: {type(input_id_images[-1])}")

for image_path,input_id_image in zip(image_path_list, input_id_images):
    dir_name = os.path.basename(image_path).split('.')[0]
    if args.prompt.endswith('.txt'):
        prompts = load_prompts(args.prompt)
    else:
        prompts = [args.prompt]
    negative_prompt = "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    seed_list = args.seed

    print(f"input_id_image shape: {input_id_image.size}")
    print(f"input_id_image type: {type(input_id_image)}")
    input_id_image_emb = pipe.prepare_reference_image_embeds(input_id_image, None, torch.device("cuda"), 1)
    # exit()
    # torch.save(input_id_image_emb, "/data3/liuyanchen/VideoBooth/input_id_image_emb.pt")
    # print(f"successfully save ref_image_emb to ref_image_emb.pt")
    
    # print("input_id_image_emb shape: ", input_id_image_emb.shape)
    # exit()
    size = (args.size, args.size)
    cnt=-1

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a directory with the timestamp
    output_dir = os.path.join(args.output, args.name, timestamp)

    for prompt in prompts:
        cnt+=1
        for seed in seed_list:
            generator = torch.Generator(device=device).manual_seed(seed)
            frames = pipe(
                prompt=prompt,
                num_frames=16,
                guidance_scale=args.cfg,
                reference_image_embeds = input_id_image_emb,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=1,
                generator=generator,
                num_inference_steps=args.num_steps,
            ).frames[0]
            os.makedirs("{}/{}/{}".format(args.output, dir_name, timestamp), exist_ok=True)
            export_to_gif(frames, "{}/{}/{}/{}_{}_seed_{}.gif".format(args.output, dir_name, timestamp, cnt, prompt.replace(' ','_'),seed))
