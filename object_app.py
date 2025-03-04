import math
import os
import random
import threading
import time
from datetime import datetime, timedelta
from typing import List, Union
import gradio as gr
import spaces
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from model.pipline import VideoMakerAnimateDiffPipeline
from diffusers import MotionAdapter, DDIMScheduler
from diffusers.utils import load_image
from moviepy import VideoFileClip
from diffusers.utils import export_to_video, export_to_gif
from diffusers.image_processor import VaeImageProcessor
# from diffusers.training_utils import free_memory
from PIL import Image
import argparse
import PIL.Image

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--base_model_path", type=str, default='./pretrain_model/Realistic_Vision_V5.1_noVAE', help="Base Model Pretrained Weight path")
    parser.add_argument("--motion_block_path", type=str, default='./pretrain_model/animatediff-motion-adapter-v1-5-3', help="Base Model Pretrained Weight path")
    parser.add_argument("--videomaker_weight_path", type=str, default='./pretrain_model/VideoMaker/human_pytorch_model.bin', help="VideoMaker Pretrained Weight path")
    return parser
    
parser = get_parser()
args = parser.parse_args()

# 0. Pre config
base_model_path = args.base_model_path
device = "cuda" if torch.cuda.is_available() else "cpu"



# 1. Prepare models
adapter = MotionAdapter.from_pretrained(args.motion_block_path)
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
pipe.set_fusion_model(unet_path=args.videomaker_weight_path)
# define and show the input ID images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# 3. Move to device.
pipe.to(device)

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)
default_neg = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch"
@spaces.GPU(enable_queue=True)
def generate(
    prompt,
    image_input,
    num_inference_steps: int = 30,
    guidance_scale: float = 8.0,
    seed: int = 42,
    negative_prompt: str = "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch",
):
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)

    input_id_images=Image.fromarray(image_input).resize((512,512))
    input_id_image_emb = pipe.prepare_reference_image_embeds(input_id_images, None, torch.device(device), 1)

    # 5. Generate Identity-Preserving Video
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    frames = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=guidance_scale,
        reference_image_embeds = input_id_image_emb,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=1,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).frames[0]

    # free_memory()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    gif_path = f"./output/{timestamp}.gif"
    export_to_video(frames,video_path,fps=8)
    export_to_gif(frames,gif_path)

    return (video_path, gif_path, seed)


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path

def save_video(tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path


examples_images = [
    ["examples/object/tiger1.jpeg", "A tiger walking through a field of sunflowers.",128],
    ["examples/object/horse1.jpg", "A horse running across a grassy meadow.",1234],
    ["examples/object/dog1.jpeg", "A dog is walking on a street. High quality and 4k background, real, in high-resolution.",2048],
    ["examples/object/panda1.png", "A panda walking through a bamboo forest.",128],
    ["examples/object/elephant1.jpg", "An elephant walking through the jungle.",128],
]


title = r"""
<h1 align="center">VideoMaker: Zero-shot Customized Video Generation with the Inherent Force of Video Diffusion Models (Customized Object Video Generation)</h1>
"""

description = r"""
❗️❗️❗️[<b>重要</b>] 我们的模型是基于<b>AnimateDiff SD1.5 (1.5B)</b>并在[VideoBooth](https://arxiv.org/pdf/2312.00777)所发布的学术数据集（48,724个视频片段组成，涵盖九类物体：熊、汽车、猫、狗、大象、马、狮子、熊猫、老虎）进行训练从而公平比较，因此<b>只能生成这9类物品的视频</b>。<b>生成效果受限于基础模型能力与数据集限制</b><br>
个性化步骤：<br>
1️⃣ 上传你想要定制的物品的图片。请确保上传图片的背景除了对象外是白色的，我们建议使用[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)或[SAM-2](https://github.com/objectbookresearch/sam2)预处理输入图像，以便在参考图像中仅保留参考物品的主体区域。<br>
2️⃣ 输入文本提示。<br>
3️⃣ 点击<b>🎬 Generate Video</b>按钮开始定制。<br>

❗️❗️❗️[<b>Important</b>] Our model is based on <b>AnimateDiff SD1.5 (1.5B)</b> and trained on the academic dataset released by [VideoBooth](https://arxiv.org/pdf/2312.00777) (comprising 48,724 video clips covering nine categories of objects: bear, car, cat, dog, elephant, horse, lion, panda, tiger) for fair comparison, therefore <b>it can only generate videos of these 9 types of objects</b>.<br>
Personalization steps:<br>
<b>The generated effect is limited by the capabilities of the base model and dataset constraints</b>
1️⃣ Upload the image of the object you want to customize. Please ensure that the background of the uploaded image is white except for the object. We recommend using [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) or [SAM-2](https://github.com/objectbookresearch/sam2) to preprocess the input image, so that only the main area of the reference object is retained in the reference image.<br>
2️⃣ Enter the text prompt.<br>
3️⃣ Click the <b>🎬 Generate Video</b> button to start customization.<br>
"""
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("VideoMaker: object Input", open=True):
                image_input = gr.Image(label="Input Image (should contain clear object, preferably only object region)")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here.", lines=5)
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here. Default is None", value=default_neg, lines=5)
            with gr.Group():
                with gr.Column():
                    num_inference_steps = gr.Slider(1, 100, value=30, step=1, label="Number of Inference Steps, Recommended 30 or 50")
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=1234
                        )
                        cfg_param = gr.Number(
                            label="Guidance Scale (Enter a positive number, default = 8.0)", value=8.0
                        )
            with gr.Accordion("Examples", open=True):
                examples_component_images = gr.Examples(
                    examples_images,
                    inputs=[image_input, prompt,seed_param],
                    cache_examples=False,
                )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            # video_output = gr.Video(label="VideoMaker Generate Video", width=512, height=512)
            gif_output = gr.Image(label="VideoMaker Generate Video", width=512, height=512)
            with gr.Row():
                # download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)



    def run(
        prompt,
        negative_prompt,
        image_input,
        num_inference_steps,
        seed_value,
        cfg_param,
        progress=gr.Progress(track_tqdm=True)
    ):
        video_path, gif_path, seed = generate(
            prompt,
            image_input,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_param,
            seed=seed_value,
        )
        gif_update = gr.update(visible=True, value=gif_path)
        seed_update = gr.update(visible=True, value=seed)

        return gif_update, seed_update, gif_path

    generate_button.click(
        fn=run,
        inputs=[prompt, negative_prompt, image_input, num_inference_steps, seed_param, cfg_param],
        outputs=[download_gif_button, seed_text, gif_output],
    )

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    demo.queue(max_size=15)
    demo.launch(server_name=args.ip, server_port=args.port, share=False)