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
default_neg = "semi-realistic, cgi, 3d, render, sketch, multiple people, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
@spaces.GPU(enable_queue=True)
def generate(
    prompt,
    image_input,
    num_inference_steps: int = 30,
    guidance_scale: float = 8.0,
    seed: int = 42,
    negative_prompt: str = "semi-realistic, cgi, 3d, render, sketch, multiple people, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
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
    ["examples/human/yifeiliu.png", "A person playing an acoustic guitar.",42],
    ["examples/human/dilireba.png", "A person is enjoying a cup of coffee in a cozy café.",2048],
    ["examples/human/Hinton.png", "A person watching a laptop, focused on the task at hand.",42],
    ["examples/human/leijun.png", "A person with a bustling urban street scene behind them, capturing the energy of the city.",4096],
    ["examples/human/barack_obama.png", "A person wearing a blue hoodie.",4096],
    ["examples/human/feifei_li.png", "A person in a sleeveless workout top, displaying an active lifestyle.",2048],
]


title = r"""
<h1 align="center">VideoMaker: Zero-shot Customized Video Generation with the Inherent Force of Video Diffusion Models (Customized Human Video Generation)</h1>
"""

description = r"""
❗️❗️❗️[<b>重要</b>] 我们的模型基于<b>AnimateDiff SD1.5 (1.5B)</b>以及学术数据集[CelebV-Text](https://celebv-text.github.io/)(70000条视频，<b>仅含有人在说话的视频</b>)进行训练，因此生成效果在一定程度上受到基础模型和数据的限制。<br>
个性化步骤：<br>
1️⃣ 上传你想要定制的人的图像。为了达到最佳效果，请确保上传图像的背景除了主体的脸部外是白色的（例如白色的身份证照片）。我们建议使用[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)或[SAM-2](https://github.com/facebookresearch/sam2)预处理输入图像，以便在参考图像中仅保留面部区域。如果希望保持生成内容中人物着装接近，无需用白色替换着装部分。<br>
2️⃣ 输入文本提示。<br>
3️⃣ 点击<b>🎬 Generate Video</b>按钮开始定制。<br>

❗️❗️❗️[<b>Important</b>] Our model is based on <b>AnimateDiff SD1.5 (1.5B)</b> and trained on the academic dataset CelebV-Text (70,000 videos, <b>containing only videos of people talking</b>), so the generated effect is to some extent limited by the base model and data constraints.<br>
Personalization steps:<br>
1️⃣ Upload the image of the person you want to customize. Please ensure that the background of the uploaded image is white except for the subject's face (e.g., a white background ID photo). We recommend using [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) or [SAM-2](https://github.com/facebookresearch/sam2) to preprocess the input image, so that only the facial area is retained in the reference image. If you wish to keep the person's attire similar in the generated content, there is no need to replace the attire part with white.<br>
2️⃣ Enter the text prompt.<br>
3️⃣ Click the <b>🎬 Generate Video</b> button to start customization.<br>
"""
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("VideoMaker: Face Input", open=True):
                image_input = gr.Image(label="Input Image (should contain clear face, preferably only face region)")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here.", lines=5)
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here. Default is None", value=default_neg, lines=5)
            with gr.Group():
                with gr.Column():
                    num_inference_steps = gr.Slider(1, 100, value=30, step=1, label="Number of Inference Steps, Recommended 30 or 50")
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=2048
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



demo.queue(max_size=15)
demo.launch(server_name=args.ip, server_port=args.port, share=False)