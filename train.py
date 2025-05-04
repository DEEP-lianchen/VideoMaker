import os
import random
import argparse
from pathlib import Path
import json
import itertools                                                                                                                                                                                                                 
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import MotionAdapter, DDIMScheduler, AutoencoderKL, DDPMScheduler, UNet2DConditionModel, UNetMotionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from datasets import get_dataset
from model.attention_processor import PersonalizedInjectionAttnProcessor2_0
from model.utills import prepare_image

from safetensors.torch import save_file

from safetensors.torch import save_file

def check_trainable_params(model):
    """
    检查模型中的所有参数是否标记为需要梯度，并返回需要训练的参数数量。

    Args:
        model: 需要检查的模型，通常是 unet。

    Returns:
        int: 需要训练的参数数量。
    """
    trainable_param_count = 0
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        if param.requires_grad:
            trainable_param_count += 1
    
    return trainable_param_count


def set_seed(seed):
    """
    设置随机种子以确保可重复性。
    
    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    设置每个工作线程的随机种子。
    
    Args:
        worker_id (int): 工作线程 ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_new_attn_processors(unet, num_frames=16):
    """
    根据 UNet 的配置，构建新的 attention processor 字典。
    
    Args:
        unet: 需要处理的 UNet 模型
        num_frames (int): 视频帧数，用于自定义的 Attention Processor

    Returns:
        dict: 新的 attn_processors 映射表
    """
    cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            # 避免意外：如果名字不属于上述三类，默认跳过
            continue

        if "motion_modules" in name or "encoder_hid_proj" in name:
            attn_procs[name] = unet.attn_processors[name]
        elif cross_attention_dim is None:
            attn_procs[name] = PersonalizedInjectionAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_frames=num_frames,
            )
        else:
            attn_procs[name] = unet.attn_processors[name]

    # # 转成 float32，防止 AMP 报错
    # for module in attn_procs.values():
    #     if isinstance(module, torch.nn.Module):
    #         module.to(dtype=torch.float32)

    # for name, attn_proc in attn_procs.items():
    #     print(f"Attn processor for {name}: {attn_proc.__class__.__name__}")

    # exit()


    return attn_procs

def compute_vae_encodings(image, vae):
        """
        Args:
            images (torch.Tensor): image to be encoded
            vae (torch.nn.Module): vae model

        Returns:
            torch.Tensor: latent encoding of the image
        """
        pixel_values = image.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
        with torch.no_grad():
            model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        return model_input

def prepare_reference_image_embeds(reference_image, vae):
        # reference_image = prepare_image(reference_image)
        # print(f"reference_image.shape: {reference_image.shape}")
        ref_latents = compute_vae_encodings(reference_image, vae)
        # print(f"ref_latents.shape: {ref_latents.shape}")
        ref_latents = ref_latents.unsqueeze(2)
        ref_latents = torch.cat([ref_latents])
        return ref_latents


def parse_args():
    parser = argparse.ArgumentParser(description="VideoMaker Training")
    parser.add_argument(
        "--pretrained_base_model",
        type=str,
        default="./pretrain_model/Realistic_Vision_V5.1_noVAE",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--pretrained_motion_adapter",
        type=str,
        default="./pretrain_model/animatediff-motion-adapter-v1-5-3",
        help="Path to pretrained motion adapter or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/data3/lyc_datas/VideoBoothDataset/",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Path to logging directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="videomaker_output/tiger2",
        help="Path to output directory",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames in the video",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=8,
        help="Interval between frames in the video",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting method for logging",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size of the images",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2000000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--ref_noisy_ratio",
        type=float,
        default=0.01,
        help="Ratio of noise to add to the reference image",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between model saves",
    )
    parser.add_argument(
        "--ref_loss_beta",
        type=float,
        default=0.1,
        help="Weight for the reference loss",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank
    else:
        args.local_rank = 0

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    set_seed(42)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, 
                                                       logging_dir=logging_dir)
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # load scheduler, model, and tokenizer
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_base_model,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    motion_adapter = MotionAdapter.from_pretrained(
        args.pretrained_motion_adapter,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_base_model, subfolder="unet")
    if isinstance(unet, UNet2DConditionModel):
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)


    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    motion_adapter.requires_grad_(False)

    attn_procs = build_new_attn_processors(unet, num_frames=args.num_frames)
    unet.set_attn_processor(attn_procs)

    for name, param in unet.named_parameters():
        if 'attentions' in name and 'transformer_blocks' in name and 'attn1' in name:
            param.requires_grad_(True)
        # if 'motion_modules' in name:
        #     param.requires_grad_(True)



    # for name, param in unet.named_parameters():
    #     # 写入json文件
    #     json_name = "./unet_params.json"
    #     with open(json_name, 'a') as f:
    #         f.write(f"{name}: {param.requires_grad}\n")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"Using weight dtype: {weight_dtype}")

    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    motion_adapter.to(accelerator.device, dtype=weight_dtype)

    # 收集所有 requires_grad=True 的参数
    params_to_opt = [param for name, param in unet.named_parameters() if param.requires_grad]

    for name, param in unet.named_parameters():
        if param.requires_grad:
            json_name = "./unet_params.json"
            with open(json_name, 'a') as f:
                f.write(f"{name}: {param.requires_grad}\n")

    # 用 AdamW 优化这些参数
    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = get_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_date_time = time.perf_counter() - begin
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    b, f, c, h, w = batch["video"].shape
                    batch["video"] = batch["video"].view(b * f, c, h, w)
                    latents = vae.encode(batch["video"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=args.num_frames)

                noise = torch.randn_like(latents, dtype=weight_dtype)
                noise = noise.to(dtype=weight_dtype)
                bsz = latents.shape[0]

                # print(f"latents.shape: {latents.shape}")
                # print(f"bsz: {bsz}")

                # print(f"batch['video_id']:{batch['video_id']}")


                with torch.no_grad():
                    b, f, c, h, w = batch["masked_first_frame"].shape
                    batch["masked_first_frame"] = batch["masked_first_frame"].view(b * f, c, h, w)
                    ref_image_emb = prepare_reference_image_embeds(batch["masked_first_frame"], vae)

                # torch.save(ref_image_emb, "/data3/liuyanchen/VideoBooth/ref_image_emb.pt")
                # print(f"successfully save ref_image_emb to ref_image_emb.pt")

                
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                # print(f"timesteps: {timesteps}")
                timesteps = timesteps.long()
                # print(f"ref_image_emb.shape: {ref_image_emb.shape}")

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                reference_image_noisy = True
                if reference_image_noisy:
                    noise_ref = torch.mean(noisy_latents, dim=2, keepdim=True)
                    noise_ref = noise_ref.to(dtype=weight_dtype)
                    ref_timesteps = torch.tensor(int(timesteps * args.ref_noisy_ratio),dtype=torch.int)
                    ref_latents = noise_scheduler.add_noise(ref_image_emb, noise_ref, ref_timesteps)

                ### drop 
                rand_num = random.random()
                if rand_num < 0.05:
                    # print("ok")
                    ref_latents = torch.zeros_like(ref_latents)
                    batch["video_prompt"] = [""]
                elif rand_num < 0.3:
                    # print("ok2")
                    ref_latents = torch.zeros_like(ref_latents)
                    batch["video_prompt"] = batch["video_prompt"]
                elif rand_num < 1.0:
                    # print("ok3")
                    ref_latents = ref_latents
                    batch["video_prompt"] = batch["video_prompt"]

                # exit()

                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch["video_prompt"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(prompt_ids).last_hidden_state

                # print(f"ref_latents.shape: {ref_latents.shape}")
                # print(f"encoder_hidden_states.shape: {encoder_hidden_states.shape}")
                # print(f"noisy_latents.shape: {noisy_latents.shape}")

                latent_model_input = torch.cat([noisy_latents, ref_latents], dim=2)

                # print(f"latent_model_input.dtype: {latent_model_input.dtype}")
                # print(f"encoder_hidden_states.dtype: {encoder_hidden_states.dtype}")
                noise_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # print(f"noise_pred.dtype: {noise_pred.dtype}")
                # assert noise_pred.dtype == weight_dtype


                noise_pred_ori = noise_pred[:,:,:args.num_frames,:,:]
                noise_pred_ref = noise_pred[:,:,args.num_frames:,:,:]

                # assert noise.dtype == weight_dtype
                # assert noise_ref.dtype == weight_dtype
                # assert noise_pred_ori.dtype == weight_dtype
                # assert noise_pred_ref.dtype == weight_dtype


                loss_ori = F.mse_loss(noise_pred_ori.float(), noise.float(), reduction="mean")
                loss_ref = F.mse_loss(noise_pred_ref.float(), noise_ref.float(), reduction="mean")

                loss = loss_ori + loss_ref * args.ref_loss_beta

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print(f"Epoch: {epoch}, Step: {step}, data_time: {load_date_time:.2f}, time: {time.perf_counter() - begin:.2f}, loss: {avg_loss:.4f}, global_step: {global_step}")
                    
                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                begin = time.perf_counter()
            
                # exit() 

if __name__ == "__main__":
    main()