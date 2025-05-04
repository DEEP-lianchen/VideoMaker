import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from easydict import EasyDict
from datasets import get_dataset

def denorm(x):
    """将 [-1, 1] 区间归一化到 [0, 1]"""
    return x * 0.5 + 0.5

def tensor_to_pil(tensor):
    """
    将 tensor 转成 PIL 图片，自动处理：
    - 归一化 [-1,1] -> [0,1]
    - 去掉 batch 维度 [1, C, H, W]
    """
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # 去掉 batch
    img = denorm(tensor).clamp(0, 1)  # 归一化并clamp
    return TF.to_pil_image(img)

def save_sample_outputs(sample, save_dir='sample_outputs'):
    os.makedirs(save_dir, exist_ok=True)

    video = sample['video']  # shape: [T, C, H, W], 范围 [-1,1]
    first_frame = sample['masked_first_frame']  # shape: [1, 3, H, W]

    # 保存视频每一帧
    for i, frame in enumerate(video):
        frame_img = tensor_to_pil(frame)
        frame_img.save(os.path.join(save_dir, f'video_frame_{i:03d}.png'))

    # 保存masked_first_frame
    first_img = tensor_to_pil(first_frame)
    first_img.save(os.path.join(save_dir, 'masked_first_frame.png'))

    print(f"Saved outputs to: {save_dir}")

# ===============================
# 下面是使用示例
args = EasyDict({
    'train_data_dir': '/data3/lyc_datas/VideoBoothDataset',
    'num_frames': 16,
    'frame_interval': 8,
    'image_size': 512
})

dataset = get_dataset(args)
print("Total samples:", len(dataset))

sample = dataset[0]  # 拿第一个样本
print(sample.keys())  # 打印包含哪些字段
print("Video ID:", sample['video_id'])
print("Video prompt:", sample['video_prompt'])
print("Video shape:", sample['video'].shape)  # [16, 3, 512, 512]
print("Masked frame shape:", sample['masked_first_frame'].shape)  # [1, 3, 512, 512]

# 保存结果
save_sample_outputs(sample, save_dir='sample_outputs')
