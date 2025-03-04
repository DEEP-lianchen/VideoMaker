from accelerate.logging import get_logger
import torch
import numpy as np
import cv2
import PIL
logger = get_logger(__name__)


def save_image(tensor, filename):
    tensor = tensor.permute(1, 2, 0)
    image_numpy = tensor.cpu().numpy()
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_numpy)

def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image