import os
import torch
import numpy as np
from PIL import Image
from notebook_helpers import get_model, run

def process_image(sample):
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    return sample

if __name__ == '__main__':
    mode = "superresolution"
    model = get_model(mode)
    custom_step = 100
    dir_path = os.path.join("data/example_conditioning/superresolution")
    image_path = os.path.join(dir_path, "sample_0.jpg")
    logs = run(model["model"], image_path, mode, custom_step)
    sample = process_image(logs["sample"])
    sample = Image.fromarray(sample.squeeze())
    sample.save(os.path.join(dir_path, "sample_0_result.png"), "png")
