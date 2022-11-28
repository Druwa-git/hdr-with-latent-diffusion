import os
import torch
import numpy as np
import argparse
import cv2
import time

from PIL import Image
from os.path import join as ospj
from hdr.test_helper import run, get_cond
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from hdr.tonemapping import tonemap_func

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step

def get_model(path_conf, path_ckpt):
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model

def process_image(sample, original_size, resize=True):
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.squeeze().numpy()
    sample = np.transpose(sample, (1, 2, 0))
    if resize:
        sample = cv2.resize(sample, original_size, interpolation=cv2.INTER_CUBIC)
    sample = sample.astype(np.uint8)
    return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default="logs/2022-11-27T15-17-05_ntire-hdr-no-cond")
    parser.add_argument('--data_root', type=str, default="../dataset/ntire_hdr/train")
    parser.add_argument('--config_root', type=str, default="configs/latent-diffusion/ntire-hdr/ntire-hdr-no-cond.yaml")
    args = parser.parse_args()

    target_number = "0992"
    file_name = ospj("train", target_number)
    sample_image = "{}_medium.png".format(target_number)
    dir_path = ospj('data', 'example_conditioning', 'hdr_imaging')
    path_conf, path_ckpt = ospj(args.config_root), ospj(args.log_root, 'checkpoints', 'last.ckpt')
    model = get_model(path_conf, path_ckpt)
    custom_step = 100
    image_path = os.path.join(args.data_root, sample_image)

    example, original_size = get_cond(image_path, square_size=True)
    logs = run(model["model"], example, custom_step)
    sample = process_image(logs["sample"], original_size)
    sample = Image.fromarray(sample.squeeze())

    time_str = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    os.makedirs(ospj(dir_path, time_str), exist_ok=True)
    log_path = ospj(dir_path, time_str)
    pred_path = os.path.join(log_path, sample_image)
    sample.save(os.path.join(dir_path, time_str, sample_image), "png")

    tonemap_func(file_name, pred_path, log_path)
