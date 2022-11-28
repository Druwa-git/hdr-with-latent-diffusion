import torch
import torchvision

from einops import rearrange, repeat
from PIL import Image
from notebook_helpers import make_convolutional_sample



def get_cond(selected_path, square_size, sr=True):
    example = dict()
    c = Image.open(selected_path)
    original_size = c.size

    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    if square_size:
        resize_shape = [512, 512]
        down_scale = 1 / 4
        down_f = lambda x: round(x * down_scale)
        c_gt = torchvision.transforms.functional.resize(c.detach().clone(), size=resize_shape)
        if sr:
            c = torchvision.transforms.functional.resize(c, size=list(map(down_f, resize_shape)))
        else:
            c = torchvision.transforms.functional.resize(c, size=resize_shape)
    else:
        down_f = 1 / 4
        c = torchvision.transforms.functional.resize(c, size=[round(down_f * c.shape[2]),
                                                          round(down_f * c.shape[3])])
        c_gt = c.detach().clone()
    c = rearrange(c, '1 c h w -> 1 h w c')
    c_gt = rearrange(c_gt, '1 c h w -> 1 h w c')
    c = 2. * c - 1.
    c = c.to(torch.device("cuda"))
    example["image"] = c
    example["gt_image"] = c_gt

    return example, original_size

def run(model, example, custom_steps, resize_enabled=False):

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.
    eta = 1.
    make_progrow = True
    custom_shape = None

    if hasattr(model, "split_input_params"):
        delattr(model, "split_input_params")

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False , masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs