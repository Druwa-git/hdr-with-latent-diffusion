import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NTIREHDRBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 image_type='medium',
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.readlines()
        self._length = len(self.image_paths)

        name_func = lambda x, img_t: x.rstrip()+"_{}.png".format(img_t)
        self.labels = {
            "relative_file_path_": [name_func(l, image_type) for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, name_func(l, image_type))
                           for l in self.image_paths],
            "gt_path_": [os.path.join(self.data_root, name_func(l, 'gt'))
                           for l in self.image_paths]
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        gt_image = Image.open(example["gt_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
            gt_image = gt_image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        gt_img = np.array(gt_image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        gt_img = gt_img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        gt_image = Image.fromarray(gt_img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            gt_image = gt_image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        gt_image = np.array(gt_image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["gt_image"] = (gt_image / 127.5 - 1.0).astype(np.float32)

        return example


class NTIREHDRTrain(NTIREHDRBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="../dataset/ntire_hdr/train_sub.txt", data_root="../dataset/ntire_hdr", **kwargs)


class NTIREHDRValidation(NTIREHDRBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="../dataset/ntire_hdr/val_sub.txt", data_root="../dataset/ntire_hdr", **kwargs)
