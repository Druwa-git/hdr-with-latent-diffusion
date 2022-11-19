import os
from os.path import join as ospj
import numpy as np
import data_io as io
import cv2
import metrics as m
import argparse

_IMAGE_TYPE = ('gt', 'long', 'medium', 'short', 'result')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_type', type=str, default='gt',
                        choices=_IMAGE_TYPE)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--data_root', type=str, default='../dataset/ntire_hdr/')
    args = parser.parse_args()

    tonemap_dir = ospj(args.data_root, 'result')
    os.makedirs(tonemap_dir, exist_ok=True)

    target_list_file = open(ospj(args.data_root, args.split+".txt"), "r")
    for target_name in target_list_file.readlines():
        target_name = target_name.rstrip()
        target_number = target_name.split('/')[-1]

        target_path = ospj(args.data_root, target_name + "_gt.png")
        align_path = ospj(args.data_root, 'alignratio', '{}_alignratio.npy'.format(target_number))
        hdr_image = io.imread_uint16_png(target_path, align_path)
        hdr_linear_image = hdr_image ** 2.24
        norm_perc = np.percentile(hdr_linear_image, 99).copy()

        if args.image_type != 'gt':
            target_path = ospj(args.data_root, target_name + "_{}.png".format(args.image_type))
            hdr_image = io.imread_uint16_png(target_path, align_path)
            hdr_linear_image = hdr_image ** 2.24

        hdr_image = (m.tanh_norm_mu_tonemap(hdr_linear_image, norm_perc) * 255.).round().astype(np.uint8)
        cv2.imwrite(ospj(tonemap_dir, "{}_tone_mapped_{}.png".format(target_number, args.image_type)),
                    cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR))