import json
import cv2
import os

import argparse
from itertools import product
import numpy as np
from skimage import io


def normalize_bbx(bbx, h, w):
    xmin, ymin, xmax, ymax = bbx

    return [int(xmin / w * 256), int(ymin / h * 256), int(xmax / w * 256), int(ymax / h * 256)]


def normalize_json(json_filename):
    with open(json_filename, 'r') as f:
        k = json.load(f)
    h, w = k['imgHeight'], k['imgWidth']
    k['objects'] = {i: {'bbox': normalize_bbx(j['bbox'], h, w), 'cls': 30} for i, j in k['objects'].items()}
    k['imgHeight'] = k['imgWidth'] = 256
    return k


def parse_box(box_image):
    ys, xs = np.where(box_img == 255)
    ymin, ymax, xmin, xmax = \
        ys.min(), ys.max(), xs.min(), xs.max()

    return dict(
        imgHeight=256,
        imgWidth=256,
        objects={'30000': {'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)], 'cls': 30}}
    )


parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('--subset_name', default='giraffe', type=str)
parser.add_argument('--base_background_path', default='./datasets/inference/background', type=str)
parser.add_argument('--base_box_path', default='./datasets/inference/box', type=str)

if __name__ == "__main__":

    args = parser.parse_args()
    subset_name = args.subset_name

    base_path_train = f'./datasets/stamp_{subset_name}'
    base_background_path = args.base_background_path
    base_box_path = args.base_box_path

    cur_json_path = f'{base_path_train}/val_bbox'
    cur_background_path = f'{base_background_path}/{subset_name}'
    cur_box_path = f'{base_box_path}/{subset_name}'

    output_run_path = f'{base_path_train}_run'  # Run for comparing many examples and generat KID/FID Scores
    output_show_path = f'{base_path_train}_show'  # Show for examples in the paper

    for name in ['img', 'label', 'inst', 'bbox']:
        cur_path = os.path.join(output_run_path, f'val_{name}')
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

    for name in ['img', 'label', 'inst', 'bbox']:
        cur_path = os.path.join(output_show_path, f'val_{name}')
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

    bgs = os.listdir(cur_background_path)
    jss = os.listdir(cur_json_path)
    bxs = os.listdir(cur_box_path)

    for bg_name, js_name in product(bgs, jss):  # generate run images - using boxes inside the dataset
        combined_name = bg_name.split('.')[0] + '_' + js_name.split('.')[0]

        bg_img = cv2.imread(f'{cur_background_path}/{bg_name}')
        bg_img = cv2.resize(bg_img, (256, 256))
        cv2.imwrite(f'{output_run_path}/val_img/{combined_name}.png', bg_img)

        bg_placer = np.zeros(bg_img.shape[:2], 'uint8')
        io.imsave(f'{output_run_path}/val_label/{combined_name}.png', bg_placer)
        io.imsave(f'{output_run_path}/val_inst/{combined_name}.png', bg_placer.astype('uint16'))

        js_new = normalize_json(f'{cur_json_path}/{js_name}')
        with open(f'{output_run_path}/val_bbox/{combined_name}.json', 'w') as f:
            json.dump(js_new, f)

    for bg_name, box_name in product(bgs, bxs):  # generate show images - using provided boxes
        combined_name = bg_name.split('.')[0] + '_' + box_name.split('.')[0]

        bg_img = cv2.imread(f'{cur_background_path}/{bg_name}')
        bg_img = cv2.resize(bg_img, (256, 256))
        cv2.imwrite(f'{output_show_path}/val_img/{combined_name}.png', bg_img)

        bg_placer = np.zeros(bg_img.shape[:2], 'uint8')
        io.imsave(f'{output_show_path}/val_label/{combined_name}.png', bg_placer)
        io.imsave(f'{output_show_path}/val_inst/{combined_name}.png', bg_placer.astype('uint16'))

        box_img = io.imread(f'{cur_box_path}/{box_name}')

        js_new = parse_box(box_img)
        with open(f'{output_show_path}/val_bbox/{combined_name}.json', 'w') as f:
            json.dump(js_new, f)