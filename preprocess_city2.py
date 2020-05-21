import os
import glob
from shutil import copy2
from PIL import Image
import json
import numpy as np
import argparse
import shutil
from skimage import io
from tqdm import tqdm
from itertools import product


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def copy_file(src, dst):
    if os.path.exists(dst):
        os.rmdir(dst)
    shutil.copytree(src, dst)


def construct_single_class_dataset(image_source_path, anno_source_path, target_paths, target_cls):
    image_paths = glob.glob(image_source_path + '/*/*.png', )
    image_names = list(map(lambda x: os.path. \
                           basename(x). \
                           replace('_leftImg8bit.png', ''), image_paths))

    for image_name, image_path in zip(image_names, image_paths):
        # Obtain the label and inst
        label_path = glob.glob(anno_source_path + '/*/' + image_name + '*_labelIds.png')[0]
        inst_path = glob.glob(anno_source_path + '/*/' + image_name + '*_instanceIds.png')[0]
        inst_map = np.array(Image.open(inst_path), dtype=np.int32)
        label_map = np.array(Image.open(label_path), dtype=np.int32)

        # get a list of unique instances
        H, W = inst_map.shape
        inst_ids = np.unique(inst_map)

        anno_idx = 0

        for iid in inst_ids:
            if int(iid) // 1000 != target_cls:  # filter out non-instance masks
                continue

            # Get the current map
            cur_inst_map = inst_map * (inst_map == iid)
            cur_label_map = label_map * (inst_map == iid)

            # Get info for the current map
            ys, xs = np.where(inst_map == iid)
            ymin, ymax, xmin, xmax = \
                ys.min(), ys.max(), xs.min(), xs.max()
            cls_label = np.median(label_map[inst_map == iid])

            # If the majority of the region is for another class, then drop this mask
            if cls_label != target_cls: continue

            # Construct the label information
            object_info = {'bbox': [xmin, ymin, xmax, ymax], 'cls': target_cls}
            inst_info = {'imgHeight': H,
                         'imgWidth': W,
                         'objects': {str(iid): object_info}}

            save_one_sample(target_paths, image_name, anno_idx,
                            image_path, cur_inst_map, cur_label_map, inst_info)

            anno_idx += 1


def save_one_sample(target_paths, image_name, anno_idx,
                    image_path, cur_inst_map, cur_label_map, inst_info):
    # Copy image file and rename
    source = image_path
    target = os.path.join(target_paths['img'], f'{image_name}-{anno_idx}.png')
    shutil.copyfile(source, target)

    # Save the label and inst map
    target = os.path.join(target_paths['inst'], f'{image_name}-{anno_idx}.png')
    Image.fromarray(cur_inst_map).save(target)

    target = os.path.join(target_paths['label'], f'{image_name}-{anno_idx}.png')
    Image.fromarray(cur_label_map).save(target)

    # Save the json bbox file
    target = os.path.join(target_paths['bbox'], f'{image_name}-{anno_idx}.json')
    with open(target, 'w') as f:
        json.dump(inst_info, f, cls=NpEncoder)
        print('wrote a bbox summary of %s-%s to %s' % (image_name, anno_idx, target))


parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('--source_path', default='datasets/cityscape', help='original cityscapes dataset path', type=str)
parser.add_argument('--target_path', default='datasets/cityscape_new', help='save path for new cityscape dataset',
                    type=str)

# organize image
if __name__ == '__main__':

    args = parser.parse_args()
    image_path_source = os.path.join(args.source_path, 'leftImg8bit')
    anno_path_source = os.path.join(args.source_path, 'gtFine')

    # create the target paths and ensure the paths exist
    subsets = ['train', 'val']
    target_folders = ['img', 'label', 'inst', 'bbox']
    target_paths = {subset: {name: os.path.join(args.target_path, f'{subset}_{name}')
                             for name in target_folders}
                    for subset in subsets}

    for _, tpaths in target_paths.items():
        for _, tpah in tpaths.items():
            if not os.path.exists(tpah):
                os.makedirs(tpah)

    # iterate different subsets
    for subset in subsets:
        image_path_source_subset = os.path.join(image_path_source, subset)
        anno_path_source_subset = os.path.join(anno_path_source, subset)

        construct_single_class_dataset(image_path_source_subset,
                                       anno_path_source_subset,
                                       target_paths[subset],
                                       target_cls=24)