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


def construct_box(inst_root, label_root, dst):
    inst_list = os.listdir(inst_root)
    cls_list = os.listdir(label_root)
    for inst, cls in zip(*(inst_list, cls_list)):
        inst_map = Image.open(os.path.join(inst_root, inst))
        # inst_map = Image.open(inst)
        inst_map = np.array(inst_map, dtype=np.int32)
        cls_map = Image.open(os.path.join(label_root, cls))
        # cls_map = Image.open(cls)
        cls_map = np.array(cls_map, dtype=np.int32)
        H, W = inst_map.shape
        # get a list of unique instances
        inst_info = {'imgHeight':H, 'imgWidth':W, 'objects':{}}
        inst_ids = np.unique(inst_map)
        for iid in inst_ids: 
            if int(iid) <=0: # filter out non-instance masks
                continue
            ys,xs = np.where(inst_map==iid)
            ymin, ymax, xmin, xmax = \
                    ys.min(), ys.max(), xs.min(), xs.max()
            cls_label = np.median(cls_map[inst_map==iid])
            inst_info['objects'][str(iid)] = {'bbox': [xmin, ymin, xmax, ymax], 'cls':int(cls_label)}
        # write a file to path
        filename = os.path.splitext(os.path.basename(inst))[0]
        savename = os.path.join(dst, filename + '.json')
        with open(savename, 'w') as f:
            json.dump(inst_info, f, cls=NpEncoder)
        print('wrote a bbox summary of %s to %s' % (inst, savename))



def copy_label(src_path, dst_path1, dst_path2):
    for img_name in tqdm(os.listdir(src_path)):
        img = io.imread(f'{src_path}/{img_name}')
        img[img == 255] = 30
        io.imsave(f'{dst_path1}/{img_name}', img)
        img = img.astype('uint16')
        img[img == 30] = 30*1000
        io.imsave(f'{dst_path2}/{img_name}', img)

def process_files(source_base_path, target_base_pth, subset, COCO_path):

    dst_path = {}
    for name in ['img','label','inst','bbox']:
        cur_path = os.path.join(target_base_pth, f'{subset}_{name}')
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        dst_path[name] = cur_path

    print('process label and inst copy')
    copy_label(source_base_path, dst_path['label'], dst_path['inst'])
    ### copy_file(dst_path['label'], dst_path['inst'])
    print('process img copy')
    copy_img_file(source_base_path, dst_path['img'], f'{COCO_path}/{subset}2017')
    construct_box(dst_path['inst'], dst_path['label'], dst_path['bbox'])

def copy_img_file(source_base_path, target_base_path, COCO_path):
    print({target_base_path})
    for filepath in tqdm(os.listdir(source_base_path)):
        basename = os.path.basename(filepath).split('.')[0]
        filename = basename.split('_')[0]
        indexid = basename.split('_')[1]
        if os.path.isfile(f'{COCO_path}/{filename}.jpg'):
            shutil.copy2(f'{COCO_path}/{filename}.jpg',
                 f'{target_base_path}/{filename}_{indexid}.jpg')
        else:
            print(f'File {filename}.jpg not Found. Please check mannually.')

# organize image
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-s', '--subset', help='class for training the model', type=str)
    args = parser.parse_args()

    source_base_path_train = f'datasets/stamp/train/{args.subset}'
    source_base_path_valid = f'datasets/stamp/val/{args.subset}'

    target_base_pth = f'datasets/stamp_{args.subset}'
    COCO_path = '/mnt/sdb/data/COCO'
    # train_img_dst = os.path.join(target_base_pth, 'train_img')
    # train_label_dst = os.path.join(target_base_pth, 'train_label')
    # train_inst_dst = os.path.join(target_base_pth, 'train_inst')
    # train_bbox_dst = os.path.join(target_base_pth, 'train_bbox')
    # val_img_dst = os.path.join(target_base_pth, 'val_img')
    # val_label_dst = os.path.join(target_base_pth, 'val_label')
    # val_inst_dst = os.path.join(target_base_pth, 'val_inst')
    # val_bbox_dst = os.path.join(target_base_pth, 'val_bbox')

    # if not os.path.exists(train_img_dst):
    #     os.makedirs(train_img_dst)
    # if not os.path.exists(train_label_dst):
    #     os.makedirs(train_label_dst)
    # if not os.path.exists(train_inst_dst):
    #     os.makedirs(train_inst_dst)
    # if not os.path.exists(val_img_dst):
    #     os.makedirs(val_img_dst)
    # if not os.path.exists(val_label_dst):
    #     os.makedirs(val_label_dst)
    # if not os.path.exists(val_inst_dst):
    #     os.makedirs(val_inst_dst)

    process_files(source_base_path_train, target_base_pth, 'train', COCO_path)
    process_files(source_base_path_valid, target_base_pth, 'val', COCO_path)
    # # train_image
    # copy_file(source_base_path_train, train_img_dst)
    # # train_label
    # copy_file('datasets/cityscape/gtFine/train',\
    #         '*_labelIds.png', train_label_dst)
    # # train_inst
    # copy_file('datasets/cityscape/gtFine/train',\
    #         '*_instanceIds.png', train_inst_dst)
    # # val_image
    # copy_file('datasets/cityscape/leftImg8bit/val',\
    #         '*_leftImg8bit.png', val_img_dst)
    # # val_label
    # copy_file('datasets/cityscape/gtFine/val',\
    #         '*_labelIds.png', val_label_dst)
    # # val_inst
    # copy_file('datasets/cityscape/gtFine/val',\
    #         '*_instanceIds.png', val_inst_dst)

    # if not os.path.exists(train_bbox_dst):
    #     os.makedirs(train_bbox_dst)
    # if not os.path.exists(val_bbox_dst):
    #     os.makedirs(val_bbox_dst)
    # # wrote a bounding box summary 
    # construct_box('datasets/cityscape/gtFine/train',\
    #         '*_instanceIds.png', '*_labelIds.png', train_bbox_dst)
    # construct_box('datasets/cityscape/gtFine/val',\
    #         '*_instanceIds.png', '*_labelIds.png', val_bbox_dst) 
