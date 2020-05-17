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
        if '.png' in img_name:
            img = io.imread(os.path.join(src_path, img_name))
            img[img == 255] = 30
            io.imsave(os.path.join(dst_path1, img_name), img)
            img = img.astype('uint16')
            img[img == 30] = 30*1000
            io.imsave(os.path.join(dst_path2, img_name), img)

def process_files(source_base_path, target_base_pth, subset, COCO_path):

    dst_path = {}
    for name in ['img','label','inst','bbox']:
        cur_path = os.path.join(target_base_pth, subset + '_' + name)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        dst_path[name] = cur_path

    print('process label and inst copy')
    copy_label(source_base_path, dst_path['label'], dst_path['inst'])
    ### copy_file(dst_path['label'], dst_path['inst'])
    print('process img copy')
    if COCO_path:
        copy_img_file(source_base_path, dst_path['img'], COCO_path+'/'+subset+'2017')
    construct_box(dst_path['inst'], dst_path['label'], dst_path['bbox'])

def copy_img_file(source_base_path, target_base_path, COCO_path):
    print({target_base_path})
    for filepath in tqdm(os.listdir(source_base_path)):
        if ('.png' in filepath) or ('.jpg' in filepath):
            basename = os.path.basename(filepath).split('.')[0]
            filename = basename.split('_')[0]
            indexid = basename.split('_')[1]
            if os.path.isfile(COCO_path + '/' + filename + '.jpg'):
                os.symlink(COCO_path + '/' + filename + '.jpg', target_base_path + '/' + filename+'_'+indexid+'.jpg')
            else:
                print('File %s.jpg not Found. Please check mannually.' %filename)

# organize image
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-s', '--subset', help='class for training the model', type=str)
    parser.add_argument('-d', '--datapath',default='/home/yam28/Documents/phdYoop/datasets/COCO', type=str)
    args = parser.parse_args()

    source_base_path_train = 'dataset/train/' + args.subset
    source_base_path_train_aug = 'dataset/train/' + args.subset+'_silvia'
    source_base_path_valid = 'dataset/val/' + args.subset

    target_base_pth = 'datasets/stamp_' + args.subset + '_aug'
    COCO_path = args.datapath


    process_files(source_base_path_train_aug, target_base_pth, 'train', None)