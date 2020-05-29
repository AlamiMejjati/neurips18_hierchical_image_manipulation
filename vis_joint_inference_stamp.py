import argparse
import os
import sys
import torch
from collections import OrderedDict

from data.segmentation_dataset import SegmentationDataset
from util.visualizer import Visualizer
from util import html
from models.joint_inference_model import JointInference
import util.util as util

from skimage import io
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--maskgen_script', type=str,
                    default='scripts/test_pretrained_box2mask_stamp.sh',
                    help='path to a test script for box2mask generator')
parser.add_argument('--imggen_script', type=str,
                    default='scripts/test_pretrained_mask2image_stamp.sh',
                    help='path to a test script for mask2img generator')
parser.add_argument('--gpu_ids', type=int,
                    default=0,
                    help='path to a test script for mask2img generator')
parser.add_argument('--how_many', type=int,
                    default=50,
                    help='number of examples to visualize')
parser.add_argument('--base_name', type=str)

joint_opt = parser.parse_args()

joint_opt.gpu_ids = [joint_opt.gpu_ids]
joint_inference_model = JointInference(joint_opt)

# Hard-coding some parameters
joint_inference_model.opt_maskgen.load_image = 1
joint_inference_model.opt_maskgen.min_box_size = 128
joint_inference_model.opt_maskgen.max_box_size = -1  # not actually used

opt_maskgen = joint_inference_model.opt_maskgen
opt_pix2pix = joint_inference_model.opt_imggen

# Load data
data_loader = SegmentationDataset()
data_loader.initialize(opt_maskgen)
visualizer = Visualizer(opt_maskgen)
# create website
base_name = os.path.join('./results_KID', joint_opt.base_name)  # 'giraffe_run'
web_dir = os.path.join('./results', base_name, 'val')

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s' %
                    ('Joint Inference', 'val'))

# Save directory
if not os.path.exists('./results'):
    os.makedirs('./results')
if not os.path.exists('./results/test_joint_inference'):
    os.makedirs('./results/test_joint_inference')
save_dir = './results/test_joint_inference/'
print(data_loader.dataset_size)
nbmasks = int(data_loader.dataset_size)/5
for i in range(data_loader.dataset_size):
    try:
        # Get data
        bg_nb = i // nbmasks
        box_id = np.mod(i, nbmasks)
        raw_inputs, inst_info = data_loader.get_raw_inputs(i)
        img_orig, label_orig = joint_inference_model.normalize_input(raw_inputs['image'], raw_inputs['label'],
                                                                     normalize_image=False)

        # Add a dimension
        img_orig = img_orig.unsqueeze(0)
        label_orig = label_orig.unsqueeze(0)
        # List of bboxes
        bboxs = inst_info['objects'].values()

        # Select bbox
        bbox_selected = joint_inference_model.sample_bbox(bboxs, opt_maskgen)
        print(bbox_selected)

        print('generating layout...')
        layout, layout_dict, _ = joint_inference_model.gen_layout(
            bbox_selected, label_orig, opt_maskgen)

        print('generating image...')
        image, test_dict, img_generated = joint_inference_model.gen_image(
            bbox_selected, img_orig, layout, opt_pix2pix)
        #   print(layout[0])
        #   print(image[0].shape)

        label_path = raw_inputs['label_path']
        label_filename = 'm_%d_bg_%d.png' %(box_id, bg_nb)
        label_imfilename = 'm_%d_bg_%d.png' % (box_id, bg_nb)
        img_to_write = layout.cpu().data.numpy()[0]
        cur_path = f'{base_name}/mask'
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        io.imsave(f'{cur_path}/{label_filename}', img_to_write[0] / 30)
        cur_path = f'{base_name}/image'
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        img_to_write = image.cpu().data.numpy()[0]
        io.imsave(f'{cur_path}/{label_imfilename}', np.transpose(img_to_write, (1, 2, 0)))

        visuals = OrderedDict([
            ('input_image_patch', util.tensor2im(test_dict['image'][0])),
            ('predicted_label_patch', util.tensor2label(test_dict['label'][0], opt_maskgen.label_nc)),
            ('predicted_image_patch', util.tensor2im(img_generated[0])),
            ('GT_label_canvas', util.tensor2label(label_orig[0], opt_maskgen.label_nc)),
            ('predicted_label_canvas', util.tensor2label(layout[0], opt_maskgen.label_nc)),
            ('GT_image_canvas', util.tensor2im(img_orig[0], normalize=False)),
            ('predicted_image_canvas', util.tensor2im(image[0], normalize=False))
        ])
        print('process image... %s' % ('%05d' % i))
        visualizer.save_images(webpage, visuals, ['%05d' % i])
    except:
        print(raw_inputs['label_path'])
webpage.save()