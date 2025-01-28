"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import MUNIT_Trainer, UNIT_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=30, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")
parser.add_argument('--input_folder_ir', type=str, help="input image folder for IR images")
parser.add_argument('--segmentation', action='store_true', help="whether to use segmentation or not")
opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_IS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 2, False, new_size=None, crop=False, height=400, width=640)
data_loader_ir = get_data_loader_folder(opts.input_folder_ir, 2, False, new_size=None, crop=False, height=400, width=640)
print("Total number of images: {}".format(len(image_names)))

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config, segmentation=opts.segmentation)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    # Loading generators
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.train()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

processed_images = []
image_count = 0
if opts.trainer == 'MUNIT':
    # Start testing
    # style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=False)
    with torch.no_grad():
        for i, (images, images_ir) in enumerate(zip(data_loader, data_loader_ir)):
            if opts.compute_CIS:
                cur_preds = []
            images = Variable(images.cuda())  # , volatile=True)
            content, _ = encode(images)
            # style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=False)
            style = Variable(torch.randn(images.size(0), style_dim, 1, 1).cuda(), volatile=False)
            # for j in range(opts.num_style):
                # s = style[j].unsqueeze(0)
            outputs = decode(content, style)
            outputs = (outputs + 1) / 2.
            if opts.compute_IS or opts.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opts.compute_IS:
                all_preds.append(pred)
            if opts.compute_CIS:
                cur_preds.append(pred)

            for k in range(images.size(0)):
                print("Processing image %04d" % (i*2+k))
                basename = os.path.basename(f"image_{i*2+k}.png")
                # path = os.path.join(opts.output_folder+"_%02d"%j,basename)
                path = os.path.join(opts.output_folder, "outputs", basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                vutils.save_image(outputs[k].data, path, padding=0, normalize=True)

            if not opts.output_only:
                for k in range(images.size(0)):
                    basename = os.path.basename(f"image_{i*2+k}.png")
                    # also save input images
                    path_in = os.path.join(opts.output_folder, "inputs", basename)
                    if not os.path.exists(os.path.dirname(path_in)):
                        os.makedirs(os.path.dirname(path_in))
                    vutils.save_image(images[k].data, path_in, padding=0, normalize=True)

                    path_gt = os.path.join(opts.output_folder, "gt", basename)
                    if not os.path.exists(os.path.dirname(path_gt)):
                        os.makedirs(os.path.dirname(path_gt))
                    vutils.save_image(images_ir[k].data, path_gt, padding=0, normalize=True)

                    if i%5==0 and image_count < 5:
                        # ensure same size for all images
                        # print(images[k].shape, outputs[k].shape, images_ir[k].shape)
                        # outputs is slightly trimmed
                        ir_img = images_ir[k].cpu().numpy()
                        output_img = outputs[k].cpu().numpy()
                        vi_img = images[k].cpu().numpy()
                        
                        ir_img = ((ir_img*0.5+0.5) * 255).astype(np.uint8).clip(0, 255)
                        vi_img = ((vi_img*0.5+0.5) * 255).astype(np.uint8).clip(0, 255)
                        output_img = ((output_img) * 255).astype(np.uint8).clip(0, 255)
                        
                        ir_img = np.resize(ir_img, (output_img.shape[0], output_img.shape[1], output_img.shape[2]))
                        vi_img = np.resize(vi_img, (output_img.shape[0], output_img.shape[1], output_img.shape[2]))
                        
                        processed_images.append(np.concatenate((ir_img, output_img, vi_img), axis=2))

                        image_count += 1

            # if opts.compute_CIS:
            #         cur_preds = np.concatenate(cur_preds, 0)
            #         py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            #         for j in range(cur_preds.shape[0]):
            #             pyx = cur_preds[j, :]
            #             CIS.append(entropy(pyx, py))

    print("Processing test image samples...")
    grid = vutils.make_grid(torch.tensor(np.stack(processed_images)), nrow=1, padding=2, normalize=False, scale_each=False)
    grid_np = grid.numpy()
    grid_np = np.transpose(grid_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    grid_image = Image.fromarray(grid_np)
    grid_image.save(os.path.join(opts.output_folder, "test_images.png"))
    print("Saved")

    if opts.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

        if opts.compute_IS:
            print("Inception Score: {}".format(np.exp(np.mean(IS))))
        if opts.compute_CIS:
            print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))

elif opts.trainer == 'UNIT':
    # Start testing
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)

        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
        basename = os.path.basename(names[1])
        path = os.path.join(opts.output_folder, basename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
else:
    pass
