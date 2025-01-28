import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
from torch.utils.data import Dataset
from skimage import io
import skimage
from skimage import exposure, img_as_ubyte
import os
import cv2
import datetime
import time
import pickle
from math import exp
from torch.autograd import Variable
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lpips
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_msssim import ms_ssim,MS_SSIM,SSIM
from typing import Union, List
from DeepLabV3s import network  #downloaded from https://github.com/VainF/DeepLabV3Plus-Pytorch?tab=readme-ov-file
# also place a .pth file inside DeepLabV3s folder (initially DeepLabV3Plus-Pytorch)
# https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view
import re
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Added by me to avoid PIL error

class LadleNet_plus(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.DeepLab = self.init_deeplab_v3_s()
        
        self.layer_trans = nn.Conv2d(19,3,1,1)
        
        self.channel_trans_1 = self.channel_trans(3,32)
        
        self.enc1 = self.conv(32,32)
        self.enc2 = self.conv(32,32)
        self.channel_trans_2 = self.channel_trans(32,64)
        
        self.enc3 = self.conv(64,64)
        self.enc4 = self.conv(64,64)
        self.channel_trans_3 = self.channel_trans(64,128)
        
        self.enc5 = self.conv(128,128)
        self.enc6 = self.conv(128,128)
        self.channel_trans_4 = self.channel_trans(128,256)
        
        self.enc7 = self.conv(256,256)
        self.enc8 = self.conv(256,256)
        self.channel_trans_5 = self.channel_trans(256,512)
        
        self.enc9 = self.conv(512,512)
        self.enc10 = self.conv(512,512)
        
        self.code = nn.Sequential(nn.Conv2d(512,1024,1,1)
                                 ,nn.Conv2d(1024,512,1,1))
        
        self.dec10 = self.conv(1024,512)
        self.dec9 = self.conv(512,512)
        
        self.channel_trans_7 = self.channel_trans(512,256)
        self.dec8 = self.conv(512,256)
        self.dec7 = self.conv(256,256)
        
        self.channel_trans_8 = self.channel_trans(256,128)
        self.dec6 = self.conv(256,128)
        self.dec5 = self.conv(128,128)
        
        self.channel_trans_9 = self.channel_trans(128,64)
        self.dec4 = self.conv(128,64)
        self.dec3 = self.conv(64,64)
        
        self.channel_trans_10 = self.channel_trans(64,32)
        self.dec2 = self.conv(64,32)
        self.dec1 = self.conv(32,32)
        
        self.channel_trans_11 = self.channel_trans(32,3)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def conv(self, in_channels, out_channels):
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,3,1,1)
                            ,nn.BatchNorm2d(out_channels)
                            ,nn.LeakyReLU(0.2))
        return conv
    
    def channel_trans(self, in_channels, out_channels):
        trans = nn.Sequential(nn.Conv2d(in_channels, out_channels,1,1)
                            ,nn.BatchNorm2d(out_channels)
                            ,nn.LeakyReLU(0.2))
        return trans
    
    def skip_pool(self, skip1, skip2):
        skip = skip1 + skip2
        
        trans = nn.Sequential(nn.MaxPool2d(2)
                             ,nn.LeakyReLU(0.2))
        return trans(skip)
    
    def up_sample(self, x):
        up_sample = nn.Upsample(scale_factor=2)
        
        return up_sample(x)
    
    def up_sample_size(self, x, fea_size):
        up_sample_size = F.interpolate(x, size=fea_size)
        
        return up_sample_size
    
    def fea_cat(self, fea1, fea2):
        ch_in = fea1.shape[1]
        cat_conv = nn.Conv2d(2*ch_in, ch_in, 3, 1, 1).to(device)
        cat = torch.cat((fea1, fea2), 1)
        
        return cat_conv(cat.to(device))
    
    def init_deeplab_v3_s(self):
        model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19,output_stride=8)        
        
        model.load_state_dict(torch.load('DeepLabV3s/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar')['model_state'])
        
        return model
    
    def forward(self, x):
        deeplab_v3s = self.DeepLab(x)
        
        layer_trans = self.layer_trans(deeplab_v3s)

        trans1 = self.channel_trans_1(layer_trans)
        enc1 = self.enc1(trans1)
        enc2 = self.enc2(enc1)
        res1 = self.skip_pool(trans1, enc2)

        trans2 = self.channel_trans_2(res1)
        enc3 = self.enc3(trans2)
        enc4 = self.enc4(enc3)
        res2 = self.skip_pool(trans2, enc4)
        
        trans3 = self.channel_trans_3(res2)
        enc5 = self.enc5(trans3)
        enc6 = self.enc6(enc5)
        res3 = self.skip_pool(trans3, enc6)

        trans4 = self.channel_trans_4(res3)
        enc7 = self.enc7(trans4)
        enc8 = self.enc8(enc7)
        res4 = self.skip_pool(trans4, enc8)

        trans5 = self.channel_trans_5(res4)
        enc9 = self.enc9(trans5)
        enc10 = self.enc10(enc9)
        res5 = self.skip_pool(trans5, enc10)
        
        #------------------------------------------------------------------------
        code = self.code(res5)
        #------------------------------------------------------------------------

        dec10 = self.dec10(torch.cat((code, res5), 1))
        up = self.up_sample(dec10)
        dec9 = self.dec9(up)
        res6 = self.LeakyReLU(up + dec9)

        trans7 = self.channel_trans_7(res6)
        dec8 = self.dec8(torch.cat((trans7, res4), 1))
        up1 = self.up_sample(dec8)
        dec7 = self.dec7(up1)
        res7 = self.LeakyReLU(up1 + dec7)

        trans8 = self.channel_trans_8(res7)
        dec6 = self.dec6(torch.cat((trans8, res3), 1))
        up2 = self.up_sample(dec6)
        dec5 = self.dec5(up2)
        res8 = self.LeakyReLU(up2 + dec5)

        trans9 = self.channel_trans_9(res8)
        dec4 = self.dec4(torch.cat((trans9, res2), 1))
        up3 = self.up_sample(dec4)
        dec3 = self.dec3(up3)
        res9 = self.LeakyReLU(up3 + dec3)

        trans10 = self.channel_trans_10(res9)
        dec2 = self.dec2(torch.cat((trans10, res1), 1))
        up4 = self.up_sample(dec2)
        dec1 = self.dec1(up4)
        res10 = self.LeakyReLU(up4 + dec1)

        trans11 = self.channel_trans_11(res10)
        
        return trans11

class Dataset_creat(Dataset):
    def __init__(self,IR_path,VI_path,transforms:Union[List[transforms.Compose]]):
        super().__init__()
        
        self.IR_path = IR_path
        self.VI_path = VI_path
        self.filename_IR = sorted(os.listdir(self.IR_path))
        self.filename_VI = sorted(os.listdir(self.VI_path))
        self.transform = transforms[0]     
    
     ############# FOR FLIR DATASET ################
        self.paired_filenames = []
        pattern = re.compile(r'frame-\d+')
        for ir_filename in self.filename_IR:
            match = pattern.search(ir_filename)
            if match:
                frame_no = match.group()
                # Find the matching VI file
                for vi_filename in self.filename_VI:
                    if frame_no in vi_filename:
                        self.paired_filenames.append((ir_filename, vi_filename))
                        break
        self.filename_IR = [pair[0] for pair in self.paired_filenames]
        print("Length of paired filenames: ", len(self.paired_filenames))
        self.filename_VI = [pair[1] for pair in self.paired_filenames]
        ###############################################

    def __len__(self):
        ############# FOR FLIR DATASET ################
        return len(self.paired_filenames)
        ###############################################
        return len(os.listdir(self.IR_path))
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        IR_dic = os.path.join(self.IR_path, self.filename_IR[idx])
        VI_dic = os.path.join(self.VI_path, self.filename_VI[idx])
        
        # TODO: NEW WAY TO LOAD IMAGES
        try:
            img_IR = cv2.imread(IR_dic, -1)
            if img_IR is None:
                print(f"Failed to load image {IR_dic}")
                return (None, None)
            img_IR = img_as_ubyte(exposure.rescale_intensity(img_IR))
            img_IR = cv2.equalizeHist(img_IR)
            img_IR = cv2.merge((img_IR, img_IR, img_IR))
        except Exception as e:
            print(f"Failed to load image {IR_dic}: {e}")
            return (None, None)

        try:
            img_VI = cv2.imread(VI_dic, -1)
            if img_VI is None or img_VI.shape[0] == 0:
                print(f"Failed to load image {VI_dic}")
                return (None, None)
            img_VI = cv2.cvtColor(img_VI, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Failed to load image {VI_dic}: {e}")
            return (None, None)
        
        if self.transform != None:
            img_IR = self.transform(img_IR)
            
            img_VI = self.transform(img_VI)
        
        img_IR = torch.transpose(img_IR, 0, 0)
        img_VI = torch.transpose(img_VI, 0, 0)
                
        package = (img_IR, img_VI)
        return package

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

batch_size = 10  # 56 
num_epochs = 50
learning_rate = 0.5
save_step = int(num_epochs * 0.2)

transform_pre = transforms.Compose([transforms.ToTensor()
                                  ,transforms.Resize((300,400))
                                  ,transforms.CenterCrop((192, 256))])

IR = '/home/anton/Desktop/SPRING24/VLR/project/Datasets/images_thermal_train/data'
# VI = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainA'
VI = '/home/anton/Desktop/SPRING24/VLR/project/Datasets/images_rgb_train/data'
dataset = Dataset_creat(IR,VI,[transform_pre])

train_ratio = 0.8
test_ratio = 1 - train_ratio

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LadleNet_plus()
# model = nn.DataParallel(model)
model.to(device)

loss_msssim = MS_SSIM(data_range=1., size_average=True, channel=3, weights=[0.5, 1., 2., 4., 8.])
loss_ssim = SSIM(data_range=1., size_average=True, channel=3)
loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False, cooldown=5)

lr_record = learning_rate
lowest_loss = float('inf')

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f'result/{now}.txt'

dir_name = os.path.dirname(file_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

file = open(file_name, "a")

# sys.stdout = file

torch.autograd.set_detect_anomaly(True)

sample_images = []
image_count = 0
for ir_inputs, vi_inputs in train_loader:
    ir_inputs, vi_inputs = ir_inputs.to(device), vi_inputs.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(vi_inputs)

    # print(f"Outputs shape: {outputs.shape}")
    # print(f"Outputs data type: {outputs.dtype}")
    # print(f"Output sample: {outputs[0, :3, 50, 50]}")
    # print(f"Inputs shape: {ir_inputs.shape}")
    # print(f"Inputs data type: {ir_inputs.dtype}")
    # print(f"Inputs sample: {ir_inputs[0, :3, 50, 50]}")
    # print(f"VI inputs shape: {vi_inputs.shape}")
    # print(f"VI inputs data type: {vi_inputs.dtype}")
    # print(f"VI inputs sample: {vi_inputs[0, :3, 50, 50]}")
    

    # Concatenate outputs and inputs for visualization
    if image_count < 10:
        ir_img = ir_inputs[0].cpu().numpy()
        output_img = outputs[0].cpu().numpy()
        vi_img = vi_inputs[0].cpu().numpy()
        sample_images.append((ir_img, output_img, vi_img))
        print(f"Image count: {image_count}")
        image_count += 1
    else:
        break
    # sample_images = torch.cat((outputs, ir_inputs), dim=3)  # Assuming (B, C, H, W) shape

# First process the images
# 1. IR images are 3 channel [0,1]
# 2. VI images are 3 channel [0,1]
# 3. Output images are 3 channel [-1,1]

processed_images = []
for ir_img, output_img, vi_img in sample_images:
    # need to convert from 1 channel to 3 the ir_image
    ir_img = torch.tensor((ir_img * 255).astype(np.uint8))
    vi_img = torch.tensor((vi_img * 255).astype(np.uint8))
    output_img = torch.tensor((output_img * 255).astype(np.uint8))

    processed_images.append(torch.cat((vi_img, ir_img, output_img), dim=2))

grid = vutils.make_grid(torch.stack(processed_images), nrow=1, padding=2, normalize=False, scale_each=False)
# Convert to PIL image and save
grid_np = grid.numpy()
grid_np = np.transpose(grid_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
grid_image = Image.fromarray(grid_np)
grid_image.save("test_output_input_grid.png")

print("Saved 'test_output_input_grid.png'") # Only process one batch for this test


file.close()
