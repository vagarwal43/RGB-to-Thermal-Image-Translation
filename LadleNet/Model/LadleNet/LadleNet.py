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
import re
from typing import Union, List
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LadleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        #----------------------------------------------------------------------#
        
        self.channel_trans_1_ = self.channel_trans(3,32)
        
        self.enc_1 = self.conv(32,32)
        self.enc_2 = self.conv(32,32)
        self.channel_trans_2_ = self.channel_trans(32,64)
        
        self.enc_3 = self.conv(64,64)
        self.enc_4 = self.conv(64,64)
        self.channel_trans_3_ = self.channel_trans(64,128)
        
        self.enc_5 = self.conv(128,128)
        self.enc_6 = self.conv(128,128)
        self.channel_trans_4_ = self.channel_trans(128,256)
        
        self.enc_7 = self.conv(256,256)
        self.enc_8 = self.conv(256,256)
        self.channel_trans_5_ = self.channel_trans(256,512)
        
        self.enc_9 = self.conv(512,512)
        self.enc_10 = self.conv(512,512)
        
        self.code_ = nn.Sequential(nn.Conv2d(512,1024,1,1)
                                 ,nn.Conv2d(1024,512,1,1))
        
        self.dec_10 = self.conv(1024,512)
        self.dec_9 = self.conv(512,512)
        
        self.channel_trans_7_ = self.channel_trans(512,256)
        self.dec_8 = self.conv(512,256)
        self.dec_7 = self.conv(256,256)
        
        self.channel_trans_8_ = self.channel_trans(256,128)
        self.dec_6 = self.conv(256,128)
        self.dec_5 = self.conv(128,128)
        
        self.channel_trans_9_ = self.channel_trans(128,64)
        self.dec_4 = self.conv(128,64)
        self.dec_3 = self.conv(64,64)
        
        self.channel_trans_10_ = self.channel_trans(64,32)
        self.dec_2 = self.conv(64,32)
        self.dec_1 = self.conv(32,32)
        
        self.channel_trans_11_ = self.channel_trans(32,3)
        
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
    
    def fea_cat(self, fea1, fea2):
        ch_in = fea1.shape[1]
        cat_conv = nn.Conv2d(2*ch_in, ch_in, 3, 1, 1).to(self.device)
        cat = torch.cat((fea1, fea2), 1)
        
        return cat_conv(cat.to(self.device))
    
    def forward(self, x):
        trans1 = self.channel_trans_1(x)
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
        
        trans_1 = self.fea_cat(self.channel_trans_1_(trans11), res10)
        enc_1 = self.enc_1(trans_1)
        enc_2 = self.enc_2(enc_1)
        res_1 = self.skip_pool(trans_1, enc_2)

        trans_2 = self.fea_cat(self.channel_trans_2_(res_1), res9)
        enc_3 = self.enc_3(trans_2)
        enc_4 = self.enc_4(enc_3)
        res_2 = self.skip_pool(trans_2, enc_4)

        trans_3 = self.fea_cat(self.channel_trans_3_(res_2), res8)
        enc_5 = self.enc_5(trans_3)
        enc_6 = self.enc_6(enc_5)
        res_3 = self.skip_pool(trans_3, enc_6)

        trans_4 = self.fea_cat(self.channel_trans_4_(res_3), res7)
        enc_7 = self.enc_7(trans_4)
        enc_8 = self.enc_8(enc_7)
        res_4 = self.skip_pool(trans_4, enc_8)

        trans_5 = self.fea_cat(self.channel_trans_5_(res_4), res6)
        enc_9 = self.enc_9(trans_5)
        enc_10 = self.enc_10(enc_9)
        res_5 = self.skip_pool(trans_5, enc_10)

        #------------------------------------------------------------------------
        code_ = self.code_(res_5) 
        #------------------------------------------------------------------------

        dec_10 = self.dec_10(torch.cat((code_, res_5), 1) + torch.cat((code, res5), 1))
        up_ = self.up_sample(dec_10)
        dec_9 = self.dec_9(up_)
        res_6 = self.LeakyReLU(up_ + dec_9)

        trans_7 = self.channel_trans_7_(res_6)
        dec_8 = self.dec_8(torch.cat((trans_7, res_4), 1) + torch.cat((trans7, res4), 1))
        up_1 = self.up_sample(dec_8)
        dec_7 = self.dec_7(up_1)
        res_7 = self.LeakyReLU(up_1 + dec_7)

        trans_8 = self.channel_trans_8_(res_7)
        dec_6 = self.dec_6(torch.cat((trans_8, res_3), 1) + torch.cat((trans8, res3), 1))
        up_2 = self.up_sample(dec_6)
        dec_5 = self.dec_5(up_2)
        res_8 = self.LeakyReLU(up_2 + dec_5)

        trans_9 = self.channel_trans_9_(res_8)
        dec_4 = self.dec_4(torch.cat((trans_9, res_2), 1) + torch.cat((trans9, res2), 1))
        up_3 = self.up_sample(dec_4)
        dec_3 = self.dec_3(up_3)
        res_9 = self.LeakyReLU(up_3 + dec_3)

        trans_10 = self.channel_trans_10_(res_9)
        dec_2 = self.dec_2(torch.cat((trans_10, res_1), 1) + torch.cat((trans10, res1), 1))
        up_4 = self.up_sample(dec_2)
        dec_1 = self.dec_1(up_4)
        res_10 = self.LeakyReLU(up_4 + dec_1)

        trans_11 = self.channel_trans_11_(res_10)

        return trans_11

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
                
        package = (img_IR,img_VI)
        return package

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    resume_path = 'weight/2021-05-06_16-00-00_LadleNet_plus.pth'
    resume_path = None

    batch_size = 5  # 40
    num_epochs = 30
    learning_rate = 0.1
    save_step = int(num_epochs * 0.1)
    
    checkpoint = None
    epoch_resume = 0
    if resume_path != None:
        checkpoint = torch.load(resume_path)
        epoch_resume = checkpoint['epoch']
        num_epochs = checkpoint['metadata']['num_epochs']
        learning_rate = checkpoint['metadata']['learning_rate']
        batch_size = checkpoint['metadata']['batch_size']
        print(f"Resuming training from epoch {epoch_resume+1}")

    transform_pre = transforms.Compose([transforms.ToTensor()
                                    ,transforms.Resize((300,400))
                                    ,transforms.CenterCrop((192, 256))])

    # IR = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainB'
    IR = '/home/anton/Desktop/SPRING24/VLR/project/Datasets/images_thermal_train/data'
    # VI = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainA'
    VI = '/home/anton/Desktop/SPRING24/VLR/project/Datasets/images_rgb_train/data'
    dataset = Dataset_creat(IR, VI, [transform_pre])

    train_ratio = 0.8
    test_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LadleNet()
    model = nn.DataParallel(model)
    model.to(device)

    loss_msssim = MS_SSIM(data_range=1., size_average=True, channel=3, weights=[0.5, 1., 2., 4., 8.])
    loss_ssim = SSIM(data_range=1., size_average=True, channel=3)
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False, cooldown=10)

    lr_record = learning_rate
    highest_msssim = - float('inf')

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'result/{now}.txt'

    # TODO: Create the directory if it does not exist
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # TODO: Redirect remove the print statement
    # sys.stdout = file
    if resume_path != None and checkpoint != None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Loaded model from {resume_path}")

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        start_time = time.time()

        model.train()
        
        train_loss = 0.0
        msssim_loss = 0.0
        l1_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            start_time_batch = time.time()
            print(f'Batch {i+1}/{len(train_loader)}')
            ir_inputs, vi_inputs = data
            ir_inputs, vi_inputs = ir_inputs.to(device), vi_inputs.to(device)

            optimizer.zero_grad()
            # TODO: Reconfigure the model
            outputs = model(vi_inputs)  # My model VI to IR
            msssim = 0.84 * (1 - loss_ssim(outputs, ir_inputs))
            l1 = (1-0.84) * loss_l1(outputs, ir_inputs)

            total_loss = msssim + l1

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            msssim_loss += msssim.item()
            l1_loss += l1.item()
            end_time_batch = time.time()
            time_diff = end_time_batch - start_time_batch
            print(f'Batch {i+1}/{len(train_loader)} Time: {time_diff}s')

        average_loss = train_loss / len(train_loader)
        avg_msssim_loss = msssim_loss / len(train_loader)
        avg_l1_loss = l1_loss / len(train_loader)

        model.eval()

        msssim_val = 0.0
        l1_val = 0.0
        ssim_val = 0.0

        sample_images = []
        image_count = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                print(f'Validation Batch {i+1}/{len(val_loader)}')
                ir_inputs_val, vi_inputs_val = data
                ir_inputs_val, vi_inputs_val = ir_inputs_val.to(device), vi_inputs_val.to(device)

                outputs_val = model(vi_inputs_val)

                # TODO: Sample the images
                if i % 15 == 0 and image_count < 10:
                    ir_img = ir_inputs_val[0].cpu().numpy()
                    output_img = outputs_val[0].cpu().numpy()
                    vi_img = vi_inputs_val[0].cpu().numpy()
                    sample_images.append((ir_img, output_img, vi_img))
                    image_count += 1

                ssim_val_value = loss_ssim(outputs_val, ir_inputs_val)
                l1_val_value = loss_l1(outputs_val, ir_inputs_val)
                msssim_val_value = loss_msssim(outputs_val, ir_inputs_val)

                ssim_val += ssim_val_value.item()
                l1_val += l1_val_value.item()
                msssim_val += msssim_val_value.item()

        # TODO: Save the images
        processed_images = []
        print("Processing test image samples...")
        for ir_img, output_img, vi_img in sample_images:
            ir_img = torch.tensor((ir_img * 255).astype(np.uint8))
            vi_img = torch.tensor((vi_img * 255).astype(np.uint8))
            output_img = torch.tensor(((output_img + 1) / 2 * 255).astype(np.uint8))

            processed_images.append(torch.cat((vi_img, ir_img, output_img), dim=2))

        grid = vutils.make_grid(torch.stack(processed_images), nrow=1, padding=2, normalize=False, scale_each=False)
        grid_np = grid.numpy()
        grid_np = np.transpose(grid_np, (1, 2, 0))
        grid_image = Image.fromarray(grid_np)
        if not os.path.exists("result/test_images"):
            os.makedirs("result/test_images")
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        grid_image.save(f"result/test_images/{now}_epoch_{epoch}.png")

        avg_ssim_val = ssim_val / len(val_loader)
        avg_msssim_val = msssim_val / len(val_loader)
        avg_l1_val = l1_val / len(val_loader)

        scheduler.step(average_loss)
        
        if avg_msssim_val > highest_msssim:
            now = datetime.datetime.now().strftime("%Y-%m-%d")
            highest_msssim = avg_msssim_val
            highest_msssim_save_path = os.path.join("weight", f"LadleNet_highest_msssim_{now}.pth")
            if not os.path.exists("weight"):
                os.makedirs("weight")
            state_dict_lowest_loss = model.state_dict()
            metadata_lowest_loss = {'learning_rate': '0.001',
                                    'batch_size': batch_size,
                                    'num_epochs': '130',
                                    'current_epoch': '130',
                                    'optimizer_name': type(optimizer).__name__.lower(),
                                    'loss_function': 'Loss_Content(MS-SSIM, L1)',
                                    'MS-SSIM': np.round(avg_msssim_val, 6),
                                    'L1': np.round(avg_l1_val, 6)}
            optimizer_lowest_loss = optimizer.state_dict()
            scheduler_lowest_loss = scheduler.state_dict()
            epoch_lowest_loss = epoch

            torch.save({'state_dict': state_dict_lowest_loss, 'metadata': metadata_lowest_loss,
                        'optimizer': optimizer_lowest_loss, 'scheduler': scheduler_lowest_loss,
                        'epoch': epoch_lowest_loss}, highest_msssim_save_path)

        if (epoch+1) % save_step == 0 or (epoch + 1) == num_epochs:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join("weight", f"{now}_LadleNet.pth")
            if not os.path.exists("weight"):
                os.makedirs("weight")
            state_dict = model.state_dict()
            metadata = {'learning_rate': '0.001',
                        'batch_size': batch_size,
                        'num_epochs': '120',
                        'current_epoch': '120',
                        'optimizer_name': type(optimizer).__name__.lower(),
                        'loss_function': 'Loss_Content(MS-SSIM, L1)',
                        'MS-SSIM': f"(train){np.round(avg_msssim_loss, 6)} / (val){np.round(avg_msssim_val, 6)}",
                        'L1': f"(train){np.round(avg_l1_loss, 6)} / (val){np.round(avg_l1_val, 6)}"}
            optimizer_state = optimizer.state_dict()
            scheduler_state = scheduler.state_dict()
            epoch_record = epoch
            torch.save({'state_dict': state_dict, 'metadata': metadata,
                        'optimizer': optimizer_state, 'scheduler': scheduler_state,
                        'epoch': epoch_record}, save_path)
            
        end_time = time.time()
        time_diff = end_time - start_time
        hours, remainder = divmod(time_diff, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if lr_record != scheduler.optimizer.param_groups[0]['lr']:
            lr_record = scheduler.optimizer.param_groups[0]['lr']
            print(f'--------------------Learning_Rate: {lr_record}--------------------')
        
        file = open(file_name, "a")
        print('Epoch [{}/{}], (Train_Loss) MS-SSIM:{:.4f}, L1:{:.4f}, Total:{:.4f}   (Val_Value) MS-SSIM:{:.4f}, SSIM:{:.4f}, L1:{:.4f}, Time: {}h-{}m-{}s'.format(epoch+1, num_epochs, avg_msssim_loss, avg_l1_loss, average_loss, avg_msssim_val, avg_ssim_val, avg_l1_val, int(hours), int(minutes), int(seconds)))
        file.write('Epoch [{}/{}], (Train_Loss) MS-SSIM:{:.4f}, L1:{:.4f}, Total:{:.4f}   (Val_Value) MS-SSIM:{:.4f}, SSIM:{:.4f}, L1:{:.4f}, Time: {}h-{}m-{}s\n'.format(epoch+1, num_epochs, avg_msssim_loss, avg_l1_loss, average_loss, avg_msssim_val, avg_ssim_val, avg_l1_val, int(hours), int(minutes), int(seconds)))
        file.close()
