from LadleNetPlus import LadleNet_plus as LadleNet
from LadleNetPlus import Dataset_creat, collate_fn
import os
import cv2
import numpy as np
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
from PIL import Image


if __name__ == '__main__':
    transform_pre = transforms.Compose([transforms.ToTensor()
                                    ,transforms.Resize((300,400))
                                    ,transforms.CenterCrop((192, 256))])

    model = LadleNet()
    model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_path = '/ocean/projects/cis220039p/ayanovic/vlr_project/LadleNet/Model/LadleNet+/weight/2024-04-11_08-37-04_LadleNet_plus.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("Loaded model")

    batch_size = 32
    num_epochs = 50
    learning_rate = 0.5
    save_step = int(num_epochs * 0.2)

    IR = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/testB'
    # VI = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainA'
    VI = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/testA'
    dataset = Dataset_creat(IR, VI, [transform_pre])

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    root_path = '/ocean/projects/cis220039p/ayanovic/vlr_project/LadleNet/Model/LadleNet+/result'
    output_path = os.path.join(root_path, 'outputs')
    gt_path = os.path.join(root_path, 'gt')
    input_path = os.path.join(root_path, 'inputs')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # Image visualization
    print("Visualizing images")
    for n, (ir_inputs, vi_inputs) in enumerate(val_loader):
        ir_inputs, vi_inputs = ir_inputs.to(device), vi_inputs.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(vi_inputs)

            # print(f"Outputs shape: {outputs.shape}")
            # print(f"Outputs data type: {outputs.dtype}")
            # print(f"Output sample: {outputs[0, :3, 50, 50]}")
            # print(f"IR shape: {ir_inputs.shape}")
            # print(f"IR data type: {ir_inputs.dtype}")
            # print(f"IR sample: {ir_inputs[0, :3, 50, 50]}")
            # print(f"VI inputs shape: {vi_inputs.shape}")
            # print(f"VI inputs data type: {vi_inputs.dtype}")
            # print(f"VI inputs sample: {vi_inputs[0, :3, 50, 50]}")

            ir_img = ir_inputs.cpu().numpy()
            output_img = outputs.cpu().numpy()
            vi_img = vi_inputs.cpu().numpy()

            ir_img = (ir_img * 255).astype(np.uint8)
            vi_img = (vi_img * 255).astype(np.uint8)
            output_img = (output_img * 255).astype(np.uint8)

            print(f"Saving {ir_img.shape[0]} images")
            for i in range(ir_img.shape[0]):
                print(f"Saving image {i*(n+1)}")
                ir_image = ir_img[i]
                vi_image = vi_img[i]
                output_image = output_img[i]

                ir_image = np.transpose(ir_image, (1, 2, 0))
                vi_image = np.transpose(vi_image, (1, 2, 0))
                output_image = np.transpose(output_image, (1, 2, 0))

                ir_image = cv2.cvtColor(ir_image, cv2.COLOR_RGB2BGR)
                vi_image = cv2.cvtColor(vi_image, cv2.COLOR_RGB2BGR)
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

                ir_image_path = os.path.join(gt_path, f"gt_{i*(n+1)}.png")
                vi_image_path = os.path.join(input_path, f"input_{i*(n+1)}.png")
                output_image_path = os.path.join(output_path, f"output_{i*(n+1)}.png")

                cv2.imwrite(ir_image_path, ir_image)
                cv2.imwrite(vi_image_path, vi_image)
                cv2.imwrite(output_image_path, output_image)
    print("Saved")
