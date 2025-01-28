from LadleNet import LadleNet, Dataset_creat, collate_fn
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

    model_path = '/home/anton/Desktop/SPRING24/VLR/project/LadleNet/Model/LadleNet/weight/2024-04-04_09-48-27_LadleNet.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("Loaded model")

    batch_size = 10  # 56 
    num_epochs = 50
    learning_rate = 0.5
    save_step = int(num_epochs * 0.2)

    IR = '/home/anton/Desktop/SPRING24/VLR/project/Datasets/images_thermal_val/data'
    # VI = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainA'
    VI = '/home/anton/Desktop/SPRING24/VLR/project/Datasets/images_rgb_val/data'
    dataset = Dataset_creat(IR, VI, [transform_pre])

    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # Image visualization

    print("Visualizing images")
    sample_images = []
    image_count = 0
    for ir_inputs, vi_inputs in val_loader:
        ir_inputs, vi_inputs = ir_inputs.to(device), vi_inputs.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(vi_inputs)

        print(f"Outputs shape: {outputs.shape}")
        print(f"Outputs data type: {outputs.dtype}")
        print(f"Output sample: {outputs[0, :3, 50, 50]}")
        print(f"IR shape: {ir_inputs.shape}")
        print(f"IR data type: {ir_inputs.dtype}")
        print(f"IR sample: {ir_inputs[0, :3, 50, 50]}")
        print(f"VI inputs shape: {vi_inputs.shape}")
        print(f"VI inputs data type: {vi_inputs.dtype}")
        print(f"VI inputs sample: {vi_inputs[0, :3, 50, 50]}")
        

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
        ir_img = (ir_img * 255).astype(np.uint8)
        vi_img = (vi_img * 255).astype(np.uint8)
        output_img = (output_img * 255).astype(np.uint8)

        processed_images.append(np.concatenate((ir_img, output_img, vi_img), axis=2))

    grid = vutils.make_grid(torch.tensor(np.stack(processed_images)), nrow=1, padding=2, normalize=False, scale_each=False)
    # Convert to PIL image and save
    grid_np = grid.numpy()
    grid_np = np.transpose(grid_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    grid_image = Image.fromarray(grid_np)
    grid_image.save("test_output_input_grid.png")

    print("Saved 'test_output_input_grid.png'")  # Only process one batch for this test
