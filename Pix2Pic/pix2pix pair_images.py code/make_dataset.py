import os
import cv2
from skimage import img_as_ubyte, exposure

def scale(source, target):
    image = cv2.imread(source, -1)
    if image is None:
        print(f"Failed to load image {source}")
        return
    image = img_as_ubyte(exposure.rescale_intensity(image))
    image = cv2.equalizeHist(image)
    cv2.imwrite(target, image)

source_dir = "/home/dell/pytorch-CycleGAN-and-pix2pix/sync_data/_2021-08-06-11-23-45/rgb/img_left"
target_dir = "/home/dell/pytorch-CycleGAN-and-pix2pix/reduced_dataset/trainA"

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# List files in the source directory
files = sorted(os.listdir(source_dir))

# Iterate over the first 5000 files
for filename in files[:5000]:
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    scale(source_path, target_path)

print("Conversion complete.")