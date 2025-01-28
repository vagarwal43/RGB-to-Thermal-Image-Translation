import cv2
import numpy as np
import sys

def freibeg_crop_ir(img):
    # center crop
    # image shape is (650, 1920)
    _, orig_width = img.shape
    crop_width = 1300
    start_x = orig_width // 2 - crop_width // 2
    img = img[:, start_x:(start_x+crop_width)]
    
    # last check if any pixel vals are 0
    if np.any(img == 0):
        print("Found 0 pixel values in IR image")
        sys.exit(1)
    return img

def freibeg_crop_rgb(img):
    # center crop
    # image shape is (650, 1920, 3)
    # print(f"Image shape inside: {img.shape}")
    _, orig_width, _ = img.shape
    crop_width = 1300
    start_x = orig_width // 2 - crop_width // 2
    img = img[:, start_x:start_x+crop_width, :]
    return img

def normalize(img):
    # Load the 16-bit thermal image
    minmax_img = cv2.normalize(
        img, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return minmax_img
