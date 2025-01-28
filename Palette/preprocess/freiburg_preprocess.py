# Created by Baole Fang at 3/3/24

import os
import glob
import cv2
from skimage import exposure, img_as_ubyte
from tqdm import tqdm
import random

def scale(source, target):
    image = cv2.imread(source, -1)
    image = img_as_ubyte(exposure.rescale_intensity(image))
    image = cv2.equalizeHist(image)
    cv2.imwrite(target, image)


def train():
    source_root = '/ocean/projects/cis220039p/bfang1/data/freiburg/train'
    target_root = '/ocean/projects/cis220039p/bfang1/data/freiburg_paired/train'
    # source_root = '/data/baole/freiburg/train'
    # target_root = '/data/baole/freiburg_paired/train'
    A='fl_ir_aligned'
    B='fl_rgb'
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(os.path.join(target_root,A), exist_ok=True)
    os.makedirs(os.path.join(target_root,B), exist_ok=True)

    idx=0
    files_A=sorted(glob.glob(source_root+f'/*/*/{A}/*'))
    files_B=sorted(glob.glob(source_root+f'/*/*/{B}/*'))
    train_list=[]

    for file_A, file_B in tqdm(zip(files_A, files_B), total=len(files_A)):
        filename=str(idx).zfill(6)+'.png'
        try:
            scale(file_A, os.path.join(target_root, A, filename))
            os.symlink(file_B,os.path.join(target_root, B, filename))
            train_list.append(filename)
            idx+=1
        except:
            print(f'{idx} fails')

    random.shuffle(train_list)
    with open(os.path.join(target_root,'train.flist'), 'w') as f:
        for item in train_list[:int(idx*0.8)]:
            print(item, file=f)
    with open(os.path.join(target_root,'val.flist'), 'w') as f:
        for item in train_list[int(idx*0.8):]:
            print(item, file=f)
    with open(os.path.join(target_root,'all.flist'), 'w') as f:
        for item in train_list:
            print(item, file=f)


def test():
    source_root = '/ocean/projects/cis220039p/bfang1/data/freiburg/test'
    target_root = '/ocean/projects/cis220039p/bfang1/data/freiburg_paired/test'
    # source_root = '/data/baole/freiburg/test'
    # target_root = '/data/baole/freiburg_paired/test'
    A='ImagesIR'
    B='ImagesRGB'
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(os.path.join(target_root,A), exist_ok=True)
    os.makedirs(os.path.join(target_root,B), exist_ok=True)

    idx=0
    files_A=sorted(glob.glob(source_root+f'/*/{A}/*'))
    files_B=sorted(glob.glob(source_root+f'/*/{B}/*'))
    train_list=[]

    for file_A, file_B in tqdm(zip(files_A, files_B), total=len(files_A)):
        filename=str(idx).zfill(6)+'.png'
        scale(file_A, os.path.join(target_root, A, filename))
        os.symlink(file_B,os.path.join(target_root, B, filename))
        train_list.append(filename)
        idx+=1

    with open(os.path.join(target_root,'test.flist'), 'w') as f:
        for item in train_list:
            print(item, file=f)

if __name__ == '__main__':
    # train()
    # test()
    with open('/data/baole/freiburg_paired/train/all.flist', 'w') as f:
        for i in range(19369+1):
            print(str(i).zfill(6)+'.png', file=f)
