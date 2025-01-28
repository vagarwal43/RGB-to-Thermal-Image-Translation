# Created by Baole Fang at 4/4/24

import os
import glob
import cv2
from skimage import exposure, img_as_ubyte
from tqdm import tqdm
import random
import multiprocessing as mp


def scale(source, target):
    image = cv2.imread(source, -1)
    image = img_as_ubyte(exposure.rescale_intensity(image))
    image = cv2.equalizeHist(image)
    cv2.imwrite(target, image)


def helper(arg):
    rgb, thr, i = arg
    filename = str(i).zfill(6)+'.png'
    try:
        scale(thr, os.path.join(target_root, 'thr', filename))
        os.symlink(rgb, os.path.join(target_root, 'rgb', filename))
        return filename
    except:
        print(f'{i} fails')
        return None




if __name__ == '__main__':
    root = '/data/baole/ms2/sync_data/*'
    target_root = '/data/baole/ms2_paired'

    rgbs = sorted(glob.glob(os.path.join(root, 'rgb/*/*.png')))
    thrs = sorted(glob.glob(os.path.join(root, 'thr/*/*.png')))

    assert len(rgbs) == len(thrs)
    n=len(rgbs)

    os.makedirs(os.path.join(target_root, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'thr'), exist_ok=True)

    i=0
    files=[]
    args= [(rgbs[i], thrs[i], i) for i in range(n)]
    with mp.Pool(os.cpu_count()) as p:
        files = list(tqdm(p.imap(helper, args), total=n))


    # for rgb, thr in tqdm(zip(rgbs, thrs), total=len(rgbs)):
    #     filename = str(i).zfill(6)+'.png'
    #     try:
    #         scale(thr, os.path.join(target_root, 'thr', filename))
    #         os.symlink(rgb, os.path.join(target_root, 'rgb', filename))
    #         i+=1
    #         files.append(filename)
    #     except:
    #         print(f'{i} fails')
    files = [f for f in files if f is not None]
    random.shuffle(files)
    with open(os.path.join(target_root,'train.flist'), 'w') as f:
        for item in files[:int(i*0.8)]:
            print(item, file=f)
    with open(os.path.join(target_root,'val.flist'), 'w') as f:
        for item in files[int(i*0.8):]:
            print(item, file=f)
