import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import glob

from tqdm import tqdm


def convert_abl(ab, l):
    """ convert AB and L to RGB """
    l = np.expand_dims(l, axis=2)
    lab = np.concatenate([l, ab], axis=2)
    if len(lab.shape) == 4:
        image_color, image_l = [], []
        for _color, _l in zip(lab, l):
            out = cv2.cvtColor(_color.astype('uint8'), cv2.COLOR_LAB2RGB)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            image_color.append(out)
            image_l.append(cv2.cvtColor(_l.astype('uint8'), cv2.COLOR_GRAY2RGB))
        image_color = np.array(image_color)
        image_l = np.array(image_l)
    else:
        image_color = cv2.cvtColor(lab.astype('uint8'), cv2.COLOR_LAB2RGB)
        image_l = cv2.cvtColor(l.astype('uint8'), cv2.COLOR_GRAY2RGB)
    return image_color, image_l


if __name__ == '__main__':
    source_root = '/data/baole/mirflickr'
    target_root = '/data/baole/mirflickr25k_paired'

    color_save_path, gray_save_path = '{}/color'.format(target_root), '{}/gray'.format(target_root)
    os.makedirs(color_save_path, exist_ok=True)
    os.makedirs(gray_save_path, exist_ok=True)
    i = 0
    files = glob.glob(source_root + '/*.jpg')
    for filename in tqdm(files):
        image = cv2.imread(filename, -1)
        image_gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        cv2.imwrite('{}/{}.png'.format(color_save_path, str(i).zfill(5)), image)
        cv2.imwrite('{}/{}.png'.format(gray_save_path, str(i).zfill(5)), image_gray)
        i += 1

    flist_save_path = os.path.join(target_root, 'flist')
    os.makedirs(flist_save_path, exist_ok=True)
    arr = np.random.permutation(25000)
    with open('{}/train.flist'.format(flist_save_path), 'w') as f:
        for item in arr[:24000]:
            print(str(item).zfill(5), file=f)
    with open('{}/test.flist'.format(flist_save_path), 'w') as f:
        for item in arr[24000:]:
            print(str(item).zfill(5), file=f)
