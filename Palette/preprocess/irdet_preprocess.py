# Created by Baole Fang at 3/28/24

import os
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    root = 'datasets/ir_det_dataset'
    files = os.listdir(os.path.join(root, 'Images', 'rgb'))
    train, test = train_test_split(files, test_size=0.2, random_state=42)
    flist = os.path.join(root, 'flist')
    os.makedirs(flist, exist_ok=True)
    with open(os.path.join(flist, 'train.flist'), 'w') as f:
        for item in train:
            print(item, file=f)
    with open(os.path.join(flist, 'test.flist'), 'w') as f:
        for item in test:
            print(item, file=f)
