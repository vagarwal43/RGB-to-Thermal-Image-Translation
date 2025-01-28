# Created by Baole Fang at 4/1/24

import argparse
import os.path

import numpy as np
import glob
import skimage
from tqdm import tqdm


def eval(root):
    gts=sorted(glob.glob(root+'/GT_*.png'))
    preds=sorted(glob.glob(root+'/Out_*.png'))
    assert len(gts)==len(preds)
    n=len(gts)
    mses=0
    maes=0
    ssims=0
    psnrs=0
    for gt, pred in tqdm(zip(gts, preds), total=n):
        try:
            gt = skimage.io.imread(gt) / 255
            pred = skimage.io.imread(pred) / 255
            mse, mae, ssim, psnr = compute_metrics(gt, pred)
        except:
            n-=1
            continue
        finally:
            mses+=mse
            maes+=mae
            ssims+=ssim
            psnrs+=psnr
    mses/=n
    maes/=n
    ssims/=n
    psnrs/=n
    return mses, maes, ssims, psnrs


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    mse = np.mean((gt - pred) ** 2)
    mae = np.mean(np.abs(gt - pred))
    ssim = skimage.metrics.structural_similarity(gt, pred, data_range=1)
    psnr = 10 * np.log10((1 ** 2) / mse)
    return mse, mae, ssim, psnr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Images directory')

    ''' parser configs '''
    args = parser.parse_args()

    mses, maes, ssims, psnrs = eval(os.path.join(args.path, 'results/test/0'))

    print('MSE: {}'.format(mses))
    print('MAE: {}'.format(maes))
    print('SSIM: {}'.format(ssims))
    print('PSNR: {}'.format(psnrs))

    with open(os.path.join(args.path, 'metrics.txt'), 'w') as f:
        print('MSE: {}'.format(mses), file=f)
        print('MAE: {}'.format(maes), file=f)
        print('SSIM: {}'.format(ssims), file=f)
        print('PSNR: {}'.format(psnrs), file=f)

