import argparse
import os.path

import numpy as np
import glob
import skimage
from tqdm import tqdm
import skimage.transform
from skimage import color


def eval(root):
    gt_folder = os.path.join(root, 'ground_truth')
    pred_folder = os.path.join(root, 'pred')
    print("Predicted folder contents:", os.listdir(pred_folder))

    gts = sorted(glob.glob(gt_folder + '/*.png'))
    print(len(gts))
    preds = sorted(glob.glob(pred_folder + '/*.png'))
    print(len(preds))
    assert len(gts) == len(preds)
    n = len(gts)
    mses = 0
    maes = 0
    ssims = 0
    psnrs = 0
    for gt, pred in tqdm(zip(gts, preds), total=n):
        try:
            gt_image = skimage.io.imread(gt, as_gray=True)  # Load as grayscale
            pred_image = skimage.io.imread(pred)
            pred_image_gray = color.rgb2gray(pred_image)
            gt_image_resized = skimage.transform.resize(gt_image, pred_image.shape[:2], preserve_range=True)
            mse, mae, ssim, psnr = compute_metrics(gt_image_resized, pred_image_gray)
        except Exception as e:
            print(f"Error processing images: {e}")
            n -= 1
            continue
        finally:
            mses += mse
            maes += mae
            ssims += ssim
            psnrs += psnr
    mses /= n
    maes /= n
    ssims /= n
    psnrs /= n
    return mses, maes, ssims, psnrs


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
    mse = np.mean((gt - pred) ** 2)
    mae = np.mean(np.abs(gt - pred))
    ssim = skimage.metrics.structural_similarity(gt, pred, data_range=1)  # Adjusted data_range for grayscale images
    psnr = 10 * np.log10((1 ** 2) / mse)  # Adjusted for grayscale images
    return mse, mae, ssim, psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Images directory')

    ''' parser configs '''
    args = parser.parse_args()

    mses, maes, ssims, psnrs = eval(os.path.join(args.path, 'results/test'))

    print('MSE: {}'.format(mses))
    print('MAE: {}'.format(maes))
    print('SSIM: {}'.format(ssims))
    print('PSNR: {}'.format(psnrs))

    with open(os.path.join(args.path, 'metrics.txt'), 'w') as f:
        print('MSE: {}'.format(mses), file=f)
        print('MAE: {}'.format(maes), file=f)
        print('SSIM: {}'.format(ssims), file=f)
        print('PSNR: {}'.format(psnrs), file=f)
