import glob
import os
import pathlib as plib
import click
from datetime import datetime
from diffcam.io import load_psf
import numpy as np
from diffcam.util import print_image_info
from PIL import Image
from diffcam.mirflickr import ADMM_MIRFLICKR, postprocess
from diffcam.metric import mse, psnr, ssim, lpips
import matplotlib.pyplot as plt
from diffcam.plot import plot_image

from utils import Convolve2DRGB, HuberNorm
from pycsou.linop.base import BlockDiagonalOperator
from pycsou.linop.diff import Gradient
from pycsou.func.loss import SquaredL2Loss
from pycsou.core.functional import DifferentiableFunctional
from pycsou.func.penalty import NonNegativeOrthant, L1Norm
from pycsou.linop.conv import Convolve2D
from pycsou.opt.proxalgs import PDS, APGD

"""
Evaluation of algorithm from section 6 in project, on a subset of Flickr images. 

```
python solutions/evaluate_nnTV.py \
--data data \
--n_files 3 --save
```

"""

@click.command()
@click.option(
    "--data",
    type=str,
    help="Dataset to work on.",
)
@click.option(
    "--n_files",
    type=int,
    default=None,
    help="Number of files to apply reconstruction on. Default is apply on all.",
)
@click.option(
    "--n_iter",
    type=int,
    default=100,
    help="Number of iterations.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Whether to take PSF as sum of RGB channels.",
)
@click.option(
    "--save",
    is_flag=True,
    help="Whether to save reconstructions.",
)

def mirflickr_dataset(data, n_files, n_iter, single_psf, save):
    assert data is not None

    dataset_dir = os.path.join(data, "dataset")
    if os.path.isdir(dataset_dir):
        diffuser_dir = os.path.join(dataset_dir, "diffuser_images")
        lensed_dir = os.path.join(dataset_dir, "ground_truth_lensed")
    else:
        diffuser_dir = os.path.join(data, "diffuser")
        lensed_dir = os.path.join(data, "lensed")
    psf_fp = os.path.join(data, "psf.tiff")
    downsample = 4  # has to be this for collected data!

    # determine files
    files = glob.glob(diffuser_dir + "/*.npy")
    print(files)
    print(n_files)
    if n_files:
        files = files[:n_files]
    files = [os.path.basename(fn) for fn in files]
    print("Number of files : ", len(files))

    # -- prepare PSF
    print("\nPrepared PSF data")
    psf_float, background = load_psf(
        psf_fp,
        downsample=downsample,
        return_float=True,
        return_bg=True,
        bg_pix=(0, 15),
        single_psf=single_psf,
    )
    print_image_info(psf_float)

    if save:
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "lasso_mirflickr" + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    H = Convolve2DRGB(psf_float.size, psf_float) #assumes psf and data are same shape
    H.compute_lipschitz_cst()

    print("\nLooping through files...")
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    for fn in files:

        bn = os.path.basename(fn).split(".")[0]
        print(f"\n{bn}")

        # load diffuser data
        lensless_fp = os.path.join(diffuser_dir, fn)
        diffuser = np.load(lensless_fp)
        diffuser_prep = diffuser - background
        diffuser_prep = np.clip(diffuser_prep, a_min=0, a_max=1)
        diffuser_prep /= np.linalg.norm(diffuser_prep.ravel())

        l22_loss = (1/2) * SquaredL2Loss(dim=H.shape[0], data=diffuser_prep.ravel())
        F = l22_loss * H
        l_factor = 0.001

        tmp = H.adjoint(diffuser_prep.flatten())
        lambda_ = l_factor * max(abs(tmp.max()), abs(tmp.min()))
        print("lamba factor: {}".format(l_factor))
        print("lambda value: {}".format(lambda_))
        G = lambda_ * L1Norm(dim=H.shape[1])
        apgd = APGD(dim=H.shape[1], F=F, G=G, acceleration="CD", verbose=10, max_iter=n_iter, accuracy_threshold=5e-4)
        estimate, converged, diagnostics = apgd.iterate()

        est = estimate['iterand'].reshape(diffuser_prep.shape)
        est[est < 0] = 0

        if save:
            np.save(os.path.join(save, f"{bn}.npy"), est)
            output_fn = os.path.join(save, f"{bn}.tif")
            im = postprocess(np.array((est) / (np.max(est))))
            image_data = (im * 255).astype(np.uint8)
            im = Image.fromarray(image_data)
            im.save(output_fn)

        est = postprocess(np.array((est) / (np.max(est))))

        lensed_fp = os.path.join(lensed_dir, fn)
        lensed = np.load(lensed_fp)
        lensed = postprocess(lensed)

        mse_scores.append(mse(lensed, est))
        psnr_scores.append(psnr(lensed, est))
        ssim_scores.append(ssim(lensed, est))
        lpips_scores.append(lpips(lensed, est))

    if save:
        print(f"\nReconstructions saved to : {save}")

    print("\nMSE (avg)", np.mean(mse_scores))
    print("PSNR (avg)", np.mean(psnr_scores))
    print("SSIM (avg)", np.mean(ssim_scores))
    print("LPIPS (avg)", np.mean(lpips_scores))

if __name__ == "__main__":
    mirflickr_dataset()

