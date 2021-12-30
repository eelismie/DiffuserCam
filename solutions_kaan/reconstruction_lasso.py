"""
This script will load the PSF data and raw measurement and get the reconstruction by LASSO problem.

Make sure that you downloaded "our_dataset" folders from https://drive.google.com/drive/folders/1QFdVsVgst7bFaG6L7zbAO0bO5vJu-Mbh?usp=sharing

Put downloaded "our_dataset" inside "data" folder.

Run the following command:
```bash
python solutions/reconstruction_lasso.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/lasso/img1.png --l_factor 0.03 --n_iter 1000 --acc_thresh 0
```

To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/lasso/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/lasso/img1_final.png
```

You can see the estimated image in "data/our_dataset/reconstructions/lasso" folder.

"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_data
from PIL import Image
import numpy as np

from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.linop.conv import Convolve2D
from pycsou.opt.proxalgs import AcceleratedProximalGradientDescent as APGD
from diffcam.plot import plot_image

from utils import Convolve2DRGB

@click.command()
@click.option(
    "--psf_fp",
    type=str,
    help="File name for recorded PSF.",
)
@click.option(
    "--data_fp",
    type=str,
    help="File name for raw measurement data.",
)
@click.option(
    "--n_iter",
    type=int,
    default=500,
    help="Number of iterations.",
)
@click.option(
    "--downsample",
    type=float,
    default=4,
    help="Downsampling factor.",
)
@click.option(
    "--disp",
    default=50,
    type=int,
    help="How many iterations to wait for intermediate plot/results. Set to negative value for no intermediate plots.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Whether to flip image.",
)
@click.option(
    "--save",
    default=None,
    type=str,
    help="Whether to save intermediate and final reconstructions.",
)
@click.option(
    "--gray",
    is_flag=True,
    help="Whether to perform construction with grayscale.",
)
@click.option(
    "--bayer",
    is_flag=True,
    help="Whether image is raw bayer data.",
)
@click.option(
    "--no_plot",
    is_flag=True,
    help="Whether to no plot.",
)
@click.option(
    "--bg",
    type=float,
    help="Blue gain.",
)
@click.option(
    "--rg",
    type=float,
    help="Red gain.",
)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--single_psf",
    is_flag=True,
    help="Same PSF for all channels (sum) or unique PSF for RGB.",
)
@click.option(
    "--l_factor",
    default=0.01,
    type=float,
    help="Scaling factor for regularization parameter.",
)
@click.option(
    "--acc_thresh",
    default=1e-3,
    type=float,
    help="Accuracy threshold for proximal algorithm.",
)
def reconstruction_lasso(
    psf_fp,
    data_fp,
    n_iter,
    downsample,
    disp,
    flip,
    gray,
    bayer,
    bg,
    rg,
    gamma,
    save,
    no_plot,
    single_psf,
    l_factor,
    acc_thresh
):
    #print(data.shape)
    psf, data = load_data(
        psf_fp=psf_fp,
        data_fp=data_fp,
        downsample=downsample,
        bayer=bayer,
        blue_gain=bg,
        red_gain=rg,
        plot=not no_plot,
        flip=flip,
        gamma=gamma,
        gray=gray,
        single_psf=single_psf,
    )

    if disp < 0:
        disp = None

    start_time = time.time()

    H = Convolve2DRGB(data.size, psf) #assumes psf and data are same shape
    H.compute_lipschitz_cst()

    l22_loss = (1/2) * SquaredL2Loss(dim=H.shape[0], data=data.ravel())
    F = l22_loss * H
    #lambda_ = 0.01
    tmp = H.adjoint(data.flatten())
    lambda_ = l_factor * max(abs(tmp.max()), abs(tmp.min()))
    print("lamba factor: {}".format(l_factor))
    print("lambda value: {}".format(lambda_))
    G = lambda_ * L1Norm(dim=H.shape[1])
    apgd = APGD(dim=H.shape[1], F=F, G=G, acceleration="CD", verbose=10, max_iter=n_iter, accuracy_threshold=acc_thresh)
    
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    # TODO : apply your reconstruction
    estimate, converged, diagnostics = apgd.iterate()
    est = estimate['iterand'].reshape(data.shape)
    ax = plot_image(est, gamma=gamma)
    
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save is not None:
        est_norm = est/est.max()
        est_nclip = np.clip(est_norm, a_min=0, a_max=est_norm.max())
        img_data = (est_nclip * 255.0).astype(np.uint8)
        im = Image.fromarray(img_data)
        im.save(save)
        bname =os.path.basename(save).split(".")[0]
        bname = bname + ".npy"
        np.save(plib.Path(os.path.dirname(save)) / bname, est)
        print(f"Files saved to : {save}")
    


if __name__ == "__main__":
    reconstruction_lasso()
