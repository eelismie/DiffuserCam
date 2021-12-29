"""
This script will load the PSF data and raw measurement for the reconstruction
that can implement afterwards.

```bash
python scripts/reconstruction_template.py --psf_fp data/psf/diffcam_rgb.png \
--data_fp data/raw_data/thumbs_up_rgb.png
```

"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_data

from pycsou.func.loss import SquaredL2Loss
from pycsou.core.functional import DifferentiableFunctional
from pycsou.func.penalty import NonNegativeOrthant, L1Norm
from pycsou.linop.conv import Convolve2D
from pycsou.opt.proxalgs import PDS, APGD
from pycsou.linop.diff import Gradient
from diffcam.plot import plot_image
import numpy as np

from utils import Convolve2DRGB, PDS_, APGD_
from pycsou.linop.base import BlockDiagonalOperator

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
    default=200,
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
    default=10,
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
    is_flag=True,
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
    "--delta",
    default=1.0,
    type=float,
    help="Threshold value for Huber norm.",
)
def reconstruction(
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
        delta
):
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
    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "YOUR_RECONSTRUCTION_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()
    # TODO : setup for your reconstruction algorithm

    is_rbg = len(data.shape) == 3

    if is_rbg:
        shape_ = (data.shape[0], data.shape[1])
        grad = BlockDiagonalOperator(Gradient(shape_), Gradient(shape_), Gradient(shape_))
        grad.compute_lipschitz_cst()
    else: 
        grad = Gradient(data.shape)
        grad.compute_lipschitz_cst()

    H = Convolve2DRGB(data.size, psf) #assumes psf and data are same shape
    H.compute_lipschitz_cst()

    l22_loss = (1 / 2) * SquaredL2Loss(dim=H.shape[0], data=data.ravel())
    F = l22_loss * H
    G = NonNegativeOrthant(dim=H.shape[1])
    tmp = H.adjoint(data.flatten())
    lambda_ = l_factor * max(abs(tmp.max()), abs(tmp.min()))
    print("lamba factor: {}".format(l_factor))
    print("lambda value: {}".format(lambda_))
    h = HuberNorm(dim=grad.shape[0], delta=delta)
    F += lambda_ * h * grad

    apgd = APGD_(dim=H.shape[1], F=F, G=G, acceleration="CD", verbose=disp, max_iter=n_iter, accuracy_threshold=1e-4, gamma=gamma, datashape=data.shape, no_plot=no_plot, save=save)

    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    # TODO : apply your reconstruction
    estimate, converged, diagnostics = apgd.iterate()
    ax = plot_image(estimate['iterand'].reshape(data.shape), gamma=gamma)
    ax.set_title("Reconstructed Data Img1 - NNTV")
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save:
        plt.savefig(plib.Path(save) / f"{n_iter}_final.png")
        print(f"Files saved to : {save}")

class HuberNorm(DifferentiableFunctional):
    def __init__(self, dim: int, delta: float):
        self.delta = delta
        super(HuberNorm, self).__init__(dim=dim, diff_lipschitz_cst=1.)

    def __call__(self, x):
        center = np.abs(x) <= self.delta
        x[center] = .5 * x[center]**2
        x[~ center] = self.delta * (np.abs(x[~ center]) - self.delta/2)
        return x.sum()

    def jacobianT(self, x):
        idx = np.abs(x) > self.delta
        x[idx] = self.delta * np.sign(x[idx])
        return x

if __name__ == "__main__":
    reconstruction()
