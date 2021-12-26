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

from pycsou.core import LinearOperator
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.linop.conv import Convolve2D
from pycsou.opt.proxalgs import AcceleratedProximalGradientDescent as APGD
from diffcam.plot import plot_image

from utils import Convolve2DRGB, IDCT_2D

import numpy as np
from scipy.fft import dctn, idctn


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
    default=300,
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
    default=0.000001,
    type=float,
    help="Scaling factor for regularization parameter.",
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
    l_factor
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
    if save:
        save = os.path.basename(data_fp).split(".")[0]
        timestamp = datetime.now().strftime("_%d%m%d%Y_%Hh%M")
        save = "YOUR_RECONSTRUCTION_" + save + timestamp
        save = plib.Path(__file__).parent / save
        save.mkdir(exist_ok=False)

    start_time = time.time()

    if gray:
        H2 = Convolve2D(size=data.size, filter=psf, shape=(data.shape[0], data.shape[1]))
        H2.compute_lipschitz_cst()
        lips = H2.lipschitz_cst
    else: 
        #TODO: find a tighter upper bound to the operator norm of RGB operator
        n1, n2 = data.shape[:2]
        imsize = n1 * n2
        listH = [Convolve2D(size=imsize, filter=psf[:,:,i], shape=(n1, n2)) for i in range(3)]
        for H in listH:
            H.compute_lipschitz_cst()
        lips = sum([i.lipschitz_cst for i in listH])

    H = Convolve2DRGB(data.size, psf, lips) #assumes psf and data are same shape

    n1 = data.shape[0]
    n2 = data.shape[1]
    obj_idct2d = IDCT_2D(size=data.size, n1=n1, n2=n2)

    l22_loss = (1/2) * SquaredL2Loss(dim=H.shape[0], data=data.ravel())
    F = l22_loss * H
    tmp = H.adjoint(data.flatten())
    lambda_ = l_factor * max(abs(tmp.max()), abs(tmp.min()))
    print("lamba factor: {}".format(l_factor))
    print("lambda value: {}".format(lambda_))
    #lambda_ = 0.001
    G = lambda_ * L1Norm(dim=H.shape[1])
    apgd = APGD(dim=H.shape[1], F=F, G=G, acceleration = "CD", verbose=10, max_iter=n_iter, accuracy_threshold=1e-3)
    
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    estimate, converged, diagnostics = apgd.iterate()
    
    ax = plot_image(obj_idct2d(estimate['iterand']).reshape(data.shape), gamma=gamma)
    
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save:
        plt.savefig(plib.Path(save) / f"{n_iter}.png")
        print(f"Files saved to : {save}")

if __name__ == "__main__":
    reconstruction()



