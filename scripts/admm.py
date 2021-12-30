"""
Apply ADMM reconstruction.

```
python scripts/admm.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --n_iter 6 --save data/our_dataset/reconstructions/admm/img1.png
```

"""

import os
import time
import pathlib as plib
import click
import matplotlib.pyplot as plt
from datetime import datetime
from diffcam.io import load_data
from diffcam.admm import ADMM
from PIL import Image
import numpy as np

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
    default=5,
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
    default=1,
    type=int,
    help="How many iterations to wait for intermediate plot. Set to negative value for no intermediate plots.",
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
def admm(
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

    start_time = time.time()
    recon = ADMM(psf)
    recon.set_data(data)
    print(f"setup time : {time.time() - start_time} s")

    start_time = time.time()
    recon.apply(n_iter=n_iter, disp_iter=disp, save=None, gamma=gamma, plot=not no_plot)
    print(f"proc time : {time.time() - start_time} s")

    if not no_plot:
        plt.show()
    if save is not None:
        est = recon.get_image_est()
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
    admm()
