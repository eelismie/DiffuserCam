"""

Example usage:
```
python scripts/compute_metrics_from_original.py \
--recon data/reconstruction/admm_thumbs_up_rgb.npy \
--original data/original/thumbs_up.png \
--vertical_crop 262 371 \
--horizontal_crop 438 527 \
--rotation -0.5 \
--save_proc_est data/reconstruction/admm_thumbs_up_rgb_final.png
```
you can get examples files from SWITCHDrive: https://drive.switch.ch/index.php/s/NdgHlcDeHVDH5ww

"""

import numpy as np
import click
import matplotlib.pyplot as plt
from diffcam.plot import plot_image
from diffcam.io import load_image
from diffcam.util import resize
from scipy.ndimage import rotate
from diffcam.metric import mse, psnr, ssim, lpips
from PIL import Image

@click.command()
@click.option(
    "--recon",
    type=str,
    help="File path to reconstruction.",
)
@click.option(
    "--original",
    type=str,
    help="File path to original file.",
)
@click.option(
    "--vertical_crop",
    type=(int, int),
    help="Cropping for vertical dimension.",
)
@click.option(
    "--horizontal_crop",
    type=(int, int),
    help="Cropping for horizontal dimension.",
)
@click.option(
    "--rotation",
    type=float,
    help="Degrees to rotate reconstruction.",
)
@click.option("-v", "--verbose", count=True)
@click.option(
    "--save_proc_est",
    default=None,
    type=str,
    help="Whether to save cropped and rotated reconstructions.",
)
def compute_metrics(recon, original, vertical_crop, horizontal_crop, rotation, verbose, save_proc_est):

    # load estimate
    est = np.load(recon)
    if verbose:
        print("estimate shape", est.shape)

    # crop
    est = rotate(
        est[vertical_crop[0] : vertical_crop[1], horizontal_crop[0] : horizontal_crop[1]],
        angle=rotation,
    )
    est /= est.max()
    est = np.clip(est, 0, 1)
    
    if verbose:
        print("estimate cropped: ")
        print(est.shape)
        print(est.dtype)
        print(est.max())
        
    if save_proc_est is not None:
        img_data = (est * 255.0).astype(np.uint8)
        im = Image.fromarray(img_data)
        im.save(save_proc_est)
        
    plot_image(est)

    # load real image
    img = load_image(original)
    img = img / img.max()

    factor = est.shape[1] / img.shape[1]
    if verbose:
        print("resize factor", factor)
    img_resize = np.zeros_like(est)
    tmp = resize(img, factor=factor).astype(est.dtype)
    img_resize[: min(est.shape[0], tmp.shape[0]), : min(est.shape[1], tmp.shape[1])] = tmp[
        : min(est.shape[0], tmp.shape[0]), : min(est.shape[1], tmp.shape[1])
    ]
    if verbose:
        print("\noriginal resized: ")
        print(img_resize.shape)
        print(img_resize.dtype)
        print(img_resize.max())

    plot_image(img_resize)

    print("\nMSE", mse(img_resize, est))
    print("PSNR", psnr(img_resize, est))
    print("SSIM", ssim(img_resize, est))
    #print("LPIPS", lpips(img_resize, est))

    plt.show()


if __name__ == "__main__":
    compute_metrics()
