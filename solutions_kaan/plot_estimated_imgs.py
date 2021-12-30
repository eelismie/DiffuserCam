"""
This script will plot ground truth, diffuser and estimated image for different images and different methods together.

Make sure that you downloaded "our_dataset" folder from https://drive.google.com/drive/folders/1QFdVsVgst7bFaG6L7zbAO0bO5vJu-Mbh?usp=sharing

Put downloaded "our_dataset" inside "data" folder.

Make sure that cropped estimated images exist in "data/our_dataset/reconstructions" folder. If it doesn't exist, do the reconstructions, otherwise this script doesn't give correct plot.

Run the following command:
```bash
python solutions/plot_estimated_imgs.py
```
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import click

@click.command()
@click.option(
    "--recon",
    type=str,
    default="all",
    help="Which reconstructions to display.",
)
def plot_estimated_imgs(recon):
    origin_dir = "data/our_dataset/original_imgs"
    parent_dir = "data/our_dataset/reconstructions"
    assert os.path.isdir(parent_dir) == True
    reconst_alg_dirs = glob.glob(os.path.join(parent_dir, "*"))
    
    if recon == "all":
        nreconst_algs = len(reconst_alg_dirs)
    else:
        nreconst_algs = 1
    fig, axs = plt.subplots(nreconst_algs+1, 11, figsize=(11,nreconst_algs+1))
    
    for j in range(11):
        fname = "img" + str(j+1) + ".jpeg"
        img_dir = os.path.join(origin_dir, fname)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[0,j].imshow(img, aspect="equal")
        axs[0,j].set_xticks([])
        axs[0,j].set_yticks([])
    axs[0,0].set(ylabel="original")
    
    if recon == "all":
        for i, reconst_alg_dir in enumerate(reconst_alg_dirs):
            for j in range(11):
                fname = "img" + str(j+1) + "_final.png"
                img_dir = os.path.join(reconst_alg_dir, fname)
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axs[i+1,j].imshow(img)
                axs[i+1,j].set_xticks([])
                axs[i+1,j].set_yticks([])
            axs[i+1,0].set(ylabel=os.path.basename(reconst_alg_dir))
    else:
        reconst_alg_dir = os.path.join(parent_dir, recon)
        for j in range(11):
            fname = "img" + str(j+1) + "_final.png"
            img_dir = os.path.join(reconst_alg_dir, fname)
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[1,j].imshow(img, aspect = "equal")
            axs[1,j].set_xticks([])
            axs[1,j].set_yticks([])
        axs[1,0].set(ylabel=os.path.basename(reconst_alg_dir))
    
    fig.tight_layout()
    
    plt.show()
    

if __name__ == "__main__":
    plot_estimated_imgs()
