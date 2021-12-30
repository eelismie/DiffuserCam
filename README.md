# DiffuserCam Project Submission COM-514

Final project submission for mathematical foundations of signal processing (COM-514) 2021. 

Authors:
- Eelis Mielonen 
- Adrian Jarret 
- Mehmet Onurcan Kaya 
- Kaan Okumus

## Instructions 

All solutions are found in the solution folder. The problems 1-8 have their dedicated files in the folder, which are to be run individually. Before running reconstructions, download our raw images, psf, and and reconstructions from the google drive (link and placement instructions in the reconstruction.py files). A small 3 image subset of the MRI-FLICKR dataset is included in the repo for the benchmark. To run the benchmark on our solution, run the eval_{}.py files. Running and plotting commands are included at the top of each file.

### Examples

Running reconstruction: 

```console 
python solutions/reconstruction_ridge.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/ridge/img1.png --l_factor 0.02 --n_iter 1000 --acc_thresh 0
```

Evaluating reconstruction (note that the crops need to be tuned by hand): 

```console 
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/ridge/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/ridge/img1_final.png
```

Plotting all reconstructions (once they've been made / downloaded from google drive):

```console
python solutions/plot_estimated_imgs.py
```

### Running the Codes

Make sure that you downloaded "our_dataset" folder from https://drive.google.com/drive/folders/1QFdVsVgst7bFaG6L7zbAO0bO5vJu-Mbh?usp=sharing
Put downloaded "our_dataset" inside "data" folder.

#### Ridge:
Run the following command:
```bash
python solutions/reconstruction_ridge.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/ridge/img1.png --l_factor 0.02 --n_iter 1000 --acc_thresh 0
```
To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/ridge/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/ridge/img1_final.png
```
You can see the estimated image in "data/our_dataset/reconstructions/ridge" folder.

#### LASSO:
Run the following command:
```bash
python solutions/reconstruction_lasso.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/lasso/img1.png --l_factor 0.03 --n_iter 1000 --acc_thresh 0
```
To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/lasso/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/lasso/img1_final.png
```
You can see the estimated image in "data/our_dataset/reconstructions/lasso" folder.

#### NNLS:
Run the following command:
```bash
python solutions/reconstruction_nnls.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/nnls/img1.png --acc_thresh 0.0007
```
To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/nnls/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/nnls/img1_final.png
```
You can see the estimated image in "data/our_dataset/reconstructions/nnls" folder.

#### DCT:
Run the following command:
```bash
python solutions/reconstruction_genlass_dct.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/genlasso_dct/img1.png --n_iter 1000 --acc_thresh 0 --l_factor 0.0000001
```
To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/genlasso_dct/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/genlasso_dct/img1_final.png
```
You can see the estimated image in "data/our_dataset/reconstructions/genlasso_dct" folder.

#### NNTV:
Run the following command:
```bash
python solutions/reconstruction_nnTV.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/nnTV/img1.png --l_factor 0.02 --n_iter 2000 --acc_thresh 0
```
To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/nnTV/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/nnTV/img1_final.png
```
You can see the estimated image in "data/our_dataset/reconstructions/nnTV" folder.

#### Huber:
Run the following command:
```bash
python solutions/reconstruction_Huber.py --psf_fp data/our_dataset/psf/psf.png --data_fp data/our_dataset/diffuser_imgs/img1.png --save data/our_dataset/reconstructions/huber/img1.png --l_factor 0.03 --delta 1.0 --acc_thresh 0 --n_iter 1000
```
To see the metrics and get the cropped image, run the following command (note that vertical_crop and horizontal_crop values should be changed for different images):
```bash
python scripts/compute_metrics_from_original.py --recon data/our_dataset/reconstructions/huber/img1.npy --original data/our_dataset/original_imgs/img1.jpeg --vertical_crop 183 411 --horizontal_crop 343 656 --rotation 0.025 --save_proc_est data/our_dataset/reconstructions/huber/img1_final.png
```
You can see the estimated image in "data/our_dataset/reconstructions/huber" folder.


## References
<a id="1">[1]</a> 
Antipa, N., Kuo, G., Heckel, R., Mildenhall, B., Bostan, E., Ng, R., & Waller, L. (2018). DiffuserCam: lensless single-exposure 3D imaging. Optica, 5(1), 1-9.

<a id="2">[2]</a> 
Monakhova, K., Yurtsever, J., Kuo, G., Antipa, N., Yanny, K., & Waller, L. (2019). Learned reconstructions for practical mask-based lensless imaging. Optics express, 27(20), 28075-28090.

<a id="3">[3]</a> 
Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc.
