# Traditional communication methods used as baselines for multi-user semantic communication

## What's New?

### 🔥 JPEG-1/2LDPC-BPSK-NOMA ✨:

The conventional approach integrates JPEG image compression technique with 1/2 LDPC channel coding and BPSK digital modulation. Futhermore, it adopts a classical NOMA scheme based on superposition coding (SC) and successive interference cancellation (SIC), which is refered to as JPEG-LDPC-BPSK-NOMA.

### 🔥 JPEG-1/2LDPC-16QAM-CDMA ✨:

The OMA scheme similarly adopts the JPEG technique combined with 1/2 LDPC channel code and 16QAM digital modulation. In addition, it employs orthogonal spreading codes for CDMA-based communication, which is referred to as JPEG-LDPC-16QAM-CDMA.

### 🔥 Mertic ✨:

The comparison schemes are evaluated using PSNR, MS-SSIM, LPIPS.

PSNR measures the fidelity of the reconstructed image, with higher values indicating reduced distortion. 

MS-SSIM evaluates perceptual similarity by considering luminance, contrast, and structural information across multiple scales.

LPIPS is employed to evaluate the perceptual similarity of the reconstructed images.

## Installation

1. Install `torch>=2.0.0`.
2. Install other pip packages via `pip install -r requirements.txt`.
3. Prepare the dataset
    <details>
    <summary> assume the AFHQ dataset is in `./dataset/AFHQ`. It should be like this:</summary> 

    ```
    ./dataset/AFHQ:
        /train:
            /cat
                flickr_cat_000002.jpg 
                flickr_cat_000003.jpg
                ...
            /dog
                flickr_dog_000002.jpg 
                flickr_dog_000003.jpg
                ...
            /wild
                flickr_wild_000002.jpg 
                flickr_wild_000003.jpg
                ...
        /val:
            /cat
                flickr_cat_000008.jpg 
                flickr_cat_000011.jpg
                ...
            /dog
                flickr_dog_000043.jpg 
                flickr_dog_000045.jpg
                ...
            /wild
                flickr_wild_000004.jpg 
                flickr_wild_000012.jpg
                ...
    ```
   **NOTE: The CIFAR-10 dataset can be downloaded using `torchvision.datasets.CIFAR10`. Run the `create_resized_imgs.py` to save the `.png` type image to the `./dataset/resize_imgs` directory.**
    </details>

4. When preparing the dataset, need to run `create_resized_imgs.py` to crop the image and automatically save it to the `./dataset/resize_imgs` directory

## Running Scripts

To perform the **JPEG-LDPC-QAM-CDMA** operation, run the command below:
```shell
# CIFAR10, AWGN Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_cdma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='awgn'

# CIFAR10, Rayleigh Fading Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_cdma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='rayleigh'

# AFHQ, AWGN Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_cdma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='awgn'

# AFHQ, Rayleigh Fading Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_cdma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='rayleigh'
```

To perform the **JPEG-LDPC-BPSK-NOMA** operation, you can run the following command:
```shell
# CIFAR10, AWGN Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_noma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='awgn'

# CIFAR10, Rayleigh Fading Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_noma.py --dataset='CIFAR10' --distortion-metric='MSE' --channel-type='rayleigh'

# AFHQ, AWGN Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_noma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='awgn'

# AFHQ, Rayleigh Fading Channel
CUDA_VISIBLE_DEVICES=0 python jpeg_ldpc_qam_noma.py --dataset='AFHQ' --distortion-metric='MSE' --channel-type='rayleigh'
```

A folder named `./history` will be created to save the mertics and logs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If our work assists your research, feel free to give us a star ⭐

## Acknowledgement
The implementation is based on [SSCC](https://github.com/BUPT-NextGE/SSCC.git).