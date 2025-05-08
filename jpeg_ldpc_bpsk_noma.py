import torch
import torchvision

import os
import numpy as np
import argparse
import re
import random
from tqdm import tqdm
from glob import glob
from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, triang_ldpc_systematic_encode
from skimage.metrics import peak_signal_noise_ratio

from utils.utils import *
from utils.datasets import get_loader
from utils.channel import pass_channel
from utils.lpips import LPIPS
from utils.distortion import MS_SSIM


# parameters
parser = argparse.ArgumentParser(description='JPEG_1/2LDPC_BPSK_NOMA')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'AFHQ'])
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'])
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh', 'rician'])
args = parser.parse_args()


# Configuration
class config():

    # System Settings
    pass_channel = True
    seed = 0
    batch_size = 64
    exist_resized_img = True
    channel_type = args.channel_type

    # Coding Settings
    multiple_snr = np.arange(-5, 26, 1)
    qam_order = 16
    ldpc_design_file = os.path.join(os.path.dirname(__file__), '1440.720.txt')
    power_allocation = [0.7, 0.2, 0.1]
    num_users = 3

    # Log Settings
    filename = "{}_JPEG_LDPC_BPSK_NOMA_{}".format(args.dataset, channel_type.upper())
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    pictures = workdir + '/pictures'
    logger = None

    # Dataset Path
    if args.dataset == 'CIFAR10':
        img_radio = 100
        target_bpp = 6
        image_dims = (3, 32, 32)
        dataset_path = "./dataset/resized_imgs/CIFAR10/0", "./dataset/resized_imgs/CIFAR10/1", "./dataset/resized_imgs/CIFAR10/2", "./dataset/resized_imgs/CIFAR10/3", "./dataset/resized_imgs/CIFAR10/4", "./dataset/resized_imgs/CIFAR10/5", "./dataset/resized_imgs/CIFAR10/6", "./dataset/resized_imgs/CIFAR10/7", "./dataset/resized_imgs/CIFAR10/8", "./dataset/resized_imgs/CIFAR10/9"
        dataset_root = "./dataset/resized_imgs/CIFAR10"
    elif args.dataset == 'AFHQ':
        img_radio = 10
        target_bpp = 0.35
        image_dims = (3, 256, 256)
        dataset_path = ["./dataset/resized_imgs/AFHQ/val/cat", "./dataset/resized_imgs/AFHQ/val/dog", "./dataset/resized_imgs/AFHQ/val/wild"]
        dataset_root = "./dataset/resized_imgs/AFHQ/val"

    # Output Path
    compress_jpeg_path = '/compress_jpeg_img'
    output_txt_path = '/jpeg_bit'
    compress_jpeg_root = pictures + compress_jpeg_path
    output_txt_root = pictures + output_txt_path
    channelcoded_output_base_path = pictures + '/recovered_jpeg_image'


def bpsk_soft_llr(received_signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / (2 * snr_linear)
    llr = 2 * received_signal / noise_var
    return llr


# LDPC coding, QAM modulation, Wireless channel
def ldpc_qam_awgn(input_signal, ldpc_param, snr, channel_type):

    # bit data
    binary_arr_user1 = input_signal[0]
    binary_arr_user2 = input_signal[1]
    binary_arr_user3 = input_signal[2]
    binary_arr = [binary_arr_user1, binary_arr_user2, binary_arr_user3]

    # Channel coding input
    message_bits_user1 = binary_arr_user1
    message_bits_user2 = binary_arr_user2
    message_bits_user3 = binary_arr_user3

    # Find the maximum length
    message_bits_len_user1 = len(message_bits_user1)
    message_bits_len_user2 = len(message_bits_user2)
    message_bits_len_user3 = len(message_bits_user3)
    message_bits_max_len = max(message_bits_len_user1, message_bits_len_user2, message_bits_len_user3)
    modulated_bits_len = [message_bits_len_user1, message_bits_len_user2, message_bits_len_user3]

    # Dimension Alignment
    message_bits_user1 = np.pad(message_bits_user1, (0, message_bits_max_len - message_bits_len_user1), 'constant', constant_values=0)
    message_bits_user2 = np.pad(message_bits_user2, (0, message_bits_max_len - message_bits_len_user2), 'constant', constant_values=0)
    message_bits_user3 = np.pad(message_bits_user3, (0, message_bits_max_len - message_bits_len_user3), 'constant', constant_values=0)
    message_bits = [message_bits_user1, message_bits_user2, message_bits_user3]

    # LDPC Channel Coding
    ldpc_encoded_bits_user1 = triang_ldpc_systematic_encode(message_bits_user1, ldpc_param)
    ldpc_encoded_bits_user2 = triang_ldpc_systematic_encode(message_bits_user2, ldpc_param)
    ldpc_encoded_bits_user3 = triang_ldpc_systematic_encode(message_bits_user3, ldpc_param)

    # Record Dimension
    first_dimension_length_user1 = ldpc_encoded_bits_user1.shape[0]
    first_dimension_length_user2 = ldpc_encoded_bits_user2.shape[0]
    first_dimension_length_user3 = ldpc_encoded_bits_user3.shape[0]
    first_dimension_length = [first_dimension_length_user1, first_dimension_length_user2, first_dimension_length_user3]

    # Dimension Transformation
    bits_user1 = ldpc_encoded_bits_user1.reshape(-1)
    bits_user2 = ldpc_encoded_bits_user2.reshape(-1)
    bits_user3 = ldpc_encoded_bits_user3.reshape(-1)

    # BPSK Modulation
    modulated_bits_user1 = 1 - 2 * bits_user1
    modulated_bits_user2 = 1 - 2 * bits_user2
    modulated_bits_user3 = 1 - 2 * bits_user3

    # Power Distribution
    transmitted_signal_user_1 = np.sqrt(config.power_allocation[0]) * modulated_bits_user1
    transmitted_signal_user_2 = np.sqrt(config.power_allocation[1]) * modulated_bits_user2
    transmitted_signal_user_3 = np.sqrt(config.power_allocation[2]) * modulated_bits_user3
    transmitted_signal = transmitted_signal_user_1 + transmitted_signal_user_2 + transmitted_signal_user_3

    # Wireless Channel
    bits_with_noise_user1 = pass_channel(transmitted_signal, snr, channel_type)
    bits_with_noise_user2 = pass_channel(transmitted_signal, snr, channel_type)
    bits_with_noise_user3 = pass_channel(transmitted_signal, snr, channel_type)
    bits_with_noise = [np.real(bits_with_noise_user1), np.real(bits_with_noise_user2), np.real(bits_with_noise_user3)]

    # Output Sequence
    data_decoded = []

    # SIC: Successive Interference Cancellation
    for i, power in enumerate(config.power_allocation):

        # Power recovery
        current_signal = bits_with_noise[i] / np.sqrt(power)

        # BPSK soft demodulation
        demodulated_bits = bpsk_soft_llr(current_signal, snr)

        # Dimension Transformation
        demodulated_bits = demodulated_bits.reshape(first_dimension_length[i], -1)

        # LDPC Channel Decoding
        ldpc_decoded_bits = ldpc_bp_decode(demodulated_bits.reshape(-1, order='F').astype(float), ldpc_param, 'MSA', 10)[0][:720].reshape(-1, order='F')[:len(message_bits[i])]

        # Output
        data_decoded.append(ldpc_decoded_bits[0:modulated_bits_len[i]])

        # LDPC Channel Coding
        ldpc_encoded_bits_estimated = triang_ldpc_systematic_encode(ldpc_decoded_bits, ldpc_param)

        # Dimension Transformation
        bits_estimated = ldpc_encoded_bits_estimated.reshape(-1)

        # BPSK Modulation
        modulated_bits_estimated = 1 - 2 * bits_estimated

        # Power Distribution
        estimated_signal = np.sqrt(power) * modulated_bits_estimated

        # Interference Cancellation
        for j in range(i+1, len(config.power_allocation)):
            bits_with_noise[j] = bits_with_noise[j] - estimated_signal

    return data_decoded[0], data_decoded[1], data_decoded[2]


if __name__ == "__main__":

    # Log
    logger = logger_configuration(config, save_log=True)
    logger.info(args)

    # Get dataset
    test_dataset = get_loader(args, config)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, pin_memory=True, 
                                              batch_size=config.batch_size, 
                                              drop_last=False, shuffle=False)

    # Defining Relevant Metrics
    results_jpeg_cbr = 0.0
    results_psnr_user1 = np.zeros(len(config.multiple_snr))
    results_msssim_user1 = np.zeros(len(config.multiple_snr))
    results_lpips_user1 = np.zeros(len(config.multiple_snr))
    results_psnr_user2 = np.zeros(len(config.multiple_snr))
    results_msssim_user2 = np.zeros(len(config.multiple_snr))
    results_lpips_user2 = np.zeros(len(config.multiple_snr))
    results_psnr_user3 = np.zeros(len(config.multiple_snr))
    results_msssim_user3 = np.zeros(len(config.multiple_snr))
    results_lpips_user3 = np.zeros(len(config.multiple_snr))

    # LPIPS Mertic
    calu_LPIPS = LPIPS().cuda().eval()

    # MS-SSIM Mertic
    if args.dataset == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=5, channel=3).cuda()

    # 1/2LPDC Coding Matrix
    ldpc_param = get_ldpc_code_params(config.ldpc_design_file)

    # JPEG Coding
    logger.info("Starting JPEG compression...")
    total_bpp = 0
    for batch_imgs, batch_imgs_path in tqdm(test_loader):
        batch_size = len(batch_imgs)
        for i in range(batch_size):

            # Create the output directory for the JPEG image coding
            output_path = config.compress_jpeg_root + '/jpeg_img/' + re.split(r'[\\/]', batch_imgs_path[i])[-2]
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_img_path = output_path + '/' + re.split(r'[\\/]', batch_imgs_path[i])[-1].split('.')[0] + '.JPEG'
            
            # JPEG Coding
            img = batch_imgs[i]
            img = ToPILImage()(img)
            bpp_per_img = find_closest_bpp(config.target_bpp, img, output_img_path, fmt='JPEG')
            total_bpp += bpp_per_img
    
    avg_bpp = total_bpp / len(test_dataset)
    logger.info('总图像数量：{}, 平均bpp: {}'.format(len(test_dataset), avg_bpp))

    # JPEG compressed image path
    jpeg_images_path = glob(os.path.join(config.compress_jpeg_root + '/jpeg_img', '**/*.JPEG'), recursive=True)

    # Read the byte stream, convert it into a bit stream, and save in txt
    logger.info("Starting JPEG bit saving...")
    for jpeg_img_dir in tqdm(jpeg_images_path):
        jpeg_img_sub_dir = os.path.basename(os.path.dirname(jpeg_img_dir))
        jpeg_img_name = jpeg_img_dir.split('/')[-1].split('.')[0]
        directory = config.output_txt_root + '/' + jpeg_img_sub_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_dir = directory + '/' + jpeg_img_name + '.txt'
        img_bit(jpeg_img_dir, file_dir)

    # Get txt paths
    input_txts_path = glob(os.path.join(config.output_txt_root, '**/*.txt'), recursive=True)
    total = int(len(input_txts_path) / config.img_radio)

    # Generate three user data
    input_txts_path_user1 = random.sample(input_txts_path, total)
    input_txts_path_user2 = random.sample(input_txts_path, total)
    input_txts_path_user3 = random.sample(input_txts_path, total)

    # Different SNR
    CBR_JPEG = 0.0
    for snr_idx, snr in enumerate(config.multiple_snr):
        logger.info("{}/{}: {} dB".format(snr_idx, len(config.multiple_snr), snr))
        psnr_user1 = 0.0
        ssim_user1 = 0.0
        lpips_user1 = 0.0
        psnr_user2 = 0.0
        ssim_user2 = 0.0
        lpips_user2 = 0.0
        psnr_user3 = 0.0
        ssim_user3 = 0.0
        lpips_user3 = 0.0

        for txt_path_user1, txt_path_user2, txt_path_user3 in tqdm(list(zip(input_txts_path_user1, input_txts_path_user2, input_txts_path_user3))):

            # Read the corresponding txt to get the bit stream array
            img_bitarray_user1 = get_bitarray(txt_path_user1)
            img_bitarray_user2 = get_bitarray(txt_path_user2)
            img_bitarray_user3 = get_bitarray(txt_path_user3)
            img_bitarray = [img_bitarray_user1, img_bitarray_user2, img_bitarray_user3]

            # Calculating JPEG compression ratio
            raw_img_bit_length = config.image_dims[0] * config.image_dims[1] * config.image_dims[2] * 8
            CBR_JPEG = CBR_JPEG + (len(img_bitarray_user1) + len(img_bitarray_user2) + len(img_bitarray_user3)) / (3 * raw_img_bit_length)

            # LDPC encoding, QAM modulation, wireless channel, QAM demodulation, LDPC decoding
            output_signal_user1, output_signal_user2, output_signal_user3 = ldpc_qam_awgn(img_bitarray, ldpc_param, snr, args.channel_type)

            # Convert to bitstream
            bitstring_user1 = bit_to_string(output_signal_user1)
            bitstring_user2 = bit_to_string(output_signal_user2)
            bitstring_user3 = bit_to_string(output_signal_user3)

            # Find the txt corresponding to the original image
            raw_img_user1, jpeg_img_path_user1 = find_raw_img(config.compress_jpeg_root, txt_path_user1, config.dataset_root, args.dataset)
            raw_img_user2, jpeg_img_path_user2 = find_raw_img(config.compress_jpeg_root, txt_path_user2, config.dataset_root, args.dataset)
            raw_img_user3, jpeg_img_path_user3 = find_raw_img(config.compress_jpeg_root, txt_path_user3, config.dataset_root, args.dataset)

            # The reconstructed image
            recovered_img_user1 = bit_to_img(bitstring_user1, jpeg_img_path_user1, output_path=config.channelcoded_output_base_path, snr=snr, dataset=args.dataset)
            recovered_img_user2 = bit_to_img(bitstring_user2, jpeg_img_path_user2, output_path=config.channelcoded_output_base_path, snr=snr, dataset=args.dataset)
            recovered_img_user3 = bit_to_img(bitstring_user3, jpeg_img_path_user3, output_path=config.channelcoded_output_base_path, snr=snr, dataset=args.dataset)

            # Image pre-transformation
            raw_img_user1 = torchvision.transforms.Resize((config.image_dims[1], config.image_dims[2]))(raw_img_user1)
            raw_img_user2 = torchvision.transforms.Resize((config.image_dims[1], config.image_dims[2]))(raw_img_user2)
            raw_img_user3 = torchvision.transforms.Resize((config.image_dims[1], config.image_dims[2]))(raw_img_user3)
            recovered_img_user1 = torchvision.transforms.Resize((config.image_dims[1], config.image_dims[2]))(recovered_img_user1)
            recovered_img_user2 = torchvision.transforms.Resize((config.image_dims[1], config.image_dims[2]))(recovered_img_user2)
            recovered_img_user3 = torchvision.transforms.Resize((config.image_dims[1], config.image_dims[2]))(recovered_img_user3)
            
            # Numpy type
            raw_img_array_user1 = np.array(raw_img_user1)
            raw_img_array_user2 = np.array(raw_img_user2)
            raw_img_array_user3 = np.array(raw_img_user3)
            recovered_img_array_user1 = np.array(recovered_img_user1)
            recovered_img_array_user2 = np.array(recovered_img_user2)
            recovered_img_array_user3 = np.array(recovered_img_user3)
            
            # PSNR
            psnr_user1 = psnr_user1 + peak_signal_noise_ratio(raw_img_array_user1, recovered_img_array_user1)
            psnr_user2 = psnr_user2 + peak_signal_noise_ratio(raw_img_array_user2, recovered_img_array_user2)
            psnr_user3 = psnr_user3 + peak_signal_noise_ratio(raw_img_array_user3, recovered_img_array_user3)

            # Tensor type
            raw_img_tensor_user1 = torch.from_numpy(raw_img_array_user1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            raw_img_tensor_user2 = torch.from_numpy(raw_img_array_user2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            raw_img_tensor_user3 = torch.from_numpy(raw_img_array_user3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            recovered_img_tensor_user1 = torch.from_numpy(recovered_img_array_user1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            recovered_img_tensor_user2 = torch.from_numpy(recovered_img_array_user2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            recovered_img_tensor_user3 = torch.from_numpy(recovered_img_array_user3).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # MS-SSIM
            ms_ssim_user1 = CalcuSSIM(raw_img_tensor_user1.cuda(), recovered_img_tensor_user1.cuda())
            ms_ssim_user2 = CalcuSSIM(raw_img_tensor_user2.cuda(), recovered_img_tensor_user2.cuda())
            ms_ssim_user3 = CalcuSSIM(raw_img_tensor_user3.cuda(), recovered_img_tensor_user3.cuda())
            ssim_user1 = ssim_user1 + ms_ssim_user1.item()
            ssim_user2 = ssim_user2 + ms_ssim_user2.item()
            ssim_user3 = ssim_user3 + ms_ssim_user3.item()

            # LPIPS
            lpips_user1 = lpips_user1 + torch.mean(calu_LPIPS(raw_img_tensor_user1.cuda().contiguous(), recovered_img_tensor_user1.cuda().contiguous()))
            lpips_user2 = lpips_user2 + torch.mean(calu_LPIPS(raw_img_tensor_user2.cuda().contiguous(), recovered_img_tensor_user2.cuda().contiguous()))
            lpips_user3 = lpips_user3 + torch.mean(calu_LPIPS(raw_img_tensor_user3.cuda().contiguous(), recovered_img_tensor_user3.cuda().contiguous()))
        
        results_psnr_user1[snr_idx] = psnr_user1 / total
        results_msssim_user1[snr_idx] = ssim_user1 / total
        results_lpips_user1[snr_idx] = lpips_user1 / total
        results_psnr_user2[snr_idx] = psnr_user2 / total
        results_msssim_user2[snr_idx] = ssim_user2 / total
        results_lpips_user2[snr_idx] = lpips_user2 / total
        results_psnr_user3[snr_idx] = psnr_user3 / total
        results_msssim_user3[snr_idx] = ssim_user3 / total
        results_lpips_user3[snr_idx] = lpips_user3 / total

        # Record mertic
        log_content = (' | '.join([
            f'CBR {CBR_JPEG / (total * (snr_idx + 1)):.4f}',
            f'SNR_USER1 {snr-10}',
            f'PSNR_USER1 {results_psnr_user1[snr_idx]:.4f}',
            f'MSSSIM_USER1 {results_msssim_user1[snr_idx]:.4f}',
            f'LPIPS_USER1 {results_lpips_user1[snr_idx]:.4f}',
            f'SNR_USER2 {snr-5}',
            f'PSNR_USER2 {results_psnr_user2[snr_idx]:.4f}',
            f'MSSSIM_USER2 {results_msssim_user2[snr_idx]:.4f}',
            f'LPIPS_USER2 {results_lpips_user2[snr_idx]:.4f}',
            f'SNR_USER3 {snr}',
            f'PSNR_USER3 {results_psnr_user3[snr_idx]:.4f}',
            f'MSSSIM_USER3 {results_msssim_user3[snr_idx]:.4f}',
            f'LPIPS_USER3 {results_lpips_user3[snr_idx]:.4f}']))
        logger.info(log_content)

    results_jpeg_cbr = CBR_JPEG / (total * len(config.multiple_snr))

    # Record mertic
    log_content = (' | '.join([
        f'CBR {results_jpeg_cbr})\n',
        f'SNR {config.multiple_snr})\n',
        f'PSNR_USER1 {results_psnr_user1.tolist()})\n',
        f'MSSSIM_USER1 {results_msssim_user1.tolist()})\n',
        f'LPIPS_USER1 {results_lpips_user1.tolist()})\n',
        f'PSNR_USER2 {results_psnr_user2.tolist()})\n',
        f'MSSSIM_USER2 {results_msssim_user2.tolist()})\n',
        f'LPIPS_USER2 {results_lpips_user2.tolist()})\n',
        f'PSNR_USER3 {results_psnr_user3.tolist()})\n',
        f'MSSSIM_USER3 {results_msssim_user3.tolist()})\n',
        f'LPIPS_USER3 {results_lpips_user3.tolist()})\n',
        f'Finish Test!']))
    logger.info(log_content)

    # Storage related mertics
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_CBRJPEG.npy', results_jpeg_cbr)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_SNR_USER1.npy', config.multiple_snr-10)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_PSNR_USER1.npy', results_psnr_user1)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_SSIM_USER1.npy', results_msssim_user1)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_LPIPS_USER1.npy', results_lpips_user1)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_SNR_USER2.npy', config.multiple_snr-5)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_PSNR_USER2.npy', results_psnr_user2)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_SSIM_USER2.npy', results_msssim_user2)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_LPIPS_USER2.npy', results_lpips_user2)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_SNR_USER3.npy', config.multiple_snr)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_PSNR_USER3.npy', results_psnr_user3)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_SSIM_USER3.npy', results_msssim_user3)
    np.save(config.samples + '/JPEG_LDPC_BPSK_NOMA_LPIPS_USER3.npy', results_lpips_user3)