import torch
from torchvision.transforms import ToPILImage

import numpy as np
import os
import logging
from PIL import Image
from scipy.linalg import hadamard


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def logger_configuration(config, save_log=False, test_mode=False):
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.pictures)
        makedirs(config.compress_jpeg_root)
        makedirs(config.output_txt_root)
        makedirs(config.channelcoded_output_base_path)

    if os.path.exists(config.log):
        os.remove(config.log)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def find_closest_bpp(target, img, dir, fmt='JPEG'):
    '''
    Determine the quality factor of JPEG encoding based on binary selection and bpp
    '''
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        bpp = pillow_encode(img, dir, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return bpp


def pillow_encode(img, output_img_path, fmt='JPEG', quality=10):
    '''
    Save the image in JPEG encoding
    '''
    img.save(output_img_path, format=fmt, quality=quality)

    filesize = os.path.getsize(output_img_path)
    bpp = filesize * float(8) / (img.size[0] * img.size[1])

    return bpp



def img_bit(input_path, output_path):
    '''
    Read the byte stream, convert it into a bit stream, and save it in txt
    '''
    file = open(input_path, 'rb')
    file_context = file.read()

    tmp_a = []
    bit_all = ''
    for i in file_context:
        tmp_a.append(i)
    tmp_b = np.array(tmp_a, dtype=np.uint8)
    for j in tmp_b:
        k = bin(j).replace('0b', '').rjust(8, '0')
        bit_all = bit_all + k
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(bit_all)
        f.close()


def get_bitarray(txt_path):
    '''
    Read the corresponding txt to get the bit stream array
    '''
    with open(txt_path, 'r') as f:
        f_context = f.read().strip()
        k_char = cut_string(f_context, 1)
        k = [int(a, 2) for a in k_char]
        bit_array = np.array(k)
        return bit_array
    

def cut_string(obj, sec):
    return [obj[i:i + sec] for i in range(0, len(obj), sec)]


def bit_to_img(string, img_dir, output_path, snr, dataset):
    '''
    Convert the bitstream string back to an image. 
    If the converted image cannot be opened, convert it to a random noise image of the corresponding size.
    '''
    split_char = cut_string(string, 8)
    
    int_8 = [int(a, 2) for a in split_char]
    out_stream = np.array(int_8, dtype=np.uint8)
    
    directory = output_path + '/SNR{}/'.format(snr) + os.path.basename(os.path.dirname(img_dir))
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_output_path = directory + '/' + os.path.basename(img_dir)

    if os.path.exists(img_output_path):
        os.remove(img_output_path)

    out_stream.tofile(img_output_path)

    try:
        out_stream = Image.open(img_output_path).convert('RGB')
    except (IOError, Image.DecompressionBombError):
        if dataset == 'CIFAR10':
            width = height = 32
        elif dataset == 'AFHQ':
            width = height = 256
        random_noise(3, width, height).save(img_output_path)
        out_stream = Image.open(img_output_path).convert('RGB')
    
    return out_stream


def random_noise(nc, width, height):
    """Generator a random noise image from tensor.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output image.
        height (int): height of output image.
    Returns:
        PIL Image.
    """
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    return img


def generate_cdma_codes(num_users, code_length):
    '''
    Generate CDMA spreading codes
    '''
    H = hadamard(code_length)
    codes = np.zeros((num_users, code_length), dtype=int)
    for i in range(num_users):
        codes[i] = H[i]
    return codes


def bit_to_string(input_signal):
    bitstring = ''
    for i in input_signal:
        bitstring += str(i)
    return bitstring


def find_raw_img(compress_jpeg_root, txt_path, dataset_root, dataset):
    '''
    Find the txt corresponding to the original image
    '''
    jpeg_img_base_dir = compress_jpeg_root + '/jpeg_img'
    jpeg_img_path = jpeg_img_base_dir + '/' + txt_path.split('/')[-2] + '/' + txt_path.split('/')[-1].split('.')[0] + '.JPEG'
    if dataset == 'CIFAR10':
        raw_img_path = dataset_root + '/' + txt_path.split('/')[-2] + '/' + txt_path.split('/')[-1].split('.')[0] + '.png'
    else:
        raw_img_path = dataset_root + '/' + txt_path.split('/')[-2] + '/' + txt_path.split('/')[-1].split('.')[0] + '.jpg'
    raw_img = Image.open(raw_img_path).convert('RGB')
    return raw_img, jpeg_img_path