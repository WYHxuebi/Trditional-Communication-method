import re
from tqdm import tqdm
from utils.utils import *
from utils.datasets import get_loader
from torchvision.transforms import ToPILImage


# Configuration
class config():

    # System Settings
    dataset = 'AFHQ'
    exist_resized_img = False

    # Dataset path
    if dataset == 'CIFAR10':
        image_dims = (3, 32, 32)
        dataset_path = "./dataset/CIFAR10"
        dataset_saved_root = "./dataset/resized_imgs/CIFAR10"
    elif dataset == 'AFHQ':
        image_dims = (3, 256, 256)
        dataset_path = ["./dataset/AFHQ/val/cat", "./dataset/AFHQ/val/dog", "./dataset/AFHQ/val/wild"]
        dataset_saved_root = "./dataset/resized_imgs/AFHQ/val"


if __name__ == "__main__":

    # Create a folder to save images
    makedirs(config.dataset_saved_root)

    # Get the dataset
    test_dataset = get_loader(None, config)
    for idx, (img, item) in tqdm(enumerate(test_dataset)):
        if config.dataset == 'CIFAR10':
            output_path = config.dataset_saved_root + '/{}'.format(item)
            makedirs(output_path)
            output_path = output_path + f"/img_{idx:05d}.png"
        elif config.dataset == 'AFHQ':
            img = ToPILImage()(img)
            output_path = config.dataset_saved_root + '/' + re.split(r'[\\/]', item)[-2]
            makedirs(output_path)
            output_path = output_path + '/' + re.split(r'[\\/]', item)[-1]
        img.save(output_path)
