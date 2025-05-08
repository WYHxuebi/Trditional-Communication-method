from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset


class CreatDataset(Dataset):
    def __init__(self, config, data_dir):
        self.imgs = []

        # Get all images in a directory
        for dir in data_dir:
            self.imgs += glob(dir + '/*.jpg')
            self.imgs += glob(dir + '/*.png')
        
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height                         # 图像裁剪大小
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        transforms_list = [
            transforms.Resize((self.im_height, self.im_width)),
            transforms.ToTensor()]
        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.imgs)
    

def get_loader(args, config):
    if config.exist_resized_img:
        dataset = CreatDataset(config, config.dataset_path)
    else:
        if config.dataset == "CIFAR10":
            dataset_ = datasets.CIFAR10
            dataset = dataset_(root=config.dataset_path, train=False, download=True)
        elif config.dataset == "AFHQ":
            dataset = CreatDataset(config, config.dataset_path)
        else:
            raise ValueError("Unsupported dataset.")
    
    return dataset