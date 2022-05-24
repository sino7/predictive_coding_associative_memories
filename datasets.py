"""
This code is adapted from the repository https://github.com/baudm/MONet-pytorch
"""

import os
import os.path
import json
import torch.utils.data as data
from PIL import Image
import torch
    import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class CLEVRDataset():
    """
    This dataset class can load a set of images specified by the path --dataroot/path/to/data.
    """

    def __init__(self, dataroot='data/CLEVR_v1.0', isTrain=False, input_nc=3, crop_size=192, load_size=64):
        self.dataroot = dataroot
        self.isTrain = isTrain
        self.input_nc = input_nc
        self.crop_size = crop_size
        self.load_size = load_size
        s = 'train' if self.isTrain else 'val'
        path = os.path.join(self.dataroot, 'images', s)
        self.A_paths = sorted(make_dataset(path))
        
        # Filter images with 3 objects
        with open(os.path.join(self.dataroot, 'scenes', 'CLEVR_' + s + '_scenes.json'), 'r') as f:
            scenes = json.load(f)
            image_ids = []
            for scene in scenes['scenes']:
                if len(scene['objects'])==3:
                    image_ids.append(scene['image_index'])
        self.image_ids = image_ids
        self.image_ids = self.image_ids[:len(self.image_ids)-len(self.image_ids)%100]
        
    def _transform(self, img):
        img = TF.resized_crop(img, 64, 29, 160, self.crop_size, self.load_size)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * self.input_nc, [0.5] * self.input_nc)
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns the image
        """
        A_path = self.A_paths[self.image_ids[index]]
        A_img = Image.open(A_path).convert('RGB')
        A = self._transform(A_img)
        return A

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_ids)
    
    def item_size(self):
        return 64*64*3
    
    def item_shape(self):
        return [3, 64, 64]
    
    
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images


class CIFARDataset():

    def __init__(self, dataroot='data', isTrain=False, download=True):
        self.dataroot = dataroot
        self.isTrain = isTrain
        self.dataset = torchvision.datasets.CIFAR10(root=dataroot, train=isTrain, download=True, transform=transforms.ToTensor())
    
    def __getitem__(self, index):
        return self.dataset[index][0]
    
    def __len__(self):
        return len(self.dataset)
    
    def item_size(self):
        return 32*32*3
    
    def item_shape(self):
        return [3, 32, 32]