from __future__ import print_function
import glob
import os
import numpy as np
from PIL import Image
from torch import Tensor
import torch.utils.data as data
import time

class ImageNetLimited(data.Dataset):
    """ImageNet Limited dataset."""
    
    def __init__(self, root_dir, transforms=None, labels=True):
        init_time = time.time()

        self.images = []
        self.labels = []

        if labels:
            print(f'Loading images from the {len(os.listdir(root_dir))} subdirectories of directory: {root_dir}')
            for i, label in enumerate(os.listdir(root_dir)): # for each subdirectory containing all images of one specific class
                print(f'Subdirectory {i+1}/{len(os.listdir(root_dir))}')

                for img_name in glob.glob(os.path.join(root_dir, label, '*.jpg')): # for each image file in this subdirectory
                    with Image.open(img_name) as img:
                        self.images.append(img.copy())
                    self.labels.append(int(label))
        else:
            # used for test directory (no labels):
            for img_name in glob.glob(os.path.join(root_dir, '*.jpg')):
                with Image.open(img_name) as img:
                    self.images.append(img.copy())
                
                # no labels on test set -> instead the list of labels is used to store the images' id
                try:
                    img_id = int(img_name.split('.')[-2].split('\\')[-1])
                except:
                    img_id = int(img_name.split('.')[-2].split('/')[-1])
                self.labels.append(img_id)

        assert len(self.images) == len(self.labels)
        print('Converting images & labels to tensors...')
        
        self.labels = Tensor(self.labels)
        self.transforms = transforms

        print('It took:', time.time() - init_time, 'seconds')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transforms:
            image = self.transforms(image)
        label = self.labels[idx]
        return image, label
        
    def set_transforms(self, transforms):
        self.transforms = transforms