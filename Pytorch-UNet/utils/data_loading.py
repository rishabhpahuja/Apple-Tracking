import logging
import os
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, images_dir: str,masks_dir: str, scale: float = 1.0,dataset =None):
        self.images_path = images_dir
        self.masks_path = masks_dir
        # import ipdb;ipdb.set_trace()
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids=os.listdir(self.images_path)
        self.transforms_image=transforms.Compose([
                                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                    transforms.ToPILImage(),
                                                    transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.25),
                                                    # transforms.RandomGrayscale(p=0.1),
                                                    # transforms.RandomInvert(p=0.1),
                                                    transforms.RandomAdjustSharpness(sharpness_factor=2,p=0.5),
                                                    transforms.RandomAutocontrast(p=0.5),
                                                    transforms.RandomEqualize(p=0.5),                                                    
                                                    transforms.ToTensor()
                                                ]
                                                    )
        self.dataset=dataset

        if len(self.ids)==0:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = self.masks_path+name
        img_file = self.images_path+name

        mask = self.load(mask_file)
        img = self.load(img_file)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = torch.as_tensor(self.preprocess(img, self.scale, is_mask=False).copy()).float().contiguous()
        mask = torch.as_tensor(self.preprocess(mask, self.scale, is_mask=True).copy()).long().contiguous()

        if self.dataset=='train':
            img=self.transforms_image(img)

        return {
            'image': img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale)