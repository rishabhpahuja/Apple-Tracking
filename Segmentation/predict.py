import argparse
import logging
import os
import sys
# sys.path.append(os.getcwd()+'/Segmentation')
# os.chdir('./Segmentation')
# sys.path.append(os.getcwd()+'/Segmentation')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
# import ipdb; ipdb.set_trace()

from seg_utils.data_loading import BasicDataset
from unet import UNet
from seg_utils.utils import plot_img_and_mask

class Segmentation:
    def __init__(self,model_path=None,mask_threshold=0.5,scale=0.4, bilinear=False):

        
        self.mask_threshold=mask_threshold
        self.scale=scale
        self.bilinear=bilinear
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        #Declaring Model
        self.model = UNet(n_channels=3, n_classes=2, bilinear=self.bilinear)
        self.model.to(self.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load('../Segmentation/checkpoints/checkpoint_epoch21.pth', map_location=self.device))


    def predict_img(self,full_img, out_threshold=0.5):
               
        self.model.eval()
        full_img=Image.fromarray(full_img,'RGB')
        # full_img=Image.open(full_img)
        # plt.imshow(full_img)
        # plt.show()
        # full_img=Image.open(full_img)
        img = torch.from_numpy(BasicDataset.preprocess(full_img, self.scale, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img)

            if self.model.n_classes > 1:
                probs = F.softmax(output, dim=1)[0]
            else:
                probs = torch.sigmoid(output)[0]

            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1], full_img.size[0])),
                transforms.ToTensor()
            ])

            full_mask = tf(probs.cpu()).squeeze()

        if self.model.n_classes == 1:
            return (full_mask > self.mask_threshold).numpy()
        else:
            out=F.one_hot(full_mask.argmax(dim=0), self.model.n_classes).permute(2, 0, 1).numpy()
            out=self.mask_to_image(out)
            return out

    def mask_to_image(self, mask: np.ndarray):
        if mask.ndim == 2:
            return Image.fromarray((mask * 255).astype(np.uint8))
        elif mask.ndim == 3:
            return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))



