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
    def __init__(self,model_path=None):

        os.chdir('./Segmentation')
        self.args=self.get_args(model_path)
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        # self.device='cuda'
        #Declaring Model
        self.model = UNet(n_channels=3, n_classes=2, bilinear=self.args.bilinear)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.args.model, map_location=self.device))
        os.chdir('../deep_sort')
        # self.model.eval()

    def predict_img(self,full_img, scale_factor=1, out_threshold=0.5):
        
        # os.chdir('../')        
        self.model.eval()
        full_img=Image.fromarray(full_img,'RGB')
        # full_img=Image.open(full_img)
        # plt.imshow(full_img)
        # plt.show()
        # full_img=Image.open(full_img)
        img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
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
        # import ipdb; ipdb.set_trace()
        if self.model.n_classes == 1:
            return (full_mask > self.args.mask_threshold).numpy()
        else:
            out=F.one_hot(full_mask.argmax(dim=0), self.model.n_classes).permute(2, 0, 1).numpy()
            out=self.mask_to_image(out)
            return out


    def get_args(self,model_path):
        parser = argparse.ArgumentParser(description='Predict masks from input images')
        
        if model_path is None:
            parser.add_argument('--model', '-m', default='./checkpoints/checkpoint_epoch21.pth', metavar='FILE',
                            help='Specify the file in which the model is stored')
        else:
            parser.add_argument('--model', '-m', default=model_path, metavar='FILE',
                            help='Specify the file in which the model is stored')
        parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
        parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
        parser.add_argument('--viz', '-v', action='store_true',
                            help='Visualize the images as they are processed')
        parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
        parser.add_argument('--mask_threshold', '-t', type=float, default=0.5,
                            help='Minimum probability value to consider a mask pixel white')
        parser.add_argument('--scale', '-s', type=float, default=0.4,
                            help='Scale factor for the input images')
        parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

        return parser.parse_args()

    def mask_to_image(self, mask: np.ndarray):
        if mask.ndim == 2:
            return Image.fromarray((mask * 255).astype(np.uint8))
        elif mask.ndim == 3:
            return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))


# if __name__ == '__main__':
#     args = get_args()
#     in_files = './AAA_4420_.png'
#     # in_files='L0085.jpeg'
#     out_files = './AAA_4420_.jpeg'

#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')
#     logging.info('Model loaded!')

#     # for i, filename in enumerate(in_files):
#     logging.info(f'\nPredicting image {in_files} ...')
#     img = Image.open(in_files)

#     mask = predict_img(net=self.model,
#                         full_img=img,
#                         scale_factor=args.scale,
#                         out_threshold=args.mask_threshold,
#                         device=self.device)
#     # import ipdb; ipdb.set_trace()
#     if not args.no_save:
#         out_filename = out_files
#         result = mask_to_image(mask)
#         result.save(out_filename)
#         logging.info(f'Mask saved to {out_filename}')

#     if args.viz:
#         logging.info(f'Visualizing results for image {in_files}, close to continue...')
#         plot_img_and_mask(img, mask)
