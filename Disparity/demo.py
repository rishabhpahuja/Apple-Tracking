import sys
import os
# sys.path.append(os.getcwd()+'/Disparity')
import ipdb
import argparse
import glob
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
import cv2

# import ipdb; ipdb.set_trace()
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

class Disparity:

    def __init__(self,resize=None,model_path=None):
        
        os.chdir('../Disparity')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args=self.main(model_path)
        self.resize=resize
        
        #Defining the model
        self.model = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
        # self.model=RAFTStereo(self.args)
        self.model.load_state_dict(torch.load(self.args.restore_ckpt))
        os.chdir('../deep_sort')
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def load_image(self,imfile):
        img=imfile.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.resize:
            img_resize=T.Resize(self.resize)(img)
            return img_resize[None].to(self.device),img.shape
        
        return img[None].to(self.device)

    def find_disparity(self,imgl,imgr,save=False):

        with torch.no_grad():

            os.chdir('../')
            
            if self.resize:
                image1,size = self.load_image(imgl)
                image2,_ = self.load_image(imgr)
            
            else:
                image1 = self.load_image(imgl)
                image2 = self.load_image(imgr)

            padder = InputPadder(image1.shape, divis_by=1)
            image1, image2 = padder.pad(image1, image2)

            import time
            start=time.time()
            _, flow_up = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
            end=time.time()
            print(end-start)

            if self.resize:
                # import ipdb;ipdb.set_trace()
                flow_up=T.Resize((size[1],size[2]))(flow_up)

            if save:
                cv2.imwrite('85_.png',(flow_up.squeeze().cpu().numpy())*-1)
            
            return((flow_up.squeeze().cpu().numpy())*-1)

    def main(self,model_path):

        parser = argparse.ArgumentParser()
        
        if model_path is None:
            parser.add_argument('--restore_ckpt',default='raftstereo-realtime.pth' ,help="restore checkpoint")
        
        else:
            parser.add_argument('--restore_ckpt',default=model_path ,help="restore checkpoint")
        parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
        parser.add_argument('--output_directory', help="directory to save output", default="./test")
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=7, help='number of flow-field updates during forward pass')

        # Architecture choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_false', help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=3, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', action='store_false', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")

        args = parser.parse_args()

        return args
