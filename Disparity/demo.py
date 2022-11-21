import sys
import os
sys.path.append(os.getcwd()+'/Disparity')
# import ipdb;ipdb.set_trace()
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    os.chdir('./Disparity')
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # output_directory = Path(args.output_directory)
    # output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        # left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        # right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        # print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        # for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        image1 = args.left_imgs
        image2 = args.right_imgs

        padder = InputPadder(image1.shape, divis_by=1)
        image1, image2 = padder.pad(image1, image2)

        import time
        start=time.time()
        _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        end=time.time()
        print(start-end)
        # file_stem = imfile1.split('/')[-2]
        # if args.save_numpy:
        #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
        # import ipdb; ipdb.set_trace()
        # cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
        cv2.imwrite('Output.png',(flow_up.squeeze().cpu().numpy())*-1)
        return((flow_up.squeeze().cpu().numpy())*-1)
        cv2.imshow('Disparity',(np.asarray(flow_up.squeeze().cpu().numpy(),np.uint8)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')


def main(imgl,imgr):
    imgl = torch.from_numpy(imgl.copy()).permute(2, 0, 1).float()[None].to(DEVICE)
    imgr = torch.from_numpy(imgr.copy()).permute(2, 0, 1).float()[None].to(DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt',default='./raftstereo-middlebury.pth' ,help="restore checkpoint")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=imgl)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=imgr)
    parser.add_argument('--output_directory', help="directory to save output", default="./test")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    disparity=demo(args)
    return disparity
