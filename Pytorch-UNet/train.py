import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from utils.dice_score import multiclass_dice_coeff, dice_coeff

dir_img_train = './data/train/RGB/'
dir_mask_train = './data/train/Masks/'
dir_img_val= './data/eval/RGB/'
dir_mask_val= './data/eval/Masks/'
dir_checkpoint = './checkpoints/'


def train_net(net, device, epochs, batch_size, learning_rate, val_percent, 
            save_checkpoint, img_scale, amp,WANDB=True,display_data=True):

    # 1. Create dataset
    dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale,dataset='train')
    dataset_eval = BasicDataset(dir_img_val, dir_mask_val, img_scale)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # # import ipdb; ipdb.set_trace()
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(dataset_train, shuffle=True, **loader_args)
    val_loader = DataLoader(dataset_eval, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if WANDB:
        experiment = wandb.init(project="Apple Tracking", entity="rpahuja")
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                    val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                    amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(dataset_train)}
        Validation size: {len(dataset_eval)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.functional.cross_entropy()
    # global_step = int(n_train/batch_size)

    # 5. Begin training
    best_score=0

    if display_data:

        for batch in (train_loader):
            image=batch['image']
            true_mask=batch['mask'].unsqueeze(1)
            grid_img= torchvision.utils.make_grid(image, nrow=5)
            grid_mask= torchvision.utils.make_grid(true_mask, nrow=5)
            mask_image=torch.cat((grid_img,grid_mask),axis=1)
            plt.imshow(mask_image.permute(1, 2, 0))
            # plt.imshow(grid_mask.permute(1,2,0))
            plt.show()
            break

    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(train_loader,unit='batch',desc='Train Epoch:'+str(epoch)) as pbar:
            for batch in pbar:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = (true_masks/255).to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss=nn.functional.cross_entropy(masks_pred,true_masks)\
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # pbar.update(images.shape[0])
                
                epoch_loss += loss.item()
                if WANDB:
                    experiment.log({
                    'train loss': loss.item(),                    
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            val_score = evaluate(net, val_loader, device)
            # print("Validation Score:",val_score)
            scheduler.step(val_score)

            logging.info('Validation Dice score: {}'.format(val_score))
    
            if save_checkpoint:
                if val_score>=best_score:
                    best_score=val_score
                    torch.save(net.state_dict(), dir_checkpoint + '/checkpoint_epoch{}.pth'.format(str(epoch)))
                    logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=3, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.4, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--WANDB', '-w', type=bool, default=False, help='whether to use WANDB for data logging')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=1.0)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    # net=torch.nn.DataParallel(net)
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100, save_checkpoint=True,
                  amp=args.amp, WANDB=args.WANDB)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise