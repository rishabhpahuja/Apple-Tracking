import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

dir_img = './data/RGB/'
dir_mask = './data/Masks/'
dir_checkpoint = './checkpoints/'


# def train_net(net, device, epoch, batch_size, learning_rate, val_percent, 
#             save_checkpoint, img_scale, amp,WANDB=True):
def train_net(net, device,optimizer,criterion, train_loader, epoch_loss,epoch ,amp):

    net.train()
    
    with tqdm(train_loader, unit='batch',desc='Train Epoch:'+str(epoch)) as pbar:
        for batch in enumerate(pbar):
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
                loss = criterion(masks_pred.squeeze(), true_masks.squeeze().float()) \
                        + dice_loss(masks_pred.squeeze().float(),
                                    true_masks.squeeze().float(),
                                    multiclass=False)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(**{'loss (batch)':loss.item()})
        
    return epoch_loss/len(train_loader)

# def validate(net,save_checkpoint,epoch):

#     net.eval()

#     val_score = evaluate(net, val_loader, device)                     
#     print('Validation Dice score: {}'.format(val_score))


#     if save_checkpoint:
#         torch.save(net.state_dict(), dir_checkpoint + '/checkpoint_epoch{}.pth'.format(str(epoch)))
#         logging.info(f'Checkpoint {epoch} saved!')

    
#     return val_score

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=3, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.4, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--WANDB', '-w', type=bool, default=False, help='whether to use WANDB for data logging')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    # import ipdb; ipdb.set_trace()
    net.to(device=device)

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, scale=args.scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * args.val / 100)
    # import ipdb; ipdb.set_trace()
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.BCEWithLogitsLoss()
    best_val=0
    loss=0

    for epoch in range(args.epochs):

        loss=train_net(net, device,optimizer,criterion, train_loader,epoch_loss=loss,epoch=epoch, amp=args.amp)
        print("Average Loss:",epoch)
        val_score = evaluate(net, val_loader, device) 
        scheduler.step(val_score)
        print("Dice coeff for epoch "+str(epoch),val_score)

        if val_score>best_val:
            torch.save(net.state_dict(), dir_checkpoint + '/checkpoint_epoch{}.pth'.format(str(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            best_val=val_score