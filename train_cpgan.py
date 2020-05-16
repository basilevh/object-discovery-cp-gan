# Basile Van Hoorick, March 2020
# Train PyTorch CP-GAN

import argparse
import itertools, os, platform, random, time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from cpgan_data import *
from cpgan_model import *
from cpgan_tools import *


if __name__ == '__main__':

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='squares', type=str, help='Image source (squares / noisy / data_folder_name) (default: squares)')
    parser.add_argument('--img_dim', default=64, type=int, help='Image dimension (if not squares or noisy) (default: 64)')
    parser.add_argument('--gpus', nargs='+', default=[], type=int, help='GPU IDs to use (0-based) (default: use all GPUs)')
    parser.add_argument('--batch_size', default=256, type=int, help='Mini-batch size (default: 256)')
    parser.add_argument('--blur_sigma', default=1.0, type=float, help='Gaussian blur sigma for input to discriminator (default: 1.0)')
    parser.add_argument('--suffix', default='', type=str, help='Custom suffix for file paths (model & tensorboard)')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs to train up to (default: 200)')
    parser.add_argument('--autoencode', default=False, action='store_true', help='If True, generate the object to copy in RGB space as well')
    parser.add_argument('--back_real', default=False, action='store_true', help='If True, also encourage the discriminator to see backgrounds as real (not just foregrounds).')
    args = parser.parse_args()

    # Enforce reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # Define model tag
    use_cifar_background = (args.dataset in ['squares', 'noisy'])
    if args.autoencode:
        print('Autoencoding enabled! Generator now has 4 output channels instead of 1')
        model_tag = 'ae_' # reconstruct object (produce ARGB)
    else:
        model_tag = 'def_' # default (produce copy mask only)
    model_tag += args.dataset
    if use_cifar_background:
        print('Image dimension: ' + str(args.img_dim))
        model_tag += '_dim' + str(args.img_dim)
    model_tag += '_bs' + str(args.batch_size)
    if args.back_real:
        print('Background loss term enabled with weight 0.5')
        model_tag += '_br'
    if args.blur_sigma != 1.0:
        print('Gaussian blur sigma: {:.1f}'.format(args.blur_sigma))
        model_tag += '_bs{:.1f}'.format(args.blur_sigma)
    if len(args.suffix):
        model_tag += '_' + args.suffix
    
    # Define paths
    pt_name_G = 'cpgan_G_{}.pt'
    pt_name_D = 'cpgan_D_{}.pt'
    save_path_G = 'models/' + model_tag + '/' + pt_name_G
    save_path_D = 'models/' + model_tag + '/' + pt_name_D
    tb_dir = 'tb_runs/' + model_tag + '/'
    imgs_dir = 'images/' + model_tag + '/'
    print('Generator path:', save_path_G)
    print('Discriminator path:', save_path_D)
    print('Tensorboard directory:', tb_dir)
    print('Image output directory:', imgs_dir)

    # Initialize dataset
    if use_cifar_background:
        # squares or noisy (always 32x32)
        train_back_dir = 'data/CIFAR-10/train/'
        val_back_dir = 'data/CIFAR-10/test/' # CIFAR-10 doesn't have val
        if not(os.path.exists(val_back_dir)): # but let's not be confusing
            val_back_dir = 'data/CIFAR-10/val/'
        train_data = MySquaresDataset(train_back_dir, rand_horz_flip=True, noisy=(args.dataset == 'noisy'), max_objects=5)
        val_data = MySquaresDataset(val_back_dir, rand_horz_flip=True, noisy=(args.dataset == 'noisy'), max_objects=5)

    else:
        # custom dataset from folders containing images
        train_back_dir = 'data/' + args.dataset + '/train_back/'
        train_fore_dir = 'data/' + args.dataset + '/train_fore/'
        train_mask_dir = 'data/' + args.dataset + '/train_mask/'
        val_back_dir = 'data/' + args.dataset + '/val_back/'
        val_fore_dir = 'data/' + args.dataset + '/val_fore/'
        val_mask_dir = 'data/' + args.dataset + '/val_mask/'
        if not(os.path.exists(train_mask_dir)):
            train_mask_dir = None
        if not(os.path.exists(val_mask_dir)):
            val_mask_dir = None # cannot measure ODP in that case
        train_data = MyCopyPasteDataset(train_fore_dir, train_back_dir, train_mask_dir, post_resize=args.img_dim, center_crop=False)
        val_data = MyCopyPasteDataset(val_fore_dir, val_back_dir, val_mask_dir, post_resize=args.img_dim, center_crop=False)

    # Initialize dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize model
    if len(args.gpus):
        device = torch.device('cuda:' + str(args.gpus[0])) # specify first GPU ID
    else:
        device = torch.device('cuda') # use all GPUs
    G_net = MyUNet(3, 4 if args.autoencode else 1, border_zero=True).to(device)
    D_net = MyUNet(3, 1, blur_sigma=args.blur_sigma).to(device)
    if len(args.gpus):
        print('GPU IDs:', args.gpus)
        G_net = nn.DataParallel(G_net, device_ids=args.gpus)
        D_net = nn.DataParallel(D_net, device_ids=args.gpus)
    else:
        G_net = nn.DataParallel(G_net)
        D_net = nn.DataParallel(D_net)
    if args.autoencode:
        print('Model to train: G = MyUNet(3, 4), D = MyUNet(3, 1)')
    else:
        print('Model to train: G = MyUNet(3, 1), D = MyUNet(3, 1)')
    print('Number of epochs:', args.num_epochs)

    # Initialize optimizer, scheduler, writer
    writer = SummaryWriter(tb_dir)
    optimizer_G = torch.optim.Adam(G_net.parameters(), lr=2e-4)
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=2e-4)
    last_epoch = min(get_last_epoch(save_path_G), get_last_epoch(save_path_D))
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.99)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.99)

    # last_epoch != -1 in scheduler caused crash, so simply call step as needed
    for i in range(last_epoch + 1):
        scheduler_G.step()
        scheduler_D.step()

    # Start training
    G_net, D_net = train_val_epochs(G_net, D_net, train_loader, val_loader, device, writer, imgs_dir,
                                    optimizer_G, optimizer_D, scheduler_G, scheduler_D, args.num_epochs,
                                    save_path_G, save_path_D, args.autoencode, args.back_real)

    writer.close()
    print('Done!')
