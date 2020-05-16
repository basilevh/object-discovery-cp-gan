# Basile Van Hoorick, March 2020
# Common code for PyTorch implementation of Copy-Pasting GAN

import copy
import itertools
import numpy as np
import os, platform, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from cpgan_data import *
from cpgan_model import *


def create_mask_gaussian_filter(blur_sigma):
    bs_round = int(blur_sigma)
    kernel_size = bs_round * 2 + 1
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1.0) / 2.0
    variance = blur_sigma ** 2.0
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    # don't repeat; no RGB
    gaussian_filter = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=bs_round, groups=1, bias=False)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def optimize_masks(masks_torch, blur_mask_sigma, select_biggest):
    # masks_torch: (B, 1, H, W)

    if blur_mask_sigma > 0.0:
        gaussian_filter = create_mask_gaussian_filter(blur_mask_sigma)
        used_masks = gaussian_filter(masks_torch)
    else:
        used_masks = masks_torch

    return used_masks


def make_grid_model_mbatch(G_net, autoencode, mbatch, device, hard_thres=-1, max_rows=32,
                           show_gt=True, blur_mask_sigma=0.0, select_biggest=False):
    # Do exactly what validation does
    fores = mbatch['fore'].to(device)
    backs = mbatch['back'].to(device)
    irrels = mbatch['irrel'].to(device)
    gfake_masks = mbatch['gfake_mask'].to(device)
    comps_gfake = mbatch['comp_gfake'].to(device)
    if 'mask' in mbatch and mbatch['mask'] is not None:
        true_masks = mbatch['mask'].to(device)
        object_cnts = mbatch['object_cnt'].to(device)
    else:
        true_masks = None
    mb_size, width, height = fores.shape[0], fores.shape[3], fores.shape[2]
    show_gt = show_gt and true_masks is not None

    # Generate masks (plus potentially reconstructed object)
    pred_masks, _ = G_net(fores)
    if autoencode:
        pred_recons = pred_masks[:, 1:] # pick last 3 out of 4 channels: (B, 3, H, W)
    else:
        pred_recons = None
    pred_masks = pred_masks[:, 0:1] # retain dimensions: (B, 1, H, W)

    # Optimize predicted mask according to given parameters
    if blur_mask_sigma > 0.0 or select_biggest:
        pred_masks = optimize_masks(pred_masks, blur_mask_sigma, select_biggest)
    if hard_thres > 0:
        pred_masks = (pred_masks > hard_thres).float() # force binary

    comps = copy_paste(pred_recons if autoencode else fores, pred_masks, backs)
    comps_irrel = copy_paste(irrels, pred_masks, backs)
    
    # Apply visual border zeroing (borders are often white but not used in copy paste anyway)
    pred_masks = apply_border_zero(pred_masks)

    # Interleave images into rows so that every column is of one type
    mb_size, width, height = fores.shape[0], fores.shape[3], fores.shape[2]
    rows = min(mb_size, max_rows)
    stride = 9 if show_gt else 8
    images = torch.zeros((rows * stride, 3, height, width))
    for i in range(rows):
        images[i * stride] = fores[i]
        images[i * stride + 1] = pred_masks[i]
        if show_gt:
            cur_true_mask = true_masks[i].sum(axis=0) # there could be multiple
            cur_true_mask = torch.sqrt(cur_true_mask / (torch.max(cur_true_mask) + 1e-6)) # reveal multiple masks but also boost brightness of single masks
            images[i * stride + 2] = cur_true_mask
#         else:
#             cur_true_mask = torch.ones_like(pred_masks[i]) * 0.25 # dark gray => no ground truth
        images[i * stride + (3 if show_gt else 2)] = backs[i]
        images[i * stride + (4 if show_gt else 3)] = comps[i]
        images[i * stride + (5 if show_gt else 4)] = irrels[i]
        images[i * stride + (6 if show_gt else 5)] = comps_irrel[i]
        images[i * stride + (7 if show_gt else 6)] = gfake_masks[i]
        images[i * stride + (8 if show_gt else 7)] = comps_gfake[i]

    grid = torchvision.utils.make_grid(images, nrow=stride, normalize=True, padding=width // 8)
    return grid


def calculate_loss_mask(mask1, mask2):
    if not(torch.is_tensor(mask2)):
        mask2 = mask2 * torch.ones_like(mask1)
    # Used loss function auto averages over all dimensions
    # Permutation issue => take minimum over two possible targets
    mask2 = mask2.detach()
    loss1 = F.binary_cross_entropy(mask1, mask2)
    loss2 = F.binary_cross_entropy(mask1, 1.0 - mask2)
    return torch.min(loss1, loss2)


def calculate_losses(pred_masks, gfake_masks, autoencode, back_real, fores, pred_recons,
                     scores_back, scores_fore, scores_comp, scores_irrel, scores_gfake,
                     aux_masks_fore, aux_masks_comp, aux_masks_irrel, aux_masks_gfake):
    ''' See Appendix A. '''
    ones_bcast = torch.ones_like(scores_fore)
    zeros_bcast = torch.zeros_like(scores_fore)

    loss_G_fake = -F.binary_cross_entropy(scores_comp, zeros_bcast) # generator wants composite to be real
    loss_G_anti = F.binary_cross_entropy(scores_irrel, zeros_bcast) # generator wants irrelevant composite to be fake
    if autoencode:
        # L1 pixel-wise loss; weight both foreground and reconstruction by predicted copy mask
        loss_G_recon = F.l1_loss(pred_recons * pred_masks, fores * pred_masks) # generator wants to autoencode copied object
    else:
        loss_G_recon = torch.tensor(0.0)
    loss_G = loss_G_fake + loss_G_anti + loss_G_recon

    loss_D_real = F.binary_cross_entropy(scores_fore, 0.75 * ones_bcast) # discriminator must think foreground is real
    loss_D_fake = F.binary_cross_entropy(scores_comp, zeros_bcast) # discriminator must think composite is fake
    loss_D_gfake = F.binary_cross_entropy(scores_gfake, zeros_bcast) # discriminator must think random copy mask composite is fake
    if back_real:
        loss_D_back = F.binary_cross_entropy(scores_back, 0.75 * ones_bcast) # if specified, discriminator must think background is real
    else:
        loss_D_back = torch.tensor(0.0)

    loss_mask_real = calculate_loss_mask(aux_masks_fore, 0) # discriminator must think foreground contains no copied parts
    loss_mask_fake = calculate_loss_mask(aux_masks_comp, pred_masks) # discriminator must mark copied parts in composite
    loss_mask_anti = calculate_loss_mask(aux_masks_irrel, pred_masks) # discriminator must mark copied parts in irrelevant composite
    loss_mask_gfake = calculate_loss_mask(aux_masks_gfake, gfake_masks) # discriminator must mark random copy mask in grounded fake
    
    loss_aux = loss_mask_real + loss_mask_fake + loss_mask_anti + loss_mask_gfake
    loss_D = loss_D_real + loss_D_fake + loss_D_gfake + 0.5 * loss_D_back + 0.1 * loss_aux

    return loss_G, loss_D, \
           loss_G_fake, loss_G_anti, loss_G_recon, \
           loss_D_real, loss_D_fake, loss_D_gfake, loss_D_back, loss_aux, \
           loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake


def is_interesting_epoch(epoch):
    return (epoch <= 50 or epoch % 100 <= 4 or epoch % 100 >= 96)


def train(G_net, D_net, autoencode, back_real, loader, device, writer, imgs_dir, optimizer_G, optimizer_D, epoch):
    start = time.time()
    G_net.train()
    D_net.train()
    
    # Visualize dataset & model examples
    mbatch = next(iter(loader))
    grid = make_grid_model_mbatch(G_net, autoencode, mbatch, device)
    if is_interesting_epoch(epoch):
        writer.add_image('images/train', grid, epoch)
    torchvision.utils.save_image(grid, os.path.join(imgs_dir, str(epoch) + '_train.png'))

    # Initialize stats
    total_loss_G = 0.0
    total_loss_D = 0.0
    total_loss_G_fake = 0.0
    total_loss_G_anti = 0.0
    total_loss_G_recon = 0.0
    total_loss_D_back = 0.0
    total_loss_D_real = 0.0
    total_loss_D_fake = 0.0
    total_loss_D_gfake = 0.0
    total_loss_aux = 0.0
    total_loss_mask_real = 0.0
    total_loss_mask_fake = 0.0
    total_loss_mask_anti = 0.0
    total_loss_mask_gfake = 0.0
    total_score_back = 0.0
    total_score_fore = 0.0
    total_score_comp = 0.0
    total_score_irrel = 0.0
    total_score_gfake = 0.0
    total_samples = 0

    # Loop once over whole dataset
    for mbatch in tqdm(loader):
        fores = mbatch['fore'].to(device)
        backs = mbatch['back'].to(device)
        irrels = mbatch['irrel'].to(device)
        gfake_masks = mbatch['gfake_mask'].to(device)
        comps_gfake = mbatch['comp_gfake'].to(device)
        mb_size, width, height = fores.shape[0], fores.shape[3], fores.shape[2]

        # Generate masks (plus potentially reconstructed object)
        pred_masks, _ = G_net(fores)
        if autoencode:
            pred_recons = pred_masks[:, 1:] # pick last 3 out of 4 channels
            pred_masks = pred_masks[:, 0:1] # retain dimensions
            # Copy reconstructed object rather than foreground
            comps = copy_paste(pred_recons, pred_masks, backs)
        else:
            pred_recons = None
            pred_masks = pred_masks[:, 0:1] # retain dimensions
            comps = copy_paste(fores, pred_masks, backs)
        comps_irrel = copy_paste(irrels, pred_masks, backs)

        # Calculate scores & auxiliary masks
        aux_masks_fore, scores_fore = D_net(fores) # score_fore => 0.5 train, 0.2 val
        aux_masks_comp, scores_comp = D_net(comps) # score_comp => 0.3 train, 0.2 val
        aux_masks_irrel, scores_irrel = D_net(comps_irrel) # score_irrel => 0.15 train, 0.15 val
        aux_masks_gfake, scores_gfake = D_net(comps_gfake) # score_gfake => 0 train, 0 val
        if back_real:
            _, scores_back = D_net(backs)
        else:
            scores_back = torch.tensor(0.0)

        # Calculate losses
        loss_G, loss_D, \
            loss_G_fake, loss_G_anti, loss_G_recon, \
            loss_D_real, loss_D_fake, loss_D_gfake, loss_D_back, loss_aux, \
            loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake = \
        calculate_losses(pred_masks, gfake_masks, autoencode, back_real, fores, pred_recons,
            scores_back, scores_fore, scores_comp, scores_irrel, scores_gfake,
            aux_masks_fore, aux_masks_comp, aux_masks_irrel, aux_masks_gfake)

        # Train either G or D in alternation; both at once caused too many 'freed buffer' issues
        # Paper: first 1000 batches: train D only
        total_batches = epoch * len(loader) + total_samples // mb_size
        cumulative_samples = total_batches * mb_size
        # train_gen = (cumulative_samples >= 256000 and total_batches % 2 == 0)
        train_gen = (cumulative_samples >= 64000 and total_batches % 2 == 0)

        # == Generator ==
        if train_gen:
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        # == Discriminator ==
        else:
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        # Update stats
        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()
        total_loss_G_fake += loss_G_fake.item()
        total_loss_G_anti += loss_G_anti.item()
        total_loss_G_recon += loss_G_recon.item()
        total_loss_D_back += loss_D_back.item()
        total_loss_D_real += loss_D_real.item()
        total_loss_D_fake += loss_D_fake.item()
        total_loss_D_gfake += loss_D_gfake.item()
        total_loss_aux += loss_aux.item()
        total_loss_mask_real += loss_mask_real.item()
        total_loss_mask_fake += loss_mask_fake.item()
        total_loss_mask_anti += loss_mask_anti.item()
        total_loss_mask_gfake += loss_mask_gfake.item()
        total_score_back += torch.sum(scores_back).item()
        total_score_fore += torch.sum(scores_fore).item()
        total_score_comp += torch.sum(scores_comp).item()
        total_score_irrel += torch.sum(scores_irrel).item()
        total_score_gfake += torch.sum(scores_gfake).item()
        total_samples += mb_size

    # Print stats
    loss_G = total_loss_G / total_samples
    loss_D = total_loss_D / total_samples
    loss_G_fake = total_loss_G_fake / total_samples
    loss_G_anti = total_loss_G_anti / total_samples
    loss_G_recon = total_loss_G_recon / total_samples
    loss_D_back = total_loss_D_back / total_samples
    loss_D_real = total_loss_D_real / total_samples
    loss_D_fake = total_loss_D_fake / total_samples
    loss_D_gfake = total_loss_D_gfake / total_samples
    loss_aux = total_loss_aux / total_samples
    loss_mask_real = total_loss_mask_real / total_samples
    loss_mask_fake = total_loss_mask_fake / total_samples
    loss_mask_anti = total_loss_mask_anti / total_samples
    loss_mask_gfake = total_loss_mask_gfake / total_samples
    score_back = total_score_back / total_samples
    score_fore = total_score_fore / total_samples
    score_comp = total_score_comp / total_samples
    score_irrel = total_score_irrel / total_samples
    score_gfake = total_score_gfake / total_samples
    end = time.time()
    print('train took {:d}m {:d}s'.format(int(end - start) // 60, int(end - start) % 60))
    print('loss_G: {:.3f}  loss_D: {:.3f}'.format(loss_G, loss_D))
    print('loss_G_fake: {:.3f}  loss_G_anti: {:.3f}  loss_G_recon: {:.3f}'.format(loss_G_fake, loss_G_anti, loss_G_recon))
    print('loss_D_back: {:.3f}  loss_D_real: {:.3f}  loss_D_fake: {:.3f}  loss_D_gfake: {:.3f}   loss_aux: {:.3f}'.format(loss_D_back, loss_D_real, loss_D_fake, loss_D_gfake, loss_aux))
    print('loss_mask_real: {:.3f}  loss_mask_fake: {:.3f}  loss_mask_anti: {:.3f}  loss_mask_gfake: {:.3f}'.format(loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake))
    print('score_back: {:.3f}  score_fore: {:.3f}  score_comp: {:.3f}  score_irrel: {:.3f}  score_gfake: {:.3f}'.format(score_back, score_fore, score_comp, score_irrel, score_gfake))
    print()

    # Write stats
    writer.add_scalar('loss_G/train', loss_G, epoch)
    writer.add_scalar('loss_D/train', loss_D, epoch)
    writer.add_scalar('loss_G_fake/train', loss_G_fake, epoch)
    writer.add_scalar('loss_G_anti/train', loss_G_anti, epoch)
    writer.add_scalar('loss_G_recon/train', loss_G_recon, epoch)
    writer.add_scalar('loss_D_back/train', loss_D_back, epoch)
    writer.add_scalar('loss_D_real/train', loss_D_real, epoch)
    writer.add_scalar('loss_D_fake/train', loss_D_fake, epoch)
    writer.add_scalar('loss_D_gfake/train', loss_D_gfake, epoch)
    writer.add_scalar('loss_aux/train', loss_aux, epoch)
    writer.add_scalar('loss_mask_real/train', loss_mask_real, epoch)
    writer.add_scalar('loss_mask_fake/train', loss_mask_fake, epoch)
    writer.add_scalar('loss_mask_anti/train', loss_mask_anti, epoch)
    writer.add_scalar('loss_mask_gfake/train', loss_mask_gfake, epoch)
    writer.add_scalar('score_back/train', score_back, epoch)
    writer.add_scalar('score_fore/train', score_fore, epoch)
    writer.add_scalar('score_comp/train', score_comp, epoch)
    writer.add_scalar('score_irrel/train', score_irrel, epoch)
    writer.add_scalar('score_gfake/train', score_gfake, epoch)

    return loss_G, loss_D, \
           loss_G_fake, loss_G_anti, loss_G_recon, \
           loss_D_back, loss_D_real, loss_D_fake, loss_D_gfake, loss_aux, \
           loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake, \
           score_back, score_fore, score_comp, score_irrel, score_gfake


# https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
def get_powerset(iterable):
    ''' Example: powerset([1, 2, 3]) --> [(1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)] '''
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1)))


def is_mask_success(true_masks, object_cnt, pred_mask, min_iou=0.5):
    '''
    Given a collection of ground truth masks for each individual object,
    calculates whether the predicted mask matches any possible subset.
    A successful case contributes positively to the Object Discovery Performance (ODP).
    '''
    true_masks = true_masks.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    true_masks = [true_masks[i] for i in range(object_cnt)] # convert to list
    tm_powerset = get_powerset(true_masks)

    # Loop over all possible subsets of ground truth objects
    for tm_subset in tm_powerset:
        true_mask = (np.array(tm_subset).sum(axis=0) > 0.5)
        intersection = np.sum(true_mask * (pred_mask > 0.5))
        union = np.sum((true_mask + pred_mask) > 0.5)

        # Check IOU
        iou = intersection / union
        if iou >= min_iou:
            return True

    # The predicted mask does not match any subset whatsoever
    return False


def get_mask_success_count(true_masks, object_cnts, pred_masks, min_iou=0.5):
    mb_size = true_masks.shape[0]
    total_valid = 0
    for i in range(mb_size):
        cur_valid = is_mask_success(true_masks[i], object_cnts[i].item(), pred_masks[i])
        total_valid += cur_valid
    return total_valid


def val(G_net, D_net, autoencode, back_real, loader, device, writer, imgs_dir, epoch):
    start = time.time()
    G_net.eval()
    D_net.eval()
    autoencode = False # DO NOT attempt to reconstruct at evaluation time!
    
    # Visualize dataset & model examples
    mbatch = next(iter(loader))
    grid = make_grid_model_mbatch(G_net, autoencode, mbatch, device)
    if is_interesting_epoch(epoch):
        writer.add_image('images/val', grid, epoch)
    torchvision.utils.save_image(grid, os.path.join(imgs_dir, str(epoch) + '_val.png'))

    # Initialize stats
    total_valid = 0
    total_loss_G = 0.0
    total_loss_D = 0.0
    total_loss_G_fake = 0.0
    total_loss_G_anti = 0.0
    total_loss_D_back = 0.0
    total_loss_D_real = 0.0
    total_loss_D_fake = 0.0
    total_loss_D_gfake = 0.0
    total_loss_aux = 0.0
    total_loss_mask_real = 0.0
    total_loss_mask_fake = 0.0
    total_loss_mask_anti = 0.0
    total_loss_mask_gfake = 0.0
    total_score_back = 0.0
    total_score_fore = 0.0
    total_score_comp = 0.0
    total_score_irrel = 0.0
    total_score_gfake = 0.0
    total_samples = 0

    with torch.no_grad():
        for mbatch in tqdm(loader):
            fores = mbatch['fore'].to(device)
            backs = mbatch['back'].to(device)
            irrels = mbatch['irrel'].to(device)
            gfake_masks = mbatch['gfake_mask'].to(device)
            comps_gfake = mbatch['comp_gfake'].to(device)
            if 'mask' in mbatch and mbatch['mask'] is not None:
                true_masks = mbatch['mask'].to(device)
                object_cnts = mbatch['object_cnt'].to(device)
            else:
                true_masks = None
            mb_size, width, height = fores.shape[0], fores.shape[3], fores.shape[2]

            # Generate masks (plus potentially reconstructed object)
            pred_masks, _ = G_net(fores)
            if autoencode:
                pred_recons = pred_masks[:, 1:] # pick last 3 out of 4 channels
                pred_masks = pred_masks[:, 0:1] # retain dimensions
                # Copy reconstructed object rather than foreground
                comps = copy_paste(pred_recons, pred_masks, backs)
            else:
                pred_recons = None
                pred_masks = pred_masks[:, 0:1] # retain dimensions
                comps = copy_paste(fores, pred_masks, backs)
            comps_irrel = copy_paste(irrels, pred_masks, backs)

            # Calculate scores & auxiliary masks
            aux_masks_fore, scores_fore = D_net(fores)
            aux_masks_comp, scores_comp = D_net(comps)
            aux_masks_irrel, scores_irrel = D_net(comps_irrel)
            aux_masks_gfake, scores_gfake = D_net(comps_gfake)
            if back_real:
                _, scores_back = D_net(backs)
            else:
                scores_back = torch.tensor(0.0)

            # Calculate losses
            loss_G, loss_D, \
                loss_G_fake, loss_G_anti, loss_G_recon, \
                loss_D_real, loss_D_fake, loss_D_gfake, loss_D_back, loss_aux, \
                loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake = \
            calculate_losses(pred_masks, gfake_masks, autoencode, back_real, fores, pred_recons,
                scores_back, scores_fore, scores_comp, scores_irrel, scores_gfake,
                aux_masks_fore, aux_masks_comp, aux_masks_irrel, aux_masks_gfake)

            # Update stats
            if true_masks is not None:
                total_valid += get_mask_success_count(true_masks, object_cnts, pred_masks)
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            total_loss_G_fake += loss_G_fake.item()
            total_loss_G_anti += loss_G_anti.item()
            total_loss_D_back += loss_D_back.item()
            total_loss_D_real += loss_D_real.item()
            total_loss_D_fake += loss_D_fake.item()
            total_loss_D_gfake += loss_D_gfake.item()
            total_loss_aux += loss_aux.item()
            total_loss_mask_real += loss_mask_real.item()
            total_loss_mask_fake += loss_mask_fake.item()
            total_loss_mask_anti += loss_mask_anti.item()
            total_loss_mask_gfake += loss_mask_gfake.item()
            total_score_back += torch.sum(scores_back).item()
            total_score_fore += torch.sum(scores_fore).item()
            total_score_comp += torch.sum(scores_comp).item()
            total_score_irrel += torch.sum(scores_irrel).item()
            total_score_gfake += torch.sum(scores_gfake).item()
            total_samples += mb_size

    # Print stats
    odp = total_valid / total_samples
    loss_G = total_loss_G / total_samples
    loss_D = total_loss_D / total_samples
    loss_G_fake = total_loss_G_fake / total_samples
    loss_G_anti = total_loss_G_anti / total_samples
    loss_D_back = total_loss_D_back / total_samples
    loss_D_real = total_loss_D_real / total_samples
    loss_D_fake = total_loss_D_fake / total_samples
    loss_D_gfake = total_loss_D_gfake / total_samples
    loss_aux = total_loss_aux / total_samples
    loss_mask_real = total_loss_mask_real / total_samples
    loss_mask_fake = total_loss_mask_fake / total_samples
    loss_mask_anti = total_loss_mask_anti / total_samples
    loss_mask_gfake = total_loss_mask_gfake / total_samples
    score_back = total_score_back / total_samples
    score_fore = total_score_fore / total_samples
    score_comp = total_score_comp / total_samples
    score_irrel = total_score_irrel / total_samples
    score_gfake = total_score_gfake / total_samples
    end = time.time()
    print('val took {:d}m {:d}s'.format(int(end - start) // 60, int(end - start) % 60))
    print('ODP: {:.3f}  loss_G: {:.3f}  loss_D: {:.3f}'.format(odp, loss_G, loss_D))
    print('loss_G_fake: {:.3f}  loss_G_anti: {:.3f}'.format(loss_G_fake, loss_G_anti))
    print('loss_D_back: {:.3f}  loss_D_real: {:.3f}  loss_D_fake: {:.3f}  loss_D_gfake: {:.3f}   loss_aux: {:.3f}'.format(loss_D_back, loss_D_real, loss_D_fake, loss_D_gfake, loss_aux))
    print('loss_mask_real: {:.3f}  loss_mask_fake: {:.3f}  loss_mask_anti: {:.3f}  loss_mask_gfake: {:.3f}'.format(loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake))
    print('score_back: {:.3f}  score_fore: {:.3f}  score_comp: {:.3f}  score_irrel: {:.3f}  score_gfake: {:.3f}'.format(score_back, score_fore, score_comp, score_irrel, score_gfake))
    print()

    # Write stats
    writer.add_scalar('ODP/val', odp, epoch)
    writer.add_scalar('loss_G/val', loss_G, epoch)
    writer.add_scalar('loss_D/val', loss_D, epoch)
    writer.add_scalar('loss_G_fake/val', loss_G_fake, epoch)
    writer.add_scalar('loss_G_anti/val', loss_G_anti, epoch)
    writer.add_scalar('loss_D_back/val', loss_D_back, epoch)
    writer.add_scalar('loss_D_real/val', loss_D_real, epoch)
    writer.add_scalar('loss_D_fake/val', loss_D_fake, epoch)
    writer.add_scalar('loss_D_gfake/val', loss_D_gfake, epoch)
    writer.add_scalar('loss_aux/val', loss_aux, epoch)
    writer.add_scalar('loss_mask_real/val', loss_mask_real, epoch)
    writer.add_scalar('loss_mask_fake/val', loss_mask_fake, epoch)
    writer.add_scalar('loss_mask_anti/val', loss_mask_anti, epoch)
    writer.add_scalar('loss_mask_gfake/val', loss_mask_gfake, epoch)
    writer.add_scalar('score_back/val', score_back, epoch)
    writer.add_scalar('score_fore/val', score_fore, epoch)
    writer.add_scalar('score_comp/val', score_comp, epoch)
    writer.add_scalar('score_irrel/val', score_irrel, epoch)
    writer.add_scalar('score_gfake/val', score_gfake, epoch)

    return odp, loss_G, loss_D, \
           loss_G_fake, loss_G_anti, \
           loss_D_back, loss_D_real, loss_D_fake, loss_D_gfake, loss_aux, \
           loss_mask_real, loss_mask_fake, loss_mask_anti, loss_mask_gfake, \
           score_back, score_fore, score_comp, score_irrel, score_gfake


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_last_epoch(model_path):
    # Load existing model if possible (in case training was interrupted) & return path
    for try_epoch in range(5000, -1, -1):
        load_path = model_path.format(try_epoch)
        if os.path.exists(load_path):
            print('Found existing model at epoch: {}'.format(try_epoch + 1))
            return try_epoch
    return -1


def train_val_epochs(G_net, D_net, train_loader, val_loader, device, writer, imgs_dir, optimizer_G, optimizer_D,
                     scheduler_G, scheduler_D, num_epochs, path_G, path_D, autoencode, back_real):
    # Create directories if needed
    if not(os.path.exists(Path(path_G).parent)):
        os.makedirs(Path(path_G).parent)
    if not(os.path.exists(Path(path_D).parent)):
        os.makedirs(Path(path_D).parent)
    if not(os.path.exists(imgs_dir)):
        os.makedirs(imgs_dir)

    # Load existing model if possible (in case training was interrupted)
    start_epoch_G = get_last_epoch(path_G) + 1
    start_epoch_D = get_last_epoch(path_D) + 1
    start_epoch = min(start_epoch_G, start_epoch_D)
    if start_epoch > 0:
        G_net.load_state_dict(torch.load(path_G.format(start_epoch - 1), map_location=device))
        D_net.load_state_dict(torch.load(path_D.format(start_epoch - 1), map_location=device))

    best_G_wts = copy.deepcopy(G_net.state_dict())
    best_D_wts = copy.deepcopy(D_net.state_dict())
    best_odp = 0

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}...'.format(epoch + 1, num_epochs))
        writer.add_scalar('Learning rate', get_learning_rate(optimizer_G), epoch)

        # Train & validate
        train(G_net, D_net, autoencode, back_real, train_loader, device, writer, imgs_dir, optimizer_G, optimizer_D, epoch)
        odp = val(G_net, D_net, autoencode, back_real, val_loader, device, writer, imgs_dir, epoch)[0]
        scheduler_G.step()
        scheduler_D.step()

        # Store checkpoint
        torch.save(G_net.state_dict(), path_G.format(epoch))
        torch.save(D_net.state_dict(), path_D.format(epoch))

        # Keep track of best weights (judging by Object Discovery Performance)
        if odp > best_odp:
            best_G_wts = copy.deepcopy(G_net.state_dict())
            best_D_wts = copy.deepcopy(D_net.state_dict())
            best_odp = odp
            print('New best model so far with ODP: {:.3f}'.format(best_odp))

    # Load best model weights
    G_net.load_state_dict(best_G_wts)
    D_net.load_state_dict(best_D_wts)
    print('Loaded best model with ODP: {:.3f}'.format(best_odp))
    print('Done!')

    return G_net, D_net
