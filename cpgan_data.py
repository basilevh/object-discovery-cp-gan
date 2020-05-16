# Basile Van Hoorick, March 2020
# Common code for PyTorch implementation of Copy-Pasting GAN

import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os, platform, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from tqdm import tqdm


def read_image_robust(img_path, monochromatic=False):
    ''' Returns an image that meets conditions along with a success flag, in order to avoid crashing. '''
    try:
        # image = plt.imread(img_path).copy()
        image = np.array(Image.open(img_path)).copy() # always uint8
        success = True
        if np.any(np.array(image.strides) < 0):
            success = False # still negative stride
        elif not(monochromatic) and (image.ndim != 3 or image.shape[2] != 3):
            success = False # not RGB
        elif monochromatic:
        #     width, height = image.shape[1], image.shape[0]
        #     image = np.broadcast_to(x[:, :, np.newaxis], (height, width, 3))
            image = image[:, :, np.newaxis] # one channel <=> only one ground truth

    except IOError:
        # Probably corrupt file
        image = None
        success = False

    return image, success


def paint_squares(image, noisy=False, channels=10):
    '''
    Paints one or more squares at random locations to create an artificial foreground image.
    Generates multiple associated ground truth masks; one per object.
    '''
    width, height = image.shape[1], image.shape[0]
    image = image.copy() # do not overwrite background
    object_count = np.random.randint(1, 5) # [1, 4] inclusive
    masks = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(object_count):
        sq_w, sq_h = 9, 9
        x1 = np.random.randint(0, width - sq_w + 1)
        y1 = np.random.randint(0, height - sq_h + 1)
        x2 = x1 + sq_w
        y2 = y1 + sq_h
        masks[y1:y2, x1:x2, i] = 255

        if not(noisy):
            # Pick one fixed (not necessarily saturated) color for the whole square
            clr = np.random.randint(0, 256, 3)
            image[y1:y2, x1:x2] = clr

        else:
            # Pick a random fully saturated (extremal) color for every pixel
            image[y1:y2, x1:x2] = np.random.choice([0, 255], (sq_h, sq_w, 3))

    return image, masks, object_count


def create_random_gfake_mask(width, height):
    ''' See Appendix D. '''
    x0, y0 = np.random.rand(2) * 0.8 + 0.1
    num_verts = np.random.randint(4, 7)
    # TODO possible improvement: allow up to more vertices?
    # TODO possible improvement: encourage convex (currently many "sharp" objects)
    radii = np.random.rand(num_verts) * 0.4 + 0.1
    # radii = np.random.rand(num_verts) * 0.8 + 0.2 # TODO: not very clear from paper
    angles = np.sort(np.random.rand(num_verts)) * 2.0 * np.pi
    poly_polar = list(zip(radii, angles))
    poly_cart = [(int(width * (x0 + r * np.cos(a)) / 1),
                  int(height * (y0 + r * np.sin(a)) / 1)) for (r, a) in poly_polar]
    # poly_cart = [(x1, y1), (x2, y2), ...]
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(poly_cart, outline=1, fill=255)
    mask = np.array(img, dtype='uint8')
    assert(mask.shape == (height, width))
    return mask


def apply_border_zero(masks):
    ndim = len(masks.shape)
    if ndim == 2:
        masks[0, :] = 0
        masks[-1, :] = 0
        masks[:, 0] = 0
        masks[:, -1] = 0
    elif ndim == 3:
        masks[:, 0, :] = 0
        masks[:, -1, :] = 0
        masks[:, :, 0] = 0
        masks[:, :, -1] = 0
    elif ndim == 4:
        masks[:, :, 0, :] = 0
        masks[:, :, -1, :] = 0
        masks[:, :, :, 0] = 0
        masks[:, :, :, -1] = 0
    else:
        raise Exception('Mask has too many dimensions')
    return masks


def copy_paste(fores, masks, backs, border_zero=True):
    # TODO possible improvement: poisson blending
#     if hard_thres > 0:
#         used_masks = (masks > hard_thres).float() # force binary
#     else:
    used_masks = masks.clone()

    # Border zeroing implemented in April 2020
    if border_zero:
        used_masks = apply_border_zero(used_masks)

    return used_masks * fores + (1.0 - used_masks) * backs


class MyCopyPasteDataset(Dataset):
    '''
    Custom dataset class with foreground, background, and optional mask folders as image sources.
    Only one object may appear per image, since the object count is not kept track of.
    Returns irrelevant foreground anti-shortcuts as well. Enforces color (RGB) images.
    '''

    def __init__(self, fore_dir, back_dir, mask_dir=None, rand_horz_flip=True, post_resize=-1, center_crop=False):
        self.fore_dir = fore_dir
        self.back_dir = back_dir
        self.rand_horz_flip = rand_horz_flip
        if post_resize <= 0:
            self.post_tf = transforms.ToTensor() # converts [0, 255] to [0.0, 1.0]
        elif center_crop:
            # Resize + square center crop
            self.post_tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(post_resize),
                transforms.CenterCrop(post_resize),
                transforms.ToTensor()
            ])
        else:
            # Resize both dimensions, possibly distorting the images
            self.post_tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((post_resize, post_resize)),
                transforms.ToTensor()
            ])
        self.has_masks = (mask_dir is not None)

        # Load all file paths; file names must be the same across all 2 or 3 given directories
        # self.all_fore_files = []
        # self.all_mask_files = []
        # self.all_back_files = []
        # for fn in os.listdir(fore_dir):
        #     fore_fp = os.path.join(fore_dir, fn)
        #     if os.path.isfile(fore_fp):
        #         back_fp = os.path.join(back_dir, fn)
        #         assert(os.path.isfile(back_fp))
        #         self.all_fore_files.append(fore_fp)
        #         self.all_back_files.append(back_fp)
        #         if self.has_masks:
        #             mask_fp = os.path.join(mask_dir, fn)
        #             assert(os.path.isfile(mask_fp))
        #             self.all_mask_files.append(mask_fp)

        # Load all file paths; file names must be the same across foreground and segmentation masks
        self.all_fore_files = []
        self.all_mask_files = []
        self.all_back_files = []

        for fn in os.listdir(fore_dir):
            fore_fp = os.path.join(fore_dir, fn)
            self.all_fore_files.append(fore_fp)
            if self.has_masks:
                mask_fp_jpg = os.path.join(mask_dir, fn[:-4] + '.jpg')
                mask_fp_png = os.path.join(mask_dir, fn[:-4] + '.png')
                if os.path.isfile(mask_fp_jpg):
                    self.all_mask_files.append(mask_fp_jpg)
                elif os.path.isfile(mask_fp_png):
                    self.all_mask_files.append(mask_fp_png)
                else:
                    raise Exception('No matching mask file found for ' + fore_fp)

        for fn in os.listdir(back_dir):
            back_fp = os.path.join(back_dir, fn)
            self.all_back_files.append(back_fp)
        
        self.fore_count = len(self.all_fore_files)
        self.back_count = len(self.all_back_files)
        print('Image file count: ' + str(self.fore_count) + ' foreground, ' + str(self.back_count) + ' background, has masks: ' + str(self.has_masks))
            

    def __len__(self):
        return self.fore_count


    def __getitem__(self, idx):
        # Force randomness (especially if num_workers > 0)
        np.random.seed(idx + int((time.time() * 654321) % 123456))

        # Read random pair of images from file system
        success = False
        while not(success):
            file_idx = np.random.choice(self.fore_count)
            fp = self.all_fore_files[file_idx]
            fore, success = read_image_robust(fp)
            if not(success):
                continue
            if self.has_masks:
                fp = self.all_mask_files[file_idx]
                mask, success = read_image_robust(fp, monochromatic=True)
                assert(success) # must match fore
                # mask = ((mask > 0) * 255.0).astype('uint8') # convert soft masks to hard
            else:
                mask = None
                
            # Read random background image
        success = False
        while not(success):
            file_idx2 = np.random.choice(self.back_count)
            fp = self.all_back_files[file_idx2]
            back, success = read_image_robust(fp)

        # Read irrelevant foreground image
        success = False
        while not(success):
            file_idx3 = np.random.choice(self.fore_count)
            if file_idx3 == file_idx:
                continue # try again, cannot pick same image
            fp = self.all_fore_files[file_idx3]
            irrel, success = read_image_robust(fp)
            
        # Transform foregrounds (+ masks) and backgrounds
        # NOTE: identical random choices must be made for some images
        if self.rand_horz_flip:
            if np.random.rand() < 0.5:
                fore = fore[:, ::-1, :].copy()
                if self.has_masks:
                    mask = mask[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                irrel = irrel[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                back = back[:, ::-1, :].copy()
        fore = self.post_tf(fore)
        irrel = self.post_tf(irrel)
        back = self.post_tf(back)
        if self.has_masks:
            mask = self.post_tf(mask)

        # Verify sizes
        assert(fore.shape[1:] == irrel.shape[1:])
        assert(fore.shape[1:] == back.shape[1:])
        if self.has_masks:
            assert(fore.shape[1:] == mask.shape[1:])

        # Create grounded fake mask and composite
        width, height = fore.shape[2], fore.shape[1] # fore is (C, H, W)
        gfake_mask = self.post_tf(create_random_gfake_mask(width, height))
        comp_gfake = copy_paste(fore, gfake_mask, back)

        # Construct dictionary; object count is unknown
        result = {'fore': fore, 'back': back, 'irrel': irrel, 'object_cnt': 1, 'gfake_mask': gfake_mask, 'comp_gfake': comp_gfake}
        if self.has_masks:
            result['mask'] = mask # don't set None, otherwise crash

        return result


class MySquaresDataset(Dataset):
    '''
    Custom dataset class with just a collection of background images as source.
    One or more artificial objects are painted to create a foreground, keeping track of object count.
    Returns irrelevant foreground anti-shortcuts as well. Enforces color (RGB) images.
    '''

    def __init__(self, back_dir, rand_horz_flip=True, noisy=False, max_objects=10):
        self.back_dir = back_dir
        self.rand_horz_flip = rand_horz_flip
        self.post_tf = transforms.ToTensor() # converts [0, 255] to [0.0, 1.0]
        self.noisy = noisy
        self.max_objects = max_objects

        # Load all file paths; file names must be the same across all 2 or 3 given directories
        self.all_back_files = []
        for fn in os.listdir(back_dir):
            back_fp = os.path.join(back_dir, fn)
            self.all_back_files.append(back_fp)
        self.file_count = len(self.all_back_files)

        print('Image file count: ' + str(self.file_count) + ', noisy: ' + str(self.noisy) + ', max objects: ' + str(self.max_objects))
            

    def __len__(self):
        return self.file_count


    def __getitem__(self, idx):
        # Read a random triplet (relevant + background + irrelevant) of non-overlapping backgrounds from file system
        success = False
        while not(success):
            file_idx = np.random.choice(self.file_count)
            fp = self.all_back_files[file_idx]
            fore, success = read_image_robust(fp)
        success = False
        while not(success):
            file_idx2 = np.random.choice(self.file_count)
            if file_idx2 == file_idx:
                continue # try again, cannot pick same image
            fp = self.all_back_files[file_idx2]
            back, success = read_image_robust(fp)
        success = False
        while not(success):
            file_idx3 = np.random.choice(self.file_count)
            if file_idx3 == file_idx or file_idx3 == file_idx2:
                continue # try again, cannot pick same image
            fp = self.all_back_files[file_idx3]
            irrel, success = read_image_robust(fp)

        # Create corresponding foregrounds and masks; leave actual background unchanged
        fore, masks, object_cnt = paint_squares(fore, noisy=self.noisy, channels=self.max_objects)
        irrel, _, _ = paint_squares(irrel, noisy=self.noisy, channels=self.max_objects)
            
        # Transform foregrounds (+ masks) and backgrounds
        # NOTE: identical random choices must be made for some images
        if self.rand_horz_flip:
            if np.random.rand() < 0.5:
                fore = fore[:, ::-1, :].copy()
                masks = masks[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                irrel = irrel[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                back = back[:, ::-1, :].copy()
        fore = self.post_tf(fore)
        masks = self.post_tf(masks)
        irrel = self.post_tf(irrel)
        back = self.post_tf(back)

        # Create grounded fake mask and composite
        width, height = fore.shape[2], fore.shape[1] # fore is (C, H, W)
        gfake_mask = self.post_tf(create_random_gfake_mask(width, height))
        comp_gfake = copy_paste(fore, gfake_mask, back)

        # Construct dictionary
        result = {'fore': fore, 'back': back, 'irrel': irrel, 'mask': masks, 'object_cnt': object_cnt, 'gfake_mask': gfake_mask, 'comp_gfake': comp_gfake}
        return result
