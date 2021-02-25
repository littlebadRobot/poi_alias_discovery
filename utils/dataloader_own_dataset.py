
import math
import os
import random
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils.utils import *

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class SiameseDataset(Dataset):
    def __init__(self, image_size, pos_path, neg_path):
        super(SiameseDataset, self).__init__()

        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        
        self.train_lines = []
        self.train_labels = []

        self.val_lines = []
        self.val_labels = []
        self.types = 0

        self.pos_pairs = load_pickle(pos_path)
        self.pos_pairs = np.array(self.pos_pairs,dtype=np.object)
        self.neg_pairs = load_pickle(neg_path)
        self.neg_pairs = np.array(self.neg_pairs,dtype=np.object)
        self.num_pos = len(self.pos_pairs)
        self.num_neg = len(self.neg_pairs)
        random.seed(1)
        shuffle_index = np.arange(self.num_pos, dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        self.pos_pairs = self.pos_pairs[shuffle_index]
        random.seed(1)
        shuffle_index = np.arange(self.num_neg, dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        self.neg_pairs = self.neg_pairs[shuffle_index]
        
        self.pos_pointer = 0
        self.neg_pointer = 0

    def __len__(self):
        return max(self.num_neg, self.num_pos)

    def load_dataset(self):
        train_path = os.path.join(self.dataset_path, 'images_background')
        for character in os.listdir(train_path):
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                self.train_lines.append(os.path.join(character_path, image))
                self.train_labels.append(self.types)
            self.types += 1

        random.seed(1)
        shuffle_index = np.arange(len(self.train_lines), dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        self.train_lines = np.array(self.train_lines,dtype=np.object)
        self.train_labels = np.array(self.train_labels)
        self.train_lines = self.train_lines[shuffle_index]
        self.train_labels = self.train_labels[shuffle_index]
        
        self.val_lines = self.train_lines[self.num_train:]
        self.val_labels = self.train_labels[self.num_train:]
    
        self.train_lines = self.train_lines[:self.num_train]
        self.train_labels = self.train_labels[:self.num_train]

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, flip_signal=False):
        image = image.convert("RGB")

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.75,1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        
        # flip image or not
        flip = rand()<.5
        if flip and flip_signal: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (255,255,255))

        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand()<.5
        if rotate: 
            angle=np.random.randint(-5,5)
            a,b=w/2,h/2
            M=cv2.getRotationMatrix2D((a,b),angle,1)
            image=cv2.warpAffine(np.array(image),M,(w,h),borderValue=[255,255,255]) 

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        if self.channel==1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        return image_data

    def _convert_path_list_to_images_and_labels(self):
        number_of_pairs = 2
        pairs_of_images = [np.zeros((number_of_pairs, self.channel, self.image_height, self.image_width)) for i in range(2)] + \
        [np.zeros((number_of_pairs, self.channel, 5, 5)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):

            if (pair + 1) % 2 == 0:
                random_idx = random.randint(1,self.num_pos-1)
                pair_data = self.pos_pairs[random_idx]
                self.pos_pointer += 1
                if self.pos_pointer == self.num_pos:
                    random.seed(1)
                    shuffle_index = np.arange(self.num_pos, dtype=np.int32)
                    shuffle(shuffle_index)
                    random.seed(None)
                    self.pos_pairs = self.pos_pairs[shuffle_index]
                    self.pos_pointer = 0
            else:
                random_idx = random.randint(1,self.num_neg-1)
                pair_data = self.neg_pairs[random_idx]
                self.neg_pointer += 1
                if self.neg_pointer == self.num_neg:
                    random.seed(1)
                    shuffle_index = np.arange(self.num_neg, dtype=np.int32)
                    shuffle(shuffle_index)
                    random.seed(None)
                    self.neg_pairs = self.neg_pairs[shuffle_index]
                    self.neg_pointer = 0

            image = pair_data[0][0].astype(np.float64) #             image = image / image.std(ddof=1) - image.mean()
            if image.std()!=0:
                image = image / image.std() - image.mean()
            if self.channel == 1:
                pairs_of_images[0][pair, 0, :, :] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image

            image = pair_data[1][0].astype(np.float64) #             image = image / image.std(ddof=1) - image.mean()
            if image.std()!=0:
                image = image / image.std() - image.mean()
            if self.channel == 1:
                pairs_of_images[1][pair, 0, :, :] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image
            
#             print(pair_data[0][1])
#             input()
            image = pair_data[0][1].astype(np.float64) #             image = image / image.std(ddof=1) - image.mean()
            if image.std()!=0:
                image = image / image.std() - image.mean()
            if self.channel == 1:
                pairs_of_images[2][pair, 0, :, :] = image
            else:
                pairs_of_images[2][pair, :, :, :] = image

            image = pair_data[1][1].astype(np.float64) #             image = image / image.std(ddof=1) - image.mean()
            if image.std()!=0:
                image = image / image.std() - image.mean()
            if self.channel == 1:
                pairs_of_images[3][pair, 0, :, :] = image
            else:
                pairs_of_images[3][pair, :, :, :] = image

            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        pairs_of_images[2][:, :, :, :] = pairs_of_images[2][random_permutation, :, :, :]
        pairs_of_images[3][:, :, :, :] = pairs_of_images[3][random_permutation, :, :, :]

        return pairs_of_images, labels

    def __getitem__(self, index):
        images, labels = self._convert_path_list_to_images_and_labels()
        return images, labels

# collate_fn used in DataLoader
def dataset_collate(batch):
    left_images = []
    right_images = []
    gl_left_images = []
    gl_right_images = []
    labels = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            gl_left_images.append(pair_imgs[2][i])
            gl_right_images.append(pair_imgs[3][i])

            labels.append(pair_labels[i])
    
    return np.array([left_images, right_images]), np.array([gl_left_images, gl_right_images]), np.array(labels)

