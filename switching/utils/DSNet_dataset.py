import numpy as np
import os
import yaml
import math
import pickle
import cv2

from itertools import permutations 

class Dataset:

    def __init__(self, cfg, mode, fr_num, camera_num, batch_size, split, iter_method='sample', frame_size=(224, 224, 3), split_ratio=0.8, shuffle=False, overlap=0, num_sample=20000, sub_sample=5, setting_id=0):
        self.cfg = cfg
        self.mode = mode
        self.split = split
        self.fr_num = fr_num
        self.shuffle = shuffle
        self.overlap = overlap
        self.num_sample = num_sample
        self.iter_method = iter_method
        
        self.camera_num = camera_num
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.sub_sample = sub_sample

        self.base_folder = './datasets'
        self.image_folder = os.path.join(self.base_folder, 'frames')
        self.label_folder = os.path.join(self.base_folder, 'labels')

        self.ignore_index = cfg.ignore_index

        # get take names
        if self.split == 'sequence':
            if mode == 'train' or mode == 'val':
                self.takes = self.cfg.takes['train']
                self.ignore_index = 
            else:
                self.takes = self.cfg.takes[mode]
        elif self.split == 'surgery':
            _takes = self.cfg.takes['train']
            perm = list(permutations(range(len(_takes)), 2))
            t_ind1, t_ind2 = perm[setting_id]
            self.takes = []
            if mode == 'train':
                for idx, t in enumerate(_takes):
                    if idx != t_ind1 and idx != t_ind2:
                        self.takes.append(t)
            else:
                for idx, t in enumerate(_takes):
                    if idx == t_ind1 or idx == t_ind2:
                        self.takes.append(t)
        # iterator specific
        self.sample_count = None
        self.take_indices = None
        self.cur_ind = None
        self.cur_tid = None
        self.cur_fr = None
        self.fr_lb = None
        self.fr_ub = None
        self.seq_len = []

        self.labels = []
        self.load_labels()

        if self.split == 'sequence':
            self.lookup_table = []
            for _len in self.seq_len:
                original = np.arange(_len)
                permute = np.arange(_len)
                if setting_id == 0:
                    permute = permute
                else:
                    permute[int(_len*0.8):_len], permute[int(_len*0.2*setting_id):int(_len*0.2*(setting_id+1))] = \
                        permute[int(_len*0.2*setting_id):int(_len*0.2*(setting_id+1))].copy(), permute[int(_len*0.8):_len].copy()
                self.lookup_table.append(np.concatenate((original, permute)))

    def load_labels(self):
        for take in self.takes:
            label_file = os.path.join(self.label_folder, take + '.csv')
            label = np.loadtxt(label_file, dtype=int, delimiter=',')[1:, 1] # only load ground truth
            _len = (len(label) // 5) * 5
            label = label[:_len]       
            self.labels.append(label)
            self.seq_len.append(len(label.tolist()))


    def __iter__(self):
        if self.iter_method == 'sample':
            self.sample_count = 0
        elif self.iter_method == 'iter':
            self.cur_ind = -1
            self.take_indices = np.arange(len(self.takes))
            if self.shuffle:
                np.random.shuffle(self.take_indices)
            self.__next_take()
        return self

    def get_lb_ub(self, _len):
        if self.split == 'sequence':
            if self.mode == 'train':
                fr_lb = 0
                fr_ub = int(_len * self.split_ratio)
            elif self.mode == 'val':
                fr_lb = int(_len * self.split_ratio)
                fr_ub = _len
            elif self.mode == 'test':
                fr_lb = int(_len * self.split_ratio)
                fr_ub = _len   
        elif self.split == 'surgery':
            fr_lb = 0
            fr_ub = _len

        return fr_lb, fr_ub

    def __next_take(self):
        self.cur_ind = self.cur_ind + 1
        if self.cur_ind < len(self.take_indices):
            self.cur_tid = self.take_indices[self.cur_ind]
            _len = self.seq_len[self.cur_tid]
            self.fr_lb, self.fr_ub = self.get_lb_ub(_len)
            self.cur_fr = self.fr_lb
            self.ignore_indices = np.asarray(self.ignore_index[self.takes[self.cur_tid]])


    def sample(self):
        labels = []
        imgs = []
        sw_labels = []
        for _ in range(self.batch_size):
            self.sample_count += self.fr_num - self.overlap
            take_ind = np.random.randint(len(self.takes))
            ignore_indices = np.asarray(self.ignore_index[self.takes[take_ind]])
            seq_len = self.seq_len[take_ind]
            fr_lb, fr_ub = self.get_lb_ub(seq_len)

            orig_list = np.array(range(fr_lb, fr_ub - self.fr_num * self.sub_sample))

            for ignore in ignore_indices:
                ignore_range = np.array(range(ignore[0] - self.fr_num * self.sub_sample, ignore[1]))
                orig_list = np.setdiff1d(orig_list, ignore_range)

            
            fr_start = np.random.shuffle(orig_list)[0]
            fr_end = fr_start + self.fr_num * self.sub_sample



            img = self.load_imgs(take_ind, fr_start, fr_end)
            label = self.convert_label(take_ind, fr_start, fr_end)
            switch_label = self.convert_label_switch(take_ind, fr_start, fr_end)
            imgs.append(img)
            labels.append(label)
            sw_labels.append(switch_label)

        
        return np.asarray(imgs), np.asarray(labels), np.asarray(sw_labels)        

    def __next__(self):
        if self.iter_method == 'sample':
            if self.sample_count >= self.num_sample:
                raise StopIteration
            return self.sample()
        elif self.iter_method == 'iter':
            if self.cur_ind >= len(self.takes):
                raise StopIteration
            ignore_indices = 
            fr_start = self.cur_fr
            fr_end = self.cur_fr + self.fr_num * self.sub_sample if self.cur_fr + self.fr_num * self.sub_sample + 10 < self.fr_ub else self.fr_ub
            img = self.load_imgs(self.cur_tid, fr_start, fr_end)
            label = self.convert_label(self.cur_tid, fr_start, fr_end)
            switch_label = self.convert_label_switch(self.cur_tid, fr_start, fr_end)

            for ind in self.ignore_indices:
                if ind[0] <= fr_end - self.overlap and ind[1] >= fr_end - self.overlap:
                    self.cur_fr = ind[1] + 1
                    break
                else:
                    self.cur_fr = fr_end - self.overlap

            if fr_end == self.fr_ub:
                self.__next_take()
        
            return np.expand_dims(img, axis=0), np.expand_dims(label, axis=0), np.expand_dims(switch_label, axis=0)


    def convert_label(self, take_ind, start, end):
        label = self.labels[take_ind][start:end:self.sub_sample]
        res_label = []
        for l in label:
            one_hot_label = np.zeros(self.camera_num)
            for c in range(self.camera_num):
                if l == c:
                    one_hot_label[c] = 1
                    break
            res_label.append(one_hot_label)
        res_label = np.asarray(res_label)
        res_label = np.transpose(res_label) # Frame, Camera -> Camera, Frame
        return np.asarray(res_label)

    def convert_label_switch(self, take_ind, start, end):
        label = self.labels[take_ind][start:end:self.sub_sample]
        res_label = label[1:] != label[:-1]
        return res_label.astype(np.int64) # The first element contains switching is occured between frame 0 and 1.
 
    def load_imgs(self, take_ind, start, end):
        take_folder = '%s/%s' % (self.image_folder, self.takes[take_ind])
        imgs_all = []
        for i in range(start, end, self.sub_sample):
            if self.split == 'sequence':
                img_ind = self.lookup_table[take_ind][i]
            else:
                img_ind = i
            img_file = os.path.join(take_folder,'%06d.npz' % (img_ind))
            imgs = np.load(img_file, allow_pickle=True)['imgs']
            imgs = np.rollaxis(imgs, 3, 1)
            if self.mode != 'train' or True:
                imgs = np.rollaxis(imgs, 3, 1)
            imgs_all.append(imgs)
        if self.mode == 'train' and False:
            imgs_all = self.augment(imgs_all)
        imgs_all = np.asarray(imgs_all)
        imgs_all = np.rollaxis(imgs_all, 0, 2)

        return imgs_all

    """TODO: Implement data augmentation"""
    """Horizontal flip, vertical flip, random crop, gaussian noise"""
    def augment(self, imgs_all):
        c_vflip = np.random.rand(1) > 0.5
        c_hflip = np.random.rand(1) > 0.5
        crop_top = np.random.randint(0, 256 - 224)
        crop_left = np.random.randint(0, 256 - 224)
        gauss = np.random.normal(0,0.01,(224,224,3))
        gauss = gauss.reshape(224,224,3)

        augmented_imgs_all = []
        for imgs in imgs_all:
            for c_idx in range(imgs.shape[0]):
                img = imgs[c_idx, :, :, :].copy()
                if c_vflip:
                    img = img[::-1, :, :]
                if c_hflip:
                    img = img[:, ::-1, :]
                img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
                img = img[crop_top:crop_top+224, crop_left:crop_left+224, :]
                img += gauss
                imgs[c_idx, :, :, :] = img
            imgs = np.rollaxis(imgs, 3, 1)
            augmented_imgs_all.append(imgs)

        return augmented_imgs_all
                