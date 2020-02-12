import numpy as np
import os
import yaml
import math
from utils import get_qvel_fd, de_heading
import pickle
import cv2

class Dataset:

    def __init__(self, cfg, mode, fr_num, camera_num, batch_size, iter_method='sample', frame_size=(224, 224, 3), split_ratio=0.8, shuffle=False, overlap=0, num_sample=20000):
        self.cfg = cfg
        self.mode = mode
        self.fr_num = fr_num
        self.shuffle = shuffle
        self.overlap = overlap
        self.num_sample = num_sample
        self.iter_method = iter_method
        
        self.camera_num = camera_num
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.base_folder = './datasets'
        self.image_folder = os.path.join(self.base_folder, 'frames')
        self.label_folder = os.path.join(self.base_folder, 'labels')

        # get take names
        
        if mode == 'train' or mode == 'val' or mode =='test':
            self.takes = self.cfg.takes['train']
        else:
            self.takes = self.cfg.takes[mode]
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

    def load_labels(self):
        for take in self.takes:
            label_file = os.path.join(self.label_folder, take + '.csv')
            label = np.loadtxt(label_file, dtype=int, delimiter=',')[:, 1] # only load ground truth
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

    def __next_take(self):
        self.cur_ind = self.cur_ind + 1
        if self.cur_ind < len(self.take_indices):
            self.cur_tid = self.take_indices[self.cur_ind]
            _len = self.seq_len[self.cur_tid]
            if self.mode == 'train':
                self.fr_lb = 0
                self.fr_ub = int(_len * self.split_ratio)
            else:
                self.fr_lb = int(_len * self.split_ratio)
                self.fr_ub = _len

            self.cur_fr = self.fr_lb

    def sample(self):
        labels = []
        imgs = []
        sw_labels = []
        for _ in range(self.batch_size):
            self.sample_count += self.fr_num - self.overlap
            take_ind = np.random.randint(len(self.takes))
            seq_len = self.seq_len[take_ind]
        
            if self.mode == 'train':
                fr_lb = 0
                fr_ub = int(seq_len * self.split_ratio)
            elif self.mode == 'val':
                fr_lb = int(seq_len * self.split_ratio)
                fr_ub = seq_len

            fr_start = np.random.randint(fr_lb, fr_ub - self.fr_num)
            fr_end = fr_start + self.fr_num 

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

            fr_start = self.cur_fr
            fr_end = self.cur_fr + self.fr_num if self.cur_fr + self.fr_num + 10 < self.fr_ub else self.fr_ub
            img = self.load_imgs(self.cur_tid, fr_start, fr_end)
            label = self.convert_label(self.cur_tid, fr_start, fr_end)
            switch_label = self.convert_label_switch(self.cur_tid, fr_start, fr_end)
            self.cur_fr = fr_end - self.overlap

            if fr_end == self.fr_ub:
                self.__next_take()
        
            return np.expand_dims(img, axis=0), np.expand_dims(label, axis=0), np.expand_dims(switch_label, axis=0)


    def convert_label(self, take_ind, start, end):
        label = self.labels[take_ind][start:end]
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
        label = self.labels[take_ind][start:end]
        res_label = label[1:] != label[:-1]
        return res_label.astype(np.int64) # The first element contains switching is occured between frame 0 and 1.
 
    def load_imgs(self, take_ind, start, end):
        take_folder = '%s/%s' % (self.image_folder, self.takes[take_ind])
        imgs_all = []
        for i in range(start, end):
            img_file = os.path.join(take_folder,'%06d.npy' % (i))
            imgs = np.load(img_file)
            imgs = np.rollaxis(imgs, 3, 1)
            imgs_all.append(imgs)
        imgs_all = np.asarray(imgs_all)
        imgs_all = np.rollaxis(imgs_all, 0, 2)

        return imgs_all
