import numpy as np
import os
import yaml
import math
from utils import get_qvel_fd, de_heading
import pickle
import cv2

class Dataset:

    def __init__(self, cfg, mode, fr_num, camera_num, batch_size, frame_size=(224, 224, 3), split_ratio=0.8, shuffle=False, overlap=0, num_sample=20000):
        self.cfg = cfg
        self.mode = mode
        self.fr_num = fr_num
        self.shuffle = shuffle
        self.overlap = overlap
        self.num_sample = num_sample
        
        self.camera_num = camera_num
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.base_folder = './datasets'
        self.image_folder = os.path.join(self.base_folder, 'frames')
        self.label_folder = os.path.join(self.base_folder, 'labels')

        # get take names
        
        if mode == 'train' or mode == 'val':
            self.takes = self.cfg.takes['train']
            self.seq_len = [54170]
            #self.seq_len = self.cfg.seq_len['train']
        else:
            self.takes = self.cfg.takes[mode]
            self.seq_len = self.cfg.seq_len[mode]

        # iterator specific
        self.sample_count = None
        self.take_indices = None
        self.cur_ind = None
        self.cur_tid = None
        self.cur_fr = None
        self.fr_lb = None
        self.fr_ub = None
        self.im_offset = None

        self.labels = []
        self.load_labels()

    def load_labels(self):
        for take in self.takes:
            label_file = os.path.join(self.label_folder, take + '.csv')
            label = np.loadtxt(label_file, dtype=int, delimiter=',')[:, 1] # only load ground truth
            self.labels.append(label)


    def __iter__(self):    
        self.sample_count = 0
        return self

    def __next__(self):
        if self.sample_count >= self.num_sample:
            raise StopIteration
        labels = []
        imgs = []
        for _ in range(self.batch_size):
            self.sample_count += self.fr_num - self.overlap
            take_ind = np.random.randint(len(self.takes))
            seq_len = self.seq_len[take_ind]
        
            if self.mode == 'train':
                fr_lb = 0
                fr_ub = int(seq_len * self.split_ratio)
                fr_start = np.random.randint(fr_lb, fr_ub - self.fr_num)
                fr_end = fr_start + self.fr_num

            elif self.mode == 'val':
                fr_lb = int(seq_len * self.split_ratio)
                fr_ub = seq_len
                fr_start = np.random.randint(fr_lb, fr_ub - self.fr_num)
                fr_end = fr_start + self.fr_num                

            img = self.load_imgs(take_ind, fr_start, fr_end)
            label = self.convert_label(take_ind, fr_start, fr_end)
            print(label.shape)
            imgs.append(img)
            labels.append(label)

        
        return np.asarray(imgs), np.asarray(labels)

    def convert_label(self, take_ind, start, end):
        label = self.labels[take_ind][start:end]
        res_label = []
        for l in label:
            one_hot_label = np.zeros(self.camera_num)
            for c in range(self.camera_num):
                if l == c:
                    one_hot_label[c] = 1
            res_label.append(label)
        res_label = np.asarray(res_label)
        res_label = np.transpose(res_label)
        return np.asarray(res_label)

    def load_imgs(self, take_ind, start, end):
        take_folder = '%s/%s' % (self.image_folder, self.takes[take_ind])
        imgs_all = []

        for i in range(start, end):
            img_file = os.path.join(take_folder,'%06d.npz' % (i))
            imgs = np.load(img_file)['imgs']
            imgs = np.rollaxis(imgs, 3, 1)
            imgs_all.append(imgs)
        imgs_all = np.asarray(imgs_all)
        #print(imgs_all.shape)
        imgs_all = np.rollaxis(imgs_all, 0, 2)
        #print(imgs_all.shape)
        #assert imgs_all.shape == (self.camera_num, end-start,(self.frame_size))
        return imgs_all
