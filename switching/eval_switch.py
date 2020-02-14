import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import sys
import pickle
import math
import time
import glob
import numpy as np
import cv2
sys.path.append(os.getcwd())

from switching.utils.DSNet_config import Config

FPS = 10.0

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='model_01')
parser.add_argument('--mode', default='vis')
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--data', default='test')
parser.add_argument('--show-type', default='compare', help='compare or selected')

args = parser.parse_args()

cfg = Config(args.cfg)
res_base_dir = 'results'
sr_res_path = '%s/%s/results/iter_%04d_%s.p' % (res_base_dir, args.cfg, args.iter, args.data)
sr_res = pickle.load(open(sr_res_path, 'rb'))





if args.mode == 'vis':
    raw_img_dir = './datasets/raw_frame'
    
    for take in cfg.takes[args.data]:
        if not take in sr_res['select_pred']:
            print(take)
            continue
        out_movie_path = '%s/%s/results/%s_iter_%04d_%s.mp4' % (res_base_dir, args.cfg, take, args.iter, args.show_type)
        select_pred = sr_res['select_pred'][take]
        select_gt = sr_res['select_orig'][take]
        start_ind = sr_res['start_ind'][take]


        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        if args.show_type == 'compare':
            res_size = (224*2, 224)
        elif args.show_type == 'selected':
            res_size = (224*5, 224)
        video_writer = cv2.VideoWriter(out_movie_path, int(fourcc), FPS, res_size)
        for ind in range(len(select_pred)):
            pred = select_pred[ind]
            gt   = select_gt[ind]

            img_file = os.path.join(raw_img_dir, take, '%06d.jpg' % (start_ind + ind))
            img = cv2.imread(img_file)
            img_size = img.shape[0]


            if args.show_type == 'compare':
                pred_img = img[:, img_size*pred:img_size*(pred+1), :]
                gt_img   = img[:, img_size*gt:img_size*(gt+1), :]
                res_img = cv2.hconcat([pred_img, gt_img])
            elif args.show_type == 'selected':
                _mask = np.zeros(img.shape, dtype=np.uint8)
                _mask[:, img_size*pred:img_size*(pred+1), :] = [255, 255, 255]
                res_img = cv2.addWeighted(img, 0.8, _mask, 0.2, 0)

            video_writer.write(res_img)
        video_writer.release()


elif args.mode =='stats':
    for take in cfg.takes[args.data]:
        select_pred = sr_res['select_pred'][take]
        select_gt = sr_res['select_orig'][take]
        start_ind = sr_res['start_ind'][take]

        print('take %s Accuracy: %.4f' % (take, np.count_nonzero(select_pred == select_gt) / float(select_gt.shape[0])))
