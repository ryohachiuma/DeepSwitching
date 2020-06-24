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
import shutil
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

sys.path.append(os.getcwd())

from switching.utils.DSNet_config import Config

FPS = 60.0

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='model_01')
parser.add_argument('--mode', default='vis')
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--data', default='test')
parser.add_argument('--show-type', default='compare', help='compare, selected, raw or image')

args = parser.parse_args()

cfg = Config(args.cfg)
res_base_dir = 'results'






if args.mode == 'vis':
    raw_img_dir = './datasets/raw_frame'
    
    for take in cfg.takes[args.data]:
        if not take in sr_res['select_pred']:
            print(take)
            continue
        
        select_pred = sr_res['select_pred'][take]
        select_gt = sr_res['select_orig'][take]
        start_ind = sr_res['start_ind'][take]

        if args.show_type == 'image':
            out_image_dir = '%s/%s/results/%s_iter_%04d' % (res_base_dir, args.cfg, take, args.iter)
            os.makedirs(out_image_dir, exist_ok=True)
            for ind in range(0, len(select_pred)):
                pred = select_pred[ind]
                gt   = select_gt[ind]

                for i in range(cfg.sub_sample):
                    
                    img_file = os.path.join(raw_img_dir, take, '%06d.jpg' % (start_ind + ind * cfg.sub_sample + i))

                    img = cv2.imread(img_file)
                    img_size = img.shape[0]
                    img_list = []
                    for c in range(5):
                        _img = img[:, img_size * c:img_size * (c+1), :]
                        img_list.append(_img)

                    img = cv2.vconcat(img_list)

                    new_file = os.path.join(out_image_dir, '%06d_pred%d_gt%d.jpg' % (start_ind + ind * cfg.sub_sample + i, pred, gt))
                    cv2.imwrite(new_file, img)
                    #shutil.copy(img_file, new_file)
        elif args.show_type == 'raw':
            out_csv_dir = '%s/%s/results/%s_iter_%04d.csv' % (res_base_dir, args.cfg, take, args.iter)
            save_array = np.concatenate([select_gt[:, np.newaxis], select_pred[:, np.newaxis]], axis=1)
            np.savetxt(out_csv_dir, save_array, fmt='%d', delimiter=',')

        elif args.show_type == 'prob':
            plt.rcParams["font.size"] = 17
            fig = plt.figure(figsize=(12.0, 5.5), dpi=300)
            select_prob = sr_res['raw_prob'][take]
            #select_prob = np.repeat(select_prob, 5, axis=0)
            ax = plt.axes(xlim=(0, select_prob.shape[0]), ylim=(0, 1.0)) 
            N = 5
            colors = [[66/255, 133/255, 244/255], [244/255, 180/255, 0/255],[15/255, 157/255, 88/255],\
                [237/255, 125/255, 49/255], [219/255, 68/255, 55/255]]
            lines = [plt.plot([], [], color=colors[i],linewidth = 4.0)[0] for i in range(N)] #lines to animate   
            circles = [patches.Ellipse(xy=(0.0, 0.0), width=2.0, height=0.04, color=colors[i]) for i in range(N)]
            #circles = [plt.Circle((0.0,0.0),0.05,fill=False, clip_on=False, color=colors[i]) for i in range(N)]

            seq_len = select_prob.shape[0]
            def init():
                #init lines
                for line in lines:
                    line.set_data([], []) 
                for circle in circles:
                    circle.center = (0.0, 0.0)
                    ax.add_patch(circle)
                return lines, circles

            def animate(ind):

                #animate lines
                pred = select_prob[:ind, :]
                for j,line in enumerate(lines):
                    line.set_data(range(ind), pred[:, j])
                print(pred.shape)
                if pred.shape[0] != 0:
                    for j, circle in enumerate(circles):
                        circle.center = ((ind, pred[-1, j]))

                if ind < 100:
                    minx = 0
                    maxx = 100
                elif ind > seq_len - 100:
                    minx = seq_len - 100
                    maxx = seq_len
                else:
                    minx, maxx = ax.get_xlim()
                    minx+=1
                    maxx+=1
                ax.set_xlim(minx, maxx)

                return lines, circles
            #plt.tight_layout()
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=seq_len, interval=0)
            #
            #plt.show()
            anim.save('test.mp4', writer='ffmpeg', fps=12)
            #exit(0)
                    


            
        else:
            out_movie_path = '%s/%s/results/%s_iter_%04d_%s.mp4' % (res_base_dir, args.cfg, take, args.iter, args.show_type)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            if args.show_type == 'compare':
                res_size = (224*2, 224)
            elif args.show_type == 'selected':
                res_size = (224*5, 224)
            video_writer = cv2.VideoWriter(out_movie_path, int(fourcc), FPS, res_size)
            for ind in range(0, len(select_pred)):
                pred = select_pred[ind]
                gt   = select_gt[ind]

                for i in range(cfg.sub_sample):
                    img_file = os.path.join(raw_img_dir, take, '%06d.jpg' % (start_ind + ind * cfg.sub_sample + i))
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
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_s = 0.0
    f1_seq = {'surgery_01': 0.0, 'surgery_02': 0.0, 'surgery_03': 0.0, 'surgery_04': 0.0, 'surgery_05': 0.0, 'surgery_06': 0.0}
    stat_mode = 'weighted'

    for c in range(5):
        '''
        if c == 2:
            sr_res_path = '%s/%s/%s/results/iter_%04d_%s.p' % (res_base_dir, args.cfg, str(c), 8000, args.data)
        else:
            sr_res_path = '%s/%s/%s/results/iter_%04d_%s.p' % (res_base_dir, args.cfg, str(c), args.iter, args.data)
        '''
        sr_res_path = '%s/%s/%s/results/iter_%04d_%s.p' % (res_base_dir, args.cfg, str(c), args.iter, args.data)
        
        sr_res = pickle.load(open(sr_res_path, 'rb'))
        for take in cfg.takes[args.data]:
            select_pred = sr_res['select_pred'][take].astype(np.int64)
            select_gt = sr_res['select_orig'][take].astype(np.int64)


            acc = accuracy_score(select_gt, select_pred)
            prec = precision_score(select_gt, select_pred, average=stat_mode)
            rec = recall_score(select_gt, select_pred, average=stat_mode)
            f1 = f1_score(select_gt, select_pred, average=stat_mode)

            print('take %s setting %s Accuracy: %.4f Precision: %.4f Recall: %.4f F1: %.4f' % (take, str(c), acc, prec, rec, f1))
            accuracy += acc
            precision += prec
            recall += rec
            f1_s += f1
            f1_seq[take] += f1

    accuracy  /=  (len(cfg.takes[args.data]) * 5)
    precision /=  (len(cfg.takes[args.data]) * 5)
    recall    /=  (len(cfg.takes[args.data]) * 5)
    f1_s      /=  (len(cfg.takes[args.data]) * 5)

    for key in f1_seq.keys():
        f1_seq[key] /= 5

    print('-' * 50)
    print('overall Accuracy: %.4f Precision: %.4f Recall: %.4f F1: %.4f' % (accuracy, precision, recall, f1_s))
    print('-' * 50)
    print(f1_seq)