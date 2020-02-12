"""Deep Switching: A deep learning based camera selection for occlusion-less surgery recording."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import sys
import pickle
import time
import torch
import numpy as np
sys.path.append(os.getcwd())

from utils import *
from models.DSNet import *
from switching import loss
from switching.utils.DSNet_dataset import Dataset
from switching.utils.DSNet_config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='model_01')
parser.add_argument('--mode', default='train')
parser.add_argument('--data', default='train')
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

cfg = Config(args.cfg, create_dirs=(args.iter == 0))

"""setup"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
tb_logger = Logger(cfg.tb_dir)
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))



"""network"""
dsnet = DSNet(2, cfg.v_hdim, cfg.cnn_fdim, dtype, device, mlp_dim=cfg.mlp_dim, frame_num=cfg.fr_num, camera_num=cfg.camera_num, \
    v_net_param=cfg.v_net_param, bi_dir=cfg.bi_dir, training=(args.mode == 'train'), is_dropout=cfg.is_dropout)


if args.iter > 0:
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    dsnet.load_state_dict(model_cp['ds_net'], strict=True)

dsnet.to(device)
class_weights = torch.tensor([0.2, 0.8], dtype=dtype, device=device)
cat_crit = nn.NLLLoss(weight=class_weights)
switch_crit = loss.SwitchingLoss()

if cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(dsnet.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
else:
    optimizer = torch.optim.SGD(dsnet.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
fr_margin = cfg.fr_margin


logger_str = {'train': "Training: ", 'val': "Validation: "}
_iter = {'train': 0, 'val': 0}
loss_log = {'train': ['loss', 'ce_loss', 'switch_loss'], 'val': ['val_loss', 'val_ce_loss', 'val_switch_loss']}

def run_epoch(dataset, mode='train'):
    """
    img: (B, Cam, S, H, W, Channel)
    labels: (B, Cam, S)
    sw_labels: (B, S - 1)
    """
    for imgs_np, labels_np, sw_labels_np in dataset:
        t0 = time.time()
        imgs = tensor(imgs_np, dtype=dtype, device=device)
        labels = tensor(labels_np[:, :, fr_margin:-fr_margin], dtype=torch.long, device=device).contiguous()
        sw_labels = tensor(sw_labels_np[:, fr_margin:-fr_margin], dtype=dtype, device=device)

        prob_pred, indices_pred = dsnet(imgs)
        prob_pred = prob_pred[:, :, fr_margin: -fr_margin, :].contiguous()
        indices_pred = indices_pred[:, fr_margin:-fr_margin]

        """1. Categorical Loss."""
        cat_loss = cat_crit(prob_pred.view(-1, 2), labels.view(-1,))
        """2. Switching loss."""
        switch_loss = switch_crit(indices_pred, sw_labels)
        loss = cat_loss + cfg.w_d * switch_loss
        loss = loss.mean()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            if cfg.save_model_interval > 0 and _iter['train'] % cfg.save_model_interval == 0:
                with to_cpu(dsnet):
                    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, _iter['train'])
                    model_cp = {'ds_net': dsnet.state_dict()}
                    pickle.dump(model_cp, open(cp_path, 'wb'))

        tb_logger.scalar_summary(loss_log[mode], [loss, cat_loss.mean(), switch_loss.mean()], _iter[mode])  
        logger.info(logger_str[mode] + 'iter {:6d}    time {:.2f}    loss {:.4f} cat_loss {:.4f} sw_loss {:.4f}'
                        .format(_iter[mode], time.time() - t0, loss, cat_loss.mean(), switch_loss.mean()))
        _iter[mode]+=1


        """clean up gpu memory"""
        torch.cuda.empty_cache()
        del imgs, labels, sw_labels



if args.mode == 'train':
    dsnet.train()

    """Dataset"""
    tr_dataset = Dataset(cfg, 'train', cfg.fr_num, cfg.camera_num, cfg.batch_size, shuffle=cfg.shuffle, overlap=2*cfg.fr_margin, num_sample=cfg.num_sample)
    val_dataset = Dataset(cfg, 'val', cfg.fr_num,  cfg.camera_num,              1, iter_method='iter', overlap=2*cfg.fr_margin)
    
    for _ in range(args.iter, cfg.num_epoch):
        run_epoch(tr_dataset, mode='train')

        with torch.no_grad:
            run_epoch(val_dataset, mode='val')




elif args.mode == 'test':
    dsnet.eval()
    dataset = Dataset(cfg, args.data, cfg.fr_num, cfg.camera_num, 1, iter_method='iter', overlap=2*cfg.fr_margin)
    torch.set_grad_enabled(False)

    res_pred = {}
    res_orig = {}
    take_start_ind = {}
    res_pred_arr = []
    res_orig_arr = []
    meta_start_arr = []
    take = dataset.takes[0]
    #take_start_ind[take] = dataset.fr_lb + fr_margin
    for imgs_np, labels_np, _ in dataset:
        print(dataset.cur_fr)
        if not take in take_start_ind:
            take_start_ind[take] = dataset.fr_lb + fr_margin
            print(dataset.fr_lb + fr_margin)
        t0 = time.time()
        imgs = tensor(imgs_np, dtype=dtype, device=device)
        prob_pred, _ = dsnet(imgs)
        prob_pred = prob_pred[:, :, fr_margin: -fr_margin, :].cpu().numpy()
        select_prob = np.squeeze(prob_pred[:, :, :, 1])
        select_ind = np.argmax(select_prob, axis=0)
        
        res_pred_arr.append(select_ind)

        select_ind_gt = np.argmax(np.squeeze(labels_np[:, :, fr_margin:-fr_margin]), axis=0)
        res_orig_arr.append(select_ind_gt)
        print('pred')
        print(select_ind)
        print('gt')
        print(select_ind_gt)

        if dataset.cur_ind >= len(dataset.takes) or dataset.takes[dataset.cur_tid] != take:
            res_pred[take] = np.vstack(res_pred_arr)
            res_orig[take] = np.vstack(res_orig_arr)
            res_pred_arr, res_orig_arr = [], []
            take = dataset.takes[dataset.cur_tid]
            take_start_ind[take] = dataset.fr_lb + fr_margin

    results = {'select_pred': res_pred, 'select_orig': res_orig, 'start_ind': take_start_ind}
    res_path = '%s/iter_%04d_%s.p' % (cfg.result_dir, args.iter, args.data)
    pickle.dump(results, open(res_path, 'wb'))
    logger.info('saved results to %s' % res_path)