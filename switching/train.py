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
import torch.nn.functional as F

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
dsnet = models_func[cfg.network](2, cfg.v_hdim, cfg.cnn_fdim, dtype, device, mlp_dim=cfg.mlp_dim, camera_num=cfg.camera_num, \
        v_net_param=cfg.v_net_param, bi_dir=cfg.bi_dir, training=(args.mode == 'train'), is_dropout=cfg.is_dropout)

if args.iter > 0:
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    dsnet.load_state_dict(model_cp['ds_net'], strict=False)

dsnet.to(device)
class_weights = torch.tensor([0.2, 0.8], dtype=dtype, device=device)
cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)
cat_crit = nn.NLLLoss(weight=class_weights)
focal_crit = loss.FocalLoss()
switch_crit = loss.SwitchingLoss()
kl_crit = loss.SelectKLLoss()

if cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(dsnet.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
else:
    optimizer = torch.optim.SGD(dsnet.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
fr_margin = cfg.fr_margin


logger_str = {'train': "Training: ", 'val': "Validation: "}
_iter = {'train': 0, 'val': 0}
loss_log = {'train': ['loss', 'acc'], 'val': ['val_loss', 'val_acc']}

def run_epoch(dataset, mode='train'):
    global dsnet, optimizer, focal_crit, switch_crit, kl_crit, _iter
    """
    img: (B, Cam, S, H, W, Channel)
    labels: (B, Cam, S)
    """
    for imgs_np, labels_np, _ in dataset:
        t0 = time.time()
        imgs = tensor(imgs_np, dtype=dtype, device=device)
        labels = tensor(labels_np, dtype=torch.long, device=device)
        if cfg.network == 'dsar':
            prob_pred = dsnet(imgs, labels)
        else:
            prob_pred = dsnet(imgs)
        
        """1. Categorical Loss."""
        prob_pred = prob_pred[:, :, fr_margin: -fr_margin, :].contiguous()
        labels_gt = labels[:, :, fr_margin:-fr_margin].contiguous()
        if cfg.loss == 'cross_entropy':
            cat_loss = cross_entropy_loss(prob_pred.view(-1, 2), labels_gt.view(-1,))
        elif cfg.loss == 'focal':
            cat_loss = focal_crit(prob_pred.view(-1, 2), labels_gt.view(-1,))
        loss = cat_loss.mean()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cfg.save_model_interval > 0 and _iter['train'] % cfg.save_model_interval == 0:
                with to_cpu(dsnet):
                    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, _iter['train'])
                    model_cp = {'ds_net': dsnet.state_dict()}
                    pickle.dump(model_cp, open(cp_path, 'wb'))


        prob_pred = F.softmax(prob_pred, dim=-1).detach().cpu().numpy()
        select_ind = np.argmax(prob_pred, axis=-1)
        label_gt = labels_np[:, :, fr_margin:-fr_margin]
        assert select_ind.shape == label_gt.shape, 'shape should match!'
        acc = np.count_nonzero(select_ind == label_gt) / float(label_gt.shape[0] * label_gt.shape[1] * label_gt.shape[2])
        tb_logger.scalar_summary(loss_log[mode], [loss, acc], _iter[mode])  
        logger.info(logger_str[mode] + 'iter {:6d}    time {:.2f}    loss {:.4f} acc {:.4f}'
                        .format(_iter[mode], time.time() - t0, loss, acc))
        _iter[mode]+=1


        """clean up gpu memory"""
        torch.cuda.empty_cache()
        del imgs, labels



if args.mode == 'train':
    to_train(dsnet)

    """Dataset"""
    tr_dataset = Dataset(cfg, 'train', cfg.fr_num, cfg.camera_num, cfg.batch_size, cfg.split, iter_method=cfg.iter_method, shuffle=cfg.shuffle, overlap=2*cfg.fr_margin, num_sample=cfg.num_sample, sub_sample=cfg.sub_sample)
    val_dataset = Dataset(cfg, 'val', cfg.fr_num,  cfg.camera_num,              1, cfg.split, iter_method='iter', overlap=2*cfg.fr_margin, sub_sample=cfg.sub_sample)
    
    for _ in range(args.iter // cfg.num_sample, cfg.num_epoch):
        run_epoch(tr_dataset, mode='train')
        run_epoch(val_dataset, mode='val')




elif args.mode == 'test':
    to_test(dsnet)
    dataset = Dataset(cfg, 'test', cfg.fr_num,  cfg.camera_num, 1, cfg.split, iter_method='iter', overlap=2*cfg.fr_margin, sub_sample=cfg.sub_sample)
    print(dataset.takes)
    torch.set_grad_enabled(False)

    res_pred = {}
    res_orig = {}
    take_ = {}
    res_pred_arr = []
    res_orig_arr = []
    meta_start_arr = []
    take = dataset.takes[0]
    for imgs_np, labels_np, _ in dataset:
        print(take)
        if not take in take_:
            take_[take] = dataset.fr_lb + fr_margin
        imgs = tensor(imgs_np, dtype=dtype, device=device)
        prob_pred = dsnet(imgs)
        prob_pred = F.softmax(prob_pred[:, :, fr_margin: -fr_margin, :], dim=-1).cpu().numpy()
        select_prob = np.squeeze(prob_pred[:, :, :, 1])
        select_ind = np.argmax(select_prob, axis=0) # along camera direction
        res_pred_arr.append(select_ind)

        select_ind_gt = np.argmax(np.squeeze(labels_np[:, :, fr_margin:-fr_margin]), axis=0)
        res_orig_arr.append(select_ind_gt)

        if dataset.cur_ind >= len(dataset.takes) or dataset.takes[dataset.cur_tid] != take:
            res_pred[take] = np.concatenate(res_pred_arr)
            res_orig[take] = np.concatenate(res_orig_arr)
            res_pred_arr, res_orig_arr = [], []
            take = dataset.takes[dataset.cur_tid]
            take_[take] = dataset.fr_lb + fr_margin

    results = {'select_pred': res_pred, 'select_orig': res_orig, '': take_}
    res_path = '%s/iter_%04d_%s.p' % (cfg.result_dir, args.iter, args.data)
    pickle.dump(results, open(res_path, 'wb'))
    logger.info('saved results to %s' % res_path)