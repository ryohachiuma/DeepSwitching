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
from switching.utils.DSNet_dataset import Dataset
from switching.utils.DSNet_config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='model_01')
parser.add_argument('--mode', default='train')
parser.add_argument('--data', default=None)
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
dsnet = DSNet(2, cfg.v_hdim, cfg.cnn_fdim, mlp_dim=cfg.mlp_dim, v_net_param=cfg.v_net_param, \
                            bi_dir=cfg.bi_dir, training=(args.mode == 'train'), is_dropout=cfg.is_dropout)


if args.iter > 0:
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp, meta = pickle.load(open(cp_path, "rb"))
    dsnet.load_state_dict(model_cp['ds_net'], strict=True)

dsnet.to(device)

ce_loss = nn.NLLLoss()

if cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(dsnet.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
else:
    optimizer = torch.optim.SGD(dsnet.parameters(), lr=cfg.lr, weight_decay=cfg.weightdecay)
fr_margin = cfg.fr_margin


def run_epoch(dataset, mode='train'):
    t0 = time.time()
    epoch_num_sample = 0
    epoch_loss = 0
    epoch_cat_loss = 0
    epoch_switch_loss = 0
    """
    img: (B, Cam, S, H, W, Channel)
    labels: (B, Cam, S)
    """
    for imgs_np, labels_np in dataset:
        num = imgs_np.shape[2] - 2 * fr_margin
        imgs = tensor(imgs_np, dtype=dtype, device=device)
        labels = tensor(labels_np, dtype=dtype, device=device)
        prob_pred, indices_pred = dsnet(imgs)
        prob_pred = prob_pred[:, :, fr_margin: -fr_margin, :]
        indices_pred = indices_pred[:, :, fr_margin:-fr_margin]
        """Compute loss"""
        """1. Categorical Loss"""
        """TODO: Change from cross entropy to focal loss"""
        cat_loss = ce_loss(prob_pred.view(-1, 2), labels.view(-1, 1))
        """2. Switching loss: if the selected camera is different from the next frame, 
        penalize that. This is just a regularization term."""
        prev_indices_pred = indices_pred[:, :-1]
        next_indices_pred = indices_pred[:, 1: ]
        switch_loss = torch.abs(next_indices_pred - prev_indices_pred)
        switch_loss = torch.sum(switch_loss / (switch_loss + 1e-8), dim=1)
        loss = cat_loss + cfg.w_d * switch_loss
        loss = loss.sum()
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # logging
        epoch_loss += loss.cpu().item() * num
        epoch_num_sample += num
        epoch_cat_loss += cat_loss.cpu().item() * num
        epoch_switch_loss += switch_loss.cpu().item() * num
        """clean up gpu memory"""
        torch.cuda.empty_cache()

    epoch_loss /= epoch_num_sample
    epoch_cat_loss /= epoch_num_sample
    epoch_switch_loss /= epoch_num_sample
    logger.info('epoch {:4d}    time {:.2f}     nsample {}   loss {:.4f} '
                    .format(i_epoch, time.time() - t0, epoch_num_sample, epoch_loss))

    return epoch_loss, epoch_cat_loss, epoch_switch_loss


if args.mode == 'train':
    dsnet.train()

    """Dataset"""
    tr_dataset = Dataset(cfg, 'train', cfg.fr_num, cfg.camera_num, cfg.batch_size, cfg.shuffle, 2*cfg.fr_margin, cfg.num_sample)
    val_dataset = Dataset(cfg, 'val', cfg.fr_num, cfg.camera_num, cfg.batch_size, cfg.shuffle, 2*cfg.fr_margin, cfg.num_sample)
    
    for i_epoch in range(args.iter, cfg.num_epoch):
        tr_loss, tr_cat_loss, tr_sw_loss = run_epoch(tr_dataset, mode='train')
        tb_logger.scalar_summary(['loss', 'ce_loss', 'switch_loss'], [tr_loss, tr_cat_loss, tr_sw_loss], i_epoch)

        val_loss, val_cat_loss, val_sw_loss = run_epoch(val_dataset, mode='val')
        tb_logger.scalar_summary(['val_loss', 'val_ce_loss', 'val_switch_loss'], [val_loss, val_cat_loss, val_sw_loss], i_epoch)


        with to_cpu(dsnet):
            if cfg.save_model_interval > 0 and (i_epoch + 1) % cfg.save_model_interval == 0:
                cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_epoch + 1)
                model_cp = {'ds_net': dsnet.state_dict()}
                pickle.dump(model_cp, open(cp_path, 'wb'))

elif args.mode == 'test':
    denet.test()
    print('test')