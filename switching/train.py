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
    model_cp, meta = pickle.load(open(cp_path, "rb"))
    dsnet.load_state_dict(model_cp['ds_net'], strict=True)

dsnet.to(device)
class_weights = torch.tensor([0.2, 0.8], dtype=dtype, device=device)
ce_loss = nn.NLLLoss(weight=class_weights)
#cat_crit = loss.FocalLossWithOutOneHot()
cat_crit = nn.NLLLoss(weight=class_weights)
switch_crit = loss.SwitchingLoss()

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
    labels: (B, S - 1)
    """
    for imgs_np, labels_np, sw_labels_np in dataset:
        num = imgs_np.shape[2] - 2 * fr_margin
        imgs = tensor(imgs_np, dtype=dtype, device=device)
        labels = tensor(labels_np, dtype=torch.long, device=device)[:, :, fr_margin:-fr_margin]
        sw_labels = tensor(sw_labels_np, dtype=dtype, device=device)[:, fr_margin:-fr_margin]
        prob_pred, indices_pred = dsnet(imgs)
        prob_pred = prob_pred[:, :, fr_margin: -fr_margin, :]
        indices_pred = indices_pred[:, fr_margin:-fr_margin]

        """1. Categorical Loss (Inputs: after-softmax logits, Outputs: Label)"""
        cat_loss = cat_crit(prob_pred.contiguous().view(-1, 2), labels.contiguous().view(-1,))
        """2. Switching loss: if the selected camera is different from the next frame, 
        penalize that. This is just a regularization term."""
        switch_loss = switch_crit(indices_pred, sw_labels)
        loss = cat_loss + cfg.w_d * switch_loss
        #loss = cat_loss
        loss = loss.mean()
        print('{:4f}, {:4f}, {:4f}'.format(loss, cat_loss.mean(), switch_loss.mean()))
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # logging
        epoch_loss += loss.cpu() * num
        epoch_num_sample += num
        epoch_cat_loss += cat_loss.sum().cpu() * num
        epoch_switch_loss += switch_loss.sum().cpu() * num
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        del imgs

    epoch_loss /= epoch_num_sample
    epoch_cat_loss /= epoch_num_sample
    epoch_switch_loss /= epoch_num_sample
    logger.info('epoch {:4d}    time {:.2f}     nsample {}   loss {:.4f} cat_loss {:.4f} sw_loss {:.4f}'
                    .format(i_epoch, time.time() - t0, epoch_num_sample, epoch_loss, epoch_cat_loss, epoch_switch_loss))

    return epoch_loss, epoch_cat_loss, epoch_switch_loss


if args.mode == 'train':
    dsnet.train()

    """Dataset"""
    tr_dataset = Dataset(cfg, 'train', cfg.fr_num, cfg.camera_num, cfg.batch_size, shuffle=cfg.shuffle, overlap=2*cfg.fr_margin, num_sample=cfg.num_sample)
    val_dataset = Dataset(cfg, 'val', cfg.fr_num,  cfg.camera_num,              1, shuffle=cfg.shuffle, overlap=2*cfg.fr_margin, num_sample=1000)
    
    for i_epoch in range(args.iter, cfg.num_epoch):
        tr_loss, tr_cat_loss, tr_sw_loss = run_epoch(tr_dataset, mode='train')
        tb_logger.scalar_summary(['loss', 'ce_loss', 'switch_loss'], [tr_loss, tr_cat_loss, tr_sw_loss], i_epoch)

        torch.cuda.empty_cache()
        '''
        val_loss, val_cat_loss, val_sw_loss = run_epoch(val_dataset, mode='val')
        tb_logger.scalar_summary(['val_loss', 'val_ce_loss', 'val_switch_loss'], [val_loss, val_cat_loss, val_sw_loss], i_epoch)
        torch.cuda.empty_cache()
        '''
        with to_cpu(dsnet):
            if cfg.save_model_interval > 0 and (i_epoch + 1) % cfg.save_model_interval == 0:
                cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_epoch + 1)
                model_cp = {'ds_net': dsnet.state_dict()}
                pickle.dump(model_cp, open(cp_path, 'wb'))

elif args.mode == 'test':
    dsnet.eval()
    print('test')
