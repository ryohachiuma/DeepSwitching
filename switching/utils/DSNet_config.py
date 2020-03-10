import yaml
import os
from utils import recreate_dirs


class Config:

    def __init__(self, cfg_id, create_dirs=False):
        self.id = cfg_id
        cfg_name = 'config/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.load(open(cfg_name, 'r'), Loader=yaml.FullLoader)
        self.data_dir = './datasets'
        self.meta = yaml.load(open('%s/meta/meta_file.yml' % (self.data_dir), 'r'), Loader=yaml.FullLoader)
        self.camera_num = self.meta['camera_num']
        self.takes = {x: self.meta[x] for x in ['train', 'test', 'val']}

        # create dirs
        self.base_dir = 'results'
        self.data_dir = cfg.get('dataset_path', 'datasets')
        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.tb_dir)

        self.seed = cfg['seed']
        self.network = cfg['network']
        self.fr_num = cfg.get('fr_num', 20)
        self.v_net_param = cfg.get('v_net_param', None)
        self.v_hdim = cfg['v_hdim']
        self.mlp_dim = cfg['mlp_dim']
        self.cnn_fdim = cfg['cnn_fdim']
        self.optimizer = cfg.get('optimizer', 'Adam')
        self.lr = cfg['lr']
        self.num_epoch = cfg['num_epoch']
        self.shuffle = cfg.get('shuffle', False)
        self.num_sample = cfg.get('num_sample', 20000)
        self.sub_sample = cfg.get('sub_sample', 1)
        self.save_model_interval = cfg['save_model_interval']
        self.fr_margin = cfg['fr_margin']
        self.is_dropout = cfg.get('dropout', False)
        self.weightdecay = cfg.get('weight_decay', 0.0)
        self.bi_dir = cfg.get('bi_dir', True)
        self.batch_size = cfg.get('batch_size', 5)

        self.cat_loss = cfg.get('cat_loss', 'ce')
    
        self.w_d = cfg.get('w_d', 1.0)

