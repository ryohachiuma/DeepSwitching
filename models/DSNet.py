from utils.torch import *
from torch import nn
from models.resnet import ResNet
from models.tcn import TemporalConvNet
from models.rnn import RNN
from models.mlp import MLP

class Baseline(nn.Module):
    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        self.mlp = MLP(v_hdim, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape))
        seq_features = self.mlp(local_feat)
        #batch, cameraNum, framenum, 2
        logits = self.lienar(seq_features)

        return logits

class Baseline_seq(nn.Module):
    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        self.v_net = nn.LSTM(cnn_fdim, v_hdim // 2, batch_first=True, bidirectional=bi_dir)
        self.mlp = MLP(v_hdim, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape)).view((-1, fr_num, self.cnn_fdim))
        #batch x cameraNum, framenum, v_hdim
        seq_features, _ = self.v_net(local_feat)
        seq_features = seq_features.contiguous().view(-1, self.v_hdim)
        #batch x cameraNum x framenum, mlp_dim
        seq_features = self.mlp(seq_features)
        #batch, cameraNum, framenum, 2
        logits = self.lienar(seq_features)

        return logits 



class Baseline_spac(nn.Module):
    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        self.mlp = MLP(v_hdim * 2, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape)).view((-1, fr_num, self.cnn_fdim))
        #batch, cameraNum, framenum, cnn_fdim
        local_feat = local_feat.contiguous().view(-1, self.camera_num, fr_num, self.cnn_fdim)
        #batch, 1, framenum, cnn_fdim
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        #batch, cameraNum, framenum, cnn_fdim
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1)
        #framenum, batch x cameraNum, cnn_fdimx2 
        cam_features = torch.cat([local_feat, glob_feat], -1).view(-1, self.cnn_fdim * 2)

        #batch x cameraNum x framenum, mlp_dim[-1] 
        features = self.mlp(cam_features)
        #batch, cameraNum, framenum, 2
        logits = self.lienar(seq_features)

        return logits

class DSNet(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        self.v_net = RNN(cnn_fdim * 2, v_hdim, bi_dir=bi_dir)
        #self.v_net = nn.LSTM(cnn_fdim * 2, v_hdim, 2, batch_first=True, dropout=0.01, bidirectional=bi_dir)
        self.mlp = MLP(v_hdim, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape)).view((-1, fr_num, self.cnn_fdim))
        #batch, cameraNum, framenum, cnn_fdim
        local_feat = local_feat.contiguous().view(-1, self.camera_num, fr_num, self.cnn_fdim)
        #batch, 1, framenum, cnn_fdim
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        #batch, cameraNum, framenum, cnn_fdim
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1)
        #framenum, batch x cameraNum, cnn_fdimx2 
        cam_features = torch.cat([local_feat, glob_feat], -1).view(-1, fr_num, self.cnn_fdim * 2).permute(1, 0, 2)
        
        #batch x cameraNum, framenum, v_hdim
        seq_features = self.v_net(cam_features).permute(1, 0, 2).contiguous().view(-1, self.v_hdim)
        #batch x cameraNum x framenum, mlp_dim[-1] 
        seq_features = self.mlp(seq_features)
        #batch, cameraNum, framenum, 2
        logits = self.lienar(seq_features)

        return logits
    


class DSNetv2(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        #self.v_net = RNN(cnn_fdim * 2, v_hdim, bi_dir=bi_dir)
        self.v_net = nn.LSTM(cnn_fdim * 2, v_hdim // 2, batch_first=True, bidirectional=bi_dir)
        self.mlp = MLP(v_hdim, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape)).view((-1, fr_num, self.cnn_fdim))
        #batch, cameraNum, framenum, cnn_fdim
        local_feat = local_feat.contiguous().view(-1, self.camera_num, fr_num, self.cnn_fdim)
        #batch, 1, framenum, cnn_fdim
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        #batch, cameraNum, framenum, cnn_fdim
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1)
        # batch x cameraNum, framenum, cnn_fdimx2 
        cam_features = torch.cat([local_feat, glob_feat], -1).view(-1, fr_num, self.cnn_fdim * 2)
        #batch x cameraNum, framenum, v_hdim
        seq_features, _ = self.v_net(cam_features)
        seq_features = seq_features.contiguous().view(-1, self.v_hdim)
        #batch x cameraNum x framenum, mlp_dim[-1] 
        seq_features = self.mlp(seq_features)
        #batch, cameraNum, framenum, 2
        logits = self.lienar(seq_features)

        return logits


class DSNetv3(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        #self.v_net = RNN(cnn_fdim * 2, v_hdim, bi_dir=bi_dir)
        self.v_net = nn.LSTM(cnn_fdim * 2, v_hdim // 2, batch_first=True, bidirectional=bi_dir)
        self.mlp = MLP(v_hdim, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape)).view((-1, fr_num, self.cnn_fdim))
        #batch, cameraNum, framenum, cnn_fdim
        local_feat = local_feat.contiguous().view(-1, self.camera_num, fr_num, self.cnn_fdim)
        #batch, 1, framenum, cnn_fdim
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        #batch, cameraNum, framenum, cnn_fdim
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1)
        # batch x cameraNum, framenum, cnn_fdimx2 
        cam_features = torch.cat([local_feat, glob_feat], -1).view(-1, fr_num, self.cnn_fdim * 2)
        #batch x cameraNum, framenum, v_hdim
        seq_features, _ = self.v_net(cam_features)
        seq_features = seq_features.contiguous().view(-1, self.v_hdim)
        #batch x cameraNum x framenum, mlp_dim[-1] 
        seq_features = self.mlp(seq_features)
        #batch, cameraNum, framenum, 2
        logits = self.lienar(seq_features)

        return logits



models_func = {         
'ds': DSNet,
'dsv2': DSNetv2,
'baseline':Baseline,
'baseline_seq':Baseline_seq,
'baseline_spac':Baseline_spac
}

if __name__ == '__main__':
    frame_num = 10
    camera_num = 3
    batch_size = 5
    net = DSNet(2, 128, 128, frame_num=frame_num, camera_num=camera_num, v_net_type='lstm')
    input = ones(batch_size, camera_num, frame_num, 3, 224, 224).contiguous() # Batch, Camera Num, Frame Num, Channel, H, W
    out, indi = net(input)
    print(out.size())
    print(indi.size())
