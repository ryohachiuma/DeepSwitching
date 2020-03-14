from utils.torch import *
from torch import nn
from models.resnet import ResNet
from models.tcn import TemporalConvNet
from models.rnn import RNN
from models.rnn import ConvLSTM
from models.mlp import MLP
from models.mlp import ResidualMLP
from torchvision import models
from torch.autograd import Variable
import math

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
        logits = self.linear(seq_features).view(-1, self.camera_num, fr_num, 2)

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
        logits = self.linear(seq_features).view(-1, self.camera_num, fr_num, 2)

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
        logits = self.linear(features).view(-1, self.camera_num, fr_num, 2)

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
        logits = self.linear(seq_features).view(-1, self.camera_num, fr_num, 2)

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
        logits = self.linear(seq_features).view(-1, self.camera_num, fr_num, 2)

        return logits

class DSNet_AR(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.training = training
        self.cnn = ResNet(cnn_fdim, running_stats=self.training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        #self.v_net = RNN(cnn_fdim * 2, v_hdim, bi_dir=bi_dir)
        self.v_net = nn.LSTM(cnn_fdim * 2, v_hdim // 2, batch_first=True, bidirectional=bi_dir)
        self.mlp = ResidualMLP(v_hdim + 2, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.scheduled_k = 0.996


    def forward(self, inputs, gt_label, _iter):
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

        initial_sampling = torch.distributions.Bernoulli(tensor(math.pow(self.scheduled_k, _iter))) # exponential decay

        inv_label = torch.abs(gt_label - 1)
        gt_label = torch.stack([gt_label, inv_label], dim=-1)
        prev_pred = gt_label[:, :, 0, :].view(-1, 2).type(self.dtype)
        logits = []
        for fr in range(fr_num):
            feat = seq_features[:, fr, :]
            ar_features = torch.cat([feat, prev_pred], dim=-1)
            ar_features = ar_features.contiguous().view(-1, self.v_hdim + 2)
            #batch x cameraNum, mlp_dim[-1] 
            ar_features = self.mlp(ar_features)
            #batch, cameraNum, 2
            pred = self.linear(ar_features)
            if initial_sampling.sample() and self.training:
                prev_pred = gt_label[:, :, fr, :].view(-1, 2).type(self.dtype)
            else:
                prev_pred = self.softmax(pred.clone())
            logits.append(pred.view(-1, self.camera_num, 2))
        #frameNum, batch, cameraNum, 2 -> batch, cameraNum, framenum, 2
        logits = torch.stack(logits).permute(1, 2, 0, 3)

        return logits

class DSNet_ARv3(nn.Module):

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
        self.mlp = ResidualMLP(v_hdim + 2, mlp_dim, 'leaky', is_dropout=is_dropout)
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

        logits = []
        prev_pred = torch.zeros((seq_features.size()[0], 2), dtype=self.dtype, device=self.device).fill_(0.5)
        for fr in range(fr_num):
            feat = seq_features[:, fr, :]
            ar_features = torch.cat([feat, prev_pred], dim=-1)
            ar_features = ar_features.contiguous().view(-1, self.v_hdim + 2)
            #batch x cameraNum, mlp_dim[-1] 
            ar_features = self.mlp(ar_features)
            #batch, cameraNum, 2
            pred = self.linear(ar_features)
            prev_pred = self.softmax(pred.clone())
            logits.append(pred.view(-1, self.camera_num, 2))
        #frameNum, batch, cameraNum, 2 -> batch, cameraNum, framenum, 2
        logits = torch.stack(logits).permute(1, 2, 0, 3)

        return logits


class DSNetv3(nn.Module):

    class ResNet(nn.Module):
        def __init__(self, fix_params=False, running_stats=False):
            super().__init__()
            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-3]      # delete the last fc layer and average pooling.
            self.resnet = nn.Sequential(*modules)
            self.bn_stats(running_stats)
        def forward(self, x):
            return self.resnet(x)
        def bn_stats(self, track_running_stats):
            for m in self.modules():
                if type(m) == nn.BatchNorm2d:
                    m.track_running_stats = track_running_stats

    class SpatialSELayer(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.conv = nn.Conv2d(num_channels, 1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_tensor, weights=None):
            """
            :param weights: weights for few shot learning
            :param input_tensor: X, shape = (batch_size, num_channels, H, W)
            :return: output_tensor
            """
            # spatial squeeze
            batch_size, channel, a, b = input_tensor.size()

            if weights:
                weights = weights.view(1, channel, 1, 1)
                out = F.conv2d(input_tensor, weights)
            else:
                out = self.conv(input_tensor)
            squeeze_tensor = self.sigmoid(out)

            # spatial excitation
            output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

            return output_tensor



    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.cnn = self.ResNet(running_stats=training)
        self.spatial_attention = self.SpatialSELayer(256)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type

        self.convlstm = ConvLSTM(self.cnn_fdim * 2, v_hdim // 2, (3,3), 1, True, True, False)
        self.mlp = MLP(v_hdim, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, inputs):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape))
        feat_size = local_feat.size()[-1]
        local_feat = self.spatial_attention(local_feat).contiguous().view(-1, self.camera_num, fr_num, self.cnn_fdim, feat_size, feat_size)
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1, 1, 1)
        cam_features = torch.cat([local_feat, glob_feat], 3).view(-1, fr_num, self.cnn_fdim * 2, feat_size, feat_size)
        for_features = self.convlstm(cam_features)[0]


        inv_idx = torch.arange(cam_features.size(1)-1, -1, -1).long()
        inv_cam_features = cam_features[:, inv_idx, :, :, :]

        back_features = self.convlstm(inv_cam_features)[0]
        seq_features = torch.cat([for_features, back_features], 2).view(-1, self.v_hdim, feat_size, feat_size)
        seq_features = torch.squeeze(self.gap(seq_features))
        logits = self.linear(self.mlp(seq_features)).view(-1, self.camera_num, fr_num, 2)

        return logits


class DSNet_ConvAR(nn.Module):

    class ResNet(nn.Module):
        def __init__(self, fix_params=False, running_stats=False):
            super().__init__()
            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-3]      # delete the last fc layer and average pooling.
            self.resnet = nn.Sequential(*modules)
            self.bn_stats(running_stats)
        def forward(self, x):
            return self.resnet(x)
        def bn_stats(self, track_running_stats):
            for m in self.modules():
                if type(m) == nn.BatchNorm2d:
                    m.track_running_stats = track_running_stats

    class SpatialSELayer(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.conv = nn.Conv2d(num_channels, 1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_tensor, weights=None):
            """
            :param weights: weights for few shot learning
            :param input_tensor: X, shape = (batch_size, num_channels, H, W)
            :return: output_tensor
            """
            # spatial squeeze
            batch_size, channel, a, b = input_tensor.size()

            if weights:
                weights = weights.view(1, channel, 1, 1)
                out = F.conv2d(input_tensor, weights)
            else:
                out = self.conv(input_tensor)
            squeeze_tensor = self.sigmoid(out)

            # spatial excitation
            output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

            return output_tensor



    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.training = training
        self.cnn = self.ResNet(running_stats=self.training)
        self.spatial_attention = self.SpatialSELayer(256)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type

        self.convlstm = ConvLSTM(self.cnn_fdim * 2, v_hdim // 2, (3,3), 1, True, True, False)
        self.mlp = ResidualMLP(v_hdim + 1, mlp_dim, 'leaky', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.scheduled_k = 0.997
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, gt_label, _iter):
        fr_num = inputs.size()[2]
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape))
        feat_size = local_feat.size()[-1]
        local_feat = self.spatial_attention(local_feat).contiguous().view(-1, self.camera_num, fr_num, self.cnn_fdim, feat_size, feat_size)
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1, 1, 1)
        cam_features = torch.cat([local_feat, glob_feat], 3).view(-1, fr_num, self.cnn_fdim * 2, feat_size, feat_size)
        for_features = self.convlstm(cam_features)[0]


        inv_idx = torch.arange(cam_features.size(1)-1, -1, -1).long()
        inv_cam_features = cam_features[:, inv_idx, :, :, :]

        back_features = self.convlstm(inv_cam_features)[0]
        seq_features = torch.cat([for_features, back_features], 2).view(-1, self.v_hdim, feat_size, feat_size)
        seq_features = torch.squeeze(self.gap(seq_features)).view(-1, fr_num, self.v_hdim)

        initial_sampling = torch.distributions.Bernoulli(tensor(math.pow(self.scheduled_k, _iter))) # exponential decay
        prev_pred = gt_label[:, :, 0].view(-1,).unsqueeze(1).type(self.dtype)
        logits = []
        for fr in range(fr_num):
            feat = seq_features[:, fr, :]
            ar_features = torch.cat([feat, prev_pred], dim=-1)
            ar_features = ar_features.contiguous().view(-1, self.v_hdim + 1)
            #batch x cameraNum, mlp_dim[-1] 
            ar_features = self.mlp(ar_features)
            #batch, cameraNum, 2
            pred = self.linear(ar_features)
            if initial_sampling.sample() and self.training:
                prev_pred = gt_label[:, :, fr].view(-1,).unsqueeze(1).type(self.dtype)
            else:
                prev_pred = self.softmax(pred.clone())[:, 1].unsqueeze(1)
            logits.append(pred.view(-1, self.camera_num, 2))
        #frameNum, batch, cameraNum, 2 -> batch, cameraNum, framenum, 2
        logits = torch.stack(logits).permute(1, 2, 0, 3)

        return logits


models_func = {         
'ds': DSNet,
'dsv2': DSNetv2,
'dsv3': DSNetv3,
'dsar': DSNet_AR,
'dsar_v3':DSNet_ARv3,
'baseline':Baseline,
'baseline_seq':Baseline_seq,
'baseline_spac':Baseline_spac,
'DSNet_ConvAR':DSNet_ConvAR
}

if __name__ == '__main__':
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=0)
    frame_num = 6
    camera_num = 3
    batch_size = 4
    net = DSNetv3(2, 128, 256, dtype, device, frame_num=frame_num, camera_num=camera_num, v_net_type='lstm')
    input = ones(batch_size, camera_num, frame_num, 3, 224, 224).contiguous()
    out, indi = net(input)
    print(out.size())
    print(indi.size())
