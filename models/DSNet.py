from utils.torch import *
from torch import nn
from models.resnet import ResNet
from models.tcn import TemporalConvNet
from models.rnn import RNN
from models.mlp import MLP
from torchvision import models
from torch.autograd import Variable

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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        layer_output_list = layer_output_list[-1:]
        return layer_output_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class DSNetv3(nn.Module):

    class ResNet(nn.Module):
        def __init__(self, fix_params=False, running_stats=False):
            super().__init__()
            resnet = models.resnet34(pretrained=True)
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
        self.spatial_attention = self.SpatialSELayer(self.cnn_fdim)
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



models_func = {         
'ds': DSNet,
'dsv2': DSNetv2,
'dsv3': DSNetv3,
'baseline':Baseline,
'baseline_seq':Baseline_seq,
'baseline_spac':Baseline_spac
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
