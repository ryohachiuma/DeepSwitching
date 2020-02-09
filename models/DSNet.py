from utils.torch import *
from torch import nn
from models.resnet import ResNet
from models.tcn import TemporalConvNet
from models.rnn import RNN
from models.mlp import MLP


class DSNet(nn.Module):

    def __init__(self, out_dim, v_hdim, cnn_fdim, dtype, device, frame_num=10, camera_num=3, frame_shape=(3, 224, 224), mlp_dim=(128, 64),
                 v_net_type='lstm', v_net_param=None, bi_dir=False, training=True, is_dropout=False):
        super().__init__()
        self.out_dim = out_dim
        self.cnn_fdim = cnn_fdim
        self.v_hdim = v_hdim
        self.frame_shape = frame_shape
        self.camera_num = camera_num
        self.frame_num = frame_num
        self.cnn = ResNet(cnn_fdim, running_stats=training)
        self.dtype = dtype
        self.device = device

        self.v_net_type = v_net_type
        self.v_net = nn.LSTM(cnn_fdim, v_hdim // 2, 2, batch_first=True, dropout=0.01, bidirectional=bi_dir)
        self.mlp = MLP(v_hdim * 2, mlp_dim, 'relu', is_dropout=is_dropout)
        self.linear = nn.Linear(self.mlp.out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def soft_argmax(self, inputs, beta=10, dim=1, epsilon=1e-12):
        '''
        applay softargmax on inputs, return \sum_i ( i * (exp(A_i * beta) / \sum_i(exp(A_i * beta))))
        according to https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        :param A:
        :param dim:
        :param epsilon:
        :return:
        '''
        A_max = torch.max(inputs, dim=dim, keepdim=True)[0]
        A_exp = torch.exp((inputs - A_max)*beta)
        A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
        indices = torch.arange(start=0, end=inputs.size()[dim], dtype=self.dtype, \
            device=self.device, requires_grad=True)
        return torch.matmul(A_softmax, indices)

    def forward(self, inputs):
        '''
        Inputs: Batch, Camera, Sequence, Channel, H, W
        Outputs: probs(Batch, Camera, Sequence, 2),  max_indices(Batch, Sequence)
        '''
        print(inputs.view((-1,) + self.frame_shape).size())
        #batch x cameraNum, framenum, cnn_fdim
        local_feat = self.cnn(inputs.view((-1,) + self.frame_shape)).view((-1, self.frame_num, self.cnn_fdim))
        print(local_feat.size())
        #batch x cameraNum, framenum, v_hdim
        local_feat, _ = self.v_net(local_feat)
        print(local_feat.size())
        #batch, cameraNum, framenum, v_hdim
        local_feat = local_feat.contiguous().view(-1, self.camera_num, self.frame_num, self.v_hdim)
        print(local_feat.size())
        #batch, 1, framenum, v_hdim
        glob_feat = torch.max(local_feat, 1, keepdim=True)[0]
        print(glob_feat.size())
        #batch, cameraNum, framenum, v_hdim
        glob_feat = glob_feat.repeat(1, self.camera_num, 1, 1)
        print(glob_feat.size())
        #batch x cameraNum x framenum, v_hdimx2 
        cam_features = torch.cat([local_feat, glob_feat], -1).view(-1, self.v_hdim * 2)
        print(cam_features.size())
        #batch x cameraNum x framenum, mlp_dim[-1] 
        cam_features = self.mlp(cam_features)
        print(cam_features.size())
        #batch, cameraNum, framenum, 2
        probs = self.softmax(self.linear(cam_features)).view(-1, self.camera_num, self.frame_num, self.out_dim)
        print(probs.size())
        #batch, cameraNum, framenum
        select_prob = probs[:, :, :, 1] # 0: not selected, 1: selected
        #batch x self.frame_num, cameraNum
        select_prob = select_prob.permute(0, 2, 1).contiguous().view(-1, self.camera_num)
        #batch x self.frame_num
        max_indices = self.soft_argmax(select_prob)
        #batch, self.frame_num
        max_indices = max_indices.view(-1, self.frame_num)

        return torch.log(probs), max_indices


if __name__ == '__main__':
    frame_num = 10
    camera_num = 3
    batch_size = 5
    net = DSNet(2, 128, 128, frame_num=frame_num, camera_num=camera_num, v_net_type='lstm')
    input = ones(batch_size, camera_num, frame_num, 3, 224, 224).contiguous() # Batch, Camera Num, Frame Num, Channel, H, W
    out, indi = net(input)
    print(out.size())
    print(indi.size())
