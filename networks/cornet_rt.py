import torch
import torch.optim
from torch import nn
import torch.utils.model_zoo as model_zoo

# Get pretrained weights of CORnet_RT from GitHub: https://github.com/dicarlolab/CORnet
"""
Download the pretrained weights from github and load them
GitHub: https://github.com/dicarlolab/CORnet
"""



class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output


class CORnet_RT(nn.Module):

    def __init__(self):
        super().__init__()
        self.feat_list = ['block1(V1)', 'block2(V2)', 'block3(V4)', 'block4(IT)', 'fc']

        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # renamed check code section "rename keys of state_dict"
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x1 = self.V1(x)
        x2 = self.V2(x1)
        x3 = self.V4(x2)
        x4 = self.IT(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x5 = self.fc(x)

        return [x1, x2, x3, x4, x5]


def cornet_rt(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        Github url: https://github.com/dicarlolab/CORnet
        weights were called "state_dict", if you named it differently you have to change it accordingly
    """
    model = CORnet_RT()

    if pretrained:
        model.load_state_dict(model_zoo.load_url("https://s3.amazonaws.com/cornet-models/cornet_rt-933c001c.pth"),strict=False)
    return model

