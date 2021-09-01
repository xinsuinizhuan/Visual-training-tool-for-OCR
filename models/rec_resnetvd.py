import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ["RecResNet"]


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBNACTWithPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=None):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=stride, stride=stride, padding=0, ceil_mode=True)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act is None:
            self.act = None
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first=False):
        super().__init__()
        if in_channels != out_channels or stride[0] != 1:
            if if_first:
                self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      padding=0, groups=1, act=None)
            else:
                self.conv = ConvBNACTWithPool(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              stride=stride, groups=1, act=None)
        elif if_first:
            self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                  padding=0, groups=1, act=None)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first):
        super().__init__()

        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                 if_first=if_first)
        self.relu = nn.ReLU()
        self.output_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv2 = ConvBNACT(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1,
                               padding=0, groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * 4, stride=stride,
                                 if_first=if_first)
        self.relu = nn.ReLU()
        self.output_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class ResNet(nn.Module):
    def __init__(self, in_channels=3, layers=18, **kwargs):
        super().__init__()
        supported_layers = {
            18: {'depth': [2, 2, 2, 2], 'block_class': BasicBlock},
            34: {'depth': [3, 4, 6, 3], 'block_class': BasicBlock},
            50: {'depth': [3, 4, 6, 3], 'block_class': BottleneckBlock},
            101: {'depth': [3, 4, 23, 3], 'block_class': BottleneckBlock},
            152: {'depth': [3, 8, 36, 3], 'block_class': BottleneckBlock},
            200: {'depth': [3, 12, 48, 3], 'block_class': BottleneckBlock}
        }
        assert layers in supported_layers, "supported layers are {} but input layer is {}".format(supported_layers,
                                                                                                  layers)

        depth = supported_layers[layers]['depth']
        block_class = supported_layers[layers]['block_class']

        num_filters = [64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            ConvBNACT(in_channels=in_channels, out_channels=32,
                      kernel_size=3, stride=1, padding=1, act='relu'),
            ConvBNACT(in_channels=32, out_channels=32, kernel_size=3,
                      stride=1, act='relu', padding=1),
            ConvBNACT(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, act='relu', padding=1)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        in_ch = 64
        for block_index in range(len(depth)):
            block_list = []
            for i in range(depth[block_index]):
                if i == 0 and block_index != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)
                block_list.append(block_class(in_channels=in_ch, out_channels=num_filters[block_index],
                                              stride=stride,
                                              if_first=block_index == i == 0))
                in_ch = block_list[-1].output_channels
            self.stages.append(nn.Sequential(*block_list))
        self.out_channels = in_ch
        self.out = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        for stage in self.stages:
            x = stage(x)
        x = self.out(x)
        return x


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)  # (NTC)(batch, width, channels)
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, bidirectional=True, num_layers=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.out_channels = hidden_size

        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        support_encoder_dict = {
            'fc': EncoderWithFC,
            'rnn': EncoderWithRNN
        }
        assert encoder_type in support_encoder_dict, '{} must in {}'.format(
            encoder_type, support_encoder_dict.keys())

        self.encoder = support_encoder_dict[encoder_type](
            self.encoder_reshape.out_channels, hidden_size)
        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        x = self.encoder_reshape(x)
        x = self.encoder(x)
        return x


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super().__init__()
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True)
        self.out_channels = out_channels

    def forward(self, x):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
        return predicts


class RecResNet(nn.Module):
    def __init__(self, in_channels, layers, nclass=60, nh=256,  use_lstm=True):
        super().__init__()
        self.backbone = ResNet(in_channels=in_channels, layers=layers)
        if use_lstm:
            encoder_type = 'rnn'
        else:
            encoder_type = 'fc'
        self.neck = SequenceEncoder(
            self.backbone.out_channels, encoder_type, hidden_size=nh)
        self.head = CTCHead(self.neck.out_channels, nclass)

    def forward(self, x) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 128)
    net = RecResNet(3, 18, 60, 256, False)
    import time
    t0 = time.time()
    out = net(x)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)
    pass
