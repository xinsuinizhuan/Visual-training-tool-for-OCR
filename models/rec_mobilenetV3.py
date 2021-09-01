import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ['RecMobileNetV3']


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HardSigmoid(nn.Module):
    def forward(self, x):
        x = F.relu6(x + 3, inplace=True) / 6
        return x


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


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super().__init__()
        num_mid_filter = out_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter,
                               kernel_size=1, out_channels=out_channels, bias=True)
        self.relu2 = HardSigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                               padding=0, act=act)

        self.conv1 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                               stride=stride,
                               padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter,
                              out_channels=num_mid_filter)
        else:
            self.se = None

        self.conv2 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                               padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)
        if not self.not_add:
            y = x + y
        return y


class MobileNetV3(nn.Module):
    def __init__(self, in_channels=3, scale=0.5, model_name='large'):
        super().__init__()
        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', (2, 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (2, 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (2, 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (2, 1)],
                [3, 72, 24, False, 'relu', (2, 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (2, 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, "supported scale are {} but input scale is {}".format(supported_scale,
                                                                                               scale)

        inplanes = 16
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(
                                   inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        inplanes = self.make_divisible(inplanes * scale)
        block_list = []
        for (k, exp, c, se, nl, s) in cfg:
            block_list.append(ResidualUnit(num_in_filter=inplanes,
                                           num_mid_filter=self.make_divisible(
                                               scale * exp),
                                           num_out_filter=self.make_divisible(
                                               scale * c),
                                           act=nl,
                                           stride=s,
                                           kernel_size=k,
                                           use_se=se))
            inplanes = self.make_divisible(scale * c)
        self.blocks = nn.Sequential(*block_list)
        self.conv2 = ConvBNACT(in_channels=inplanes,
                               out_channels=self.make_divisible(
                                   scale * cls_ch_squeeze),
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act='hard_swish')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = self.make_divisible(scale * cls_ch_squeeze)

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class Im2Seq(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
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
    def __init__(self, in_channels, encoder_type, hidden_size=48):
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
    def __init__(self, in_channels, out_channels):
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


class RecMobileNetV3(nn.Module):
    def __init__(self, in_channels, model_name='large', nclass=60, nh=256, use_lstm=True):
        super().__init__()
        self.backbone = MobileNetV3(
            in_channels=in_channels, model_name=model_name)
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
    x = torch.randn(1, 3, 32, 256)
    net = RecMobileNetV3(3, model_name='small')
    out = net(x)
    print(out.shape)
    pass
