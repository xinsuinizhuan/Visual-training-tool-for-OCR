import torch
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ['RecCRNN']


class CRNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(
                True), nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(
                True), nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(128, 256, 3, (2, 1), 1), nn.BatchNorm2d(
                256), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(
                True), nn.MaxPool2d(2, 1, 1),  # 256x4x25
            nn.Conv2d(256, 512, 3, (2, 1), 1), nn.BatchNorm2d(
                512), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(
                True), nn.MaxPool2d(2, 1, 1),  # 512x2x25
            nn.Conv2d(512, 512, 3, (2, 1), 0), nn.BatchNorm2d(512), nn.ReLU(True))  # 512x1x25
        self.out_channels = 512

    def forward(self, x):
        x = self.cnn(x)
        return x


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
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


class RecCRNN(nn.Module):
    def __init__(self, in_channels=3, nclass=60, nh=256, use_lstm=True):
        super().__init__()
        self.backbone = CRNN(in_channels)
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
    net = RecCRNN(3, 60, 256, False)
    import time
    t0 = time.time()
    out = net(x)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)

