from .rec_resnetvd import RecResNet
from .rec_crnn import RecCRNN
from .rec_mobilenetV3 import RecMobileNetV3


__all__ = ['build_rec_model']


def build_rec_model(cfg, nclass):
    if cfg.model_type == 'crnn':
        model = RecCRNN(3, nclass=nclass, nh=cfg.nh, use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'resnet18':
        model = RecResNet(3, 18, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'resnet34':
        model = RecResNet(3, 34, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'resnet50':
        model = RecResNet(3, 50, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'resnet101':
        model = RecResNet(3, 101, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'resnet152':
        model = RecResNet(3, 152, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'resnet200':
        model = RecResNet(3, 200, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'mbv3_large':
        model = RecMobileNetV3(
            3, nclass=nclass, nh=cfg.nh, use_lstm=cfg.use_lstm)
    elif cfg.model_type == 'mbv3_small':
        model = RecMobileNetV3(
            3, nclass=nclass, nh=cfg.nh, use_lstm=cfg.use_lstm, model_name='small')
    else:
        model = RecResNet(3, 18, nclass=nclass, nh=cfg.nh,
                          use_lstm=cfg.use_lstm)
    return model
