from .ctc_loss import CTCLossWarapper


__all__ = ['build_rec_loss']


def build_rec_loss(blank_idx=0, reduction='sum'):
    return CTCLossWarapper(blank_idx, reduction)

