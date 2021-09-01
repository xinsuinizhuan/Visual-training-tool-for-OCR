from utils import CTCLabelConverter
from .rec_metric import RecMetric

__all__ = ['build_rec_metric']


def build_rec_metric(converter: CTCLabelConverter):
    return RecMetric(converter)
