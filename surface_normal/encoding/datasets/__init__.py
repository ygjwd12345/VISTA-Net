from .base import *
from .pcontext import ContextSegmentation
from .kitti import KITTIDataset

datasets = {
    'pcontext': ContextSegmentation,
    'kitti': KITTIDataset,
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
