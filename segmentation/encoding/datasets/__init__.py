from .base import *
from .pcontext import ContextSegmentation
from .kitti import KITTIDataset
from .cityscapes import CitySegmentation
from .ade20k import ADE20KSegmentation
from .coco import COCOSegmentation
from  .pascal_voc import VOCSegmentation
from  .pascal_aug import VOCAugSegmentation
datasets = {
    'pcontext': ContextSegmentation,
    'kitti': KITTIDataset,
    'cityscape': CitySegmentation,
    'ade20k': ADE20KSegmentation,
    'coco': COCOSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
