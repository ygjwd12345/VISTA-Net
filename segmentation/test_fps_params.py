import os
import sys

import torch
import torchvision.transforms as transform
import scipy.io as sio
import encoding.utils as utils
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from torch.utils import data

from encoding.nn import BatchNorm2d
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_model, MultiEvalModule

from option import Options
import time

def test(args):

    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = get_dataset(args.dataset, split=args.split, mode=args.mode,
                          transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)

    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_model(args.model, dataset=args.dataset,
                          backbone=args.backbone, dilated=args.dilated,
                          lateral=args.lateral, attentiongraph=args.attentiongraph, aux=args.aux,
                          se_loss=args.se_loss, norm_layer=BatchNorm2d,
                          base_size=args.base_size, crop_size=args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
    evaluator.eval()
    print(evaluator)
    num_parameters_pre = sum([l.nelement() for l in model.pretrained.parameters()])
    print(num_parameters_pre)
    num_parameters_head = sum([l.nelement() for l in model.head.parameters()])
    print(num_parameters_head)
    print('%e'%(num_parameters_head+num_parameters_pre))
    evaluator.cuda()
    evaluator.eval()
    x = torch.Tensor(1, 3, 512, 512).cuda()

    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = evaluator.parallel_forward(x)

        result = []
        for _ in range(10):
            st = time.time()
            for _ in range(N):
                out = evaluator.parallel_forward(x)
            result.append(N/(time.time()-st))


        print(np.mean(result), np.std(result))


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
