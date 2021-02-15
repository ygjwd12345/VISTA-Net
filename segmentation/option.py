###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
import argparse
import time
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet101',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--attentiongraph', action='store_true', default=
                            False, help='AttentionGraphModel')
        parser.add_argument('--pretrained', action='store_true', default=
                            False, help='add pretrain model from official website')
        parser.add_argument('--dilated', action='store_true', default=
                            False, help='dilation')
        parser.add_argument('--lateral', action='store_true', default=
                            False, help='employ FPN')
        parser.add_argument('--dataset', type=str, default='pcontext',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--dataroot', type=str, default='./datasets', help='dataset root path(raleted root dont forget ./)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--rank',type=int,default=0, help='pga rank ')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--checkpoint-path', type=str, default=None, 
                            help='put the path to save checkpoint if needed')
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--split', default='val')
        parser.add_argument('--mode', default='testval')
        parser.add_argument('--ms', action='store_true', default=False,
                            help='multi scale & flip')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        parser.add_argument('--save-folder', type=str, default='results',
                            help = 'path to save images')
        parser.add_argument('--exp', type=str, default='noname',
                            help = 'experiment name')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'cityscape': 240,
                'pascal_voc': 50,
                'pascal_aug': 50,
                'pcontext': 80,
                'ade20k': 180
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 16
            #args.batch_size = 6
        if args.test_batch_size is None:
            #args.test_batch_size = args.batch_size
            args.test_batch_size = 1
        if args.lr is None:
            lrs = {
                'coco': 0.004,
                'cityscape': 0.004,
                'pascal_voc': 0.0001,
                'pascal_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.004
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        args.checkname=args.model+'_'+args.backbone+'_' + args.dataset + '_'+'rank_'+str(args.rank)+'_'+time.strftime("%Y-%m-%d", time.localtime())
        print(args)
        return args
