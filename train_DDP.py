from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from torch.utils.data import DataLoader
import copy
from datetime import datetime

from backbones import __models__
from datasets import __datasets__
from disparity_estimators import __disparity_estimator__
from losses import __loss__

torch.backends.cudnn.benchmark = True

# multiprocessing distributed training
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp


def get_parser():
    parser = argparse.ArgumentParser(description='Muti-Modal Groundtruth Distribution')
    parser.add_argument('--model', default='PSMNet', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192,help='maxium disparity')

    parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', required=True, default='/data0/xp/Scence_Flow/',help='data path')
    parser.add_argument('--trainlist', required=True, help='training list')

    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='training batch size')

    parser.add_argument('--savemodeldir', required=True, default='/data0/xp/Check_Point/MMGD/',help='the directory to save logs and checkpoints')
    parser.add_argument('--model_name',default='PSMNet',help='log name')    
    parser.add_argument('--loadmodel', help='load the weights from a specific checkpoint')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')

    # disparity estimator and loss function
    parser.add_argument('--estimator',default='softargmax',help='disparity regression methods',choices=__disparity_estimator__.keys())
    parser.add_argument('--loss_func',default='SL1',help='loss function',choices=__loss__.keys())


    # for distributed training
    parser.add_argument('--btrain', '-btrain', type=int, default=None)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()

    args.btrain = args.batch_size
    args.start_epoch = 0

    if not args.dist_url:
        args.dist_url = "tcp://127.0.0.1:{}".format(random_int() % 30000)

    return args

    
def main():
    
    args = get_parser()

    reset_seed(args.seed)

    ## distributed training
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    args.ngpus_per_node = ngpus_per_node

    args.distributed = ngpus_per_node > 1 and (args.world_size > 1 or args.multiprocessing_distributed)
    args.multiprocessing_distributed = args.distributed

    if args.distributed and args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args)


def main_process(args):
    return (not args.multiprocessing_distributed) or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main_worker(gpu, ngpus_per_node, args):
    print("Using GPU: {} for training".format(gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # --------------Model------------------
    if args.model is not None:
        model = __models__[args.model](args.maxdisp)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.btrain = int(args.btrain / ngpus_per_node)
#         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=False)

    elif ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    train_dataset = StereoDataset(args.datapath, args.trainlist, True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    TrainImgLoader = DataLoader(
        train_dataset, 
        batch_size=args.btrain, shuffle=(train_sampler is None), num_workers=4, drop_last=True,
        sampler=train_sampler)

    if args.loadmodel is not None:
        if args.distributed:
            state_dict = torch.load(args.loadmodel,map_location=model.device)
            model.load_state_dict(state_dict['state_dict'], strict=False)
        else:
            state_dict = torch.load(args.loadmodel)
            model.load_state_dict(state_dict['state_dict'], strict=False)

        if 'optimizer' in state_dict:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
            except Exception as e:
                print('fail to load optimizer')
        else:
            if main_process(args):
                print('No saved optimizer')

    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
                
        if args.distributed:
            train_sampler.set_epoch(epoch)

        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch, args=args)

        for batch_idx, sample in enumerate(TrainImgLoader):
            loss = train(model,optimizer,sample['left'],sample['right'],sample['disparity'],args,gpu)

            if main_process(args):
                total_train_loss += loss

    
        if main_process(args):
            print('\nepoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

            # ---------save loss-------------
            if args.dataset == 'sceneflow':
                logdir = './log/SceneFlow/'
            elif args.dataset == 'kitti':
                logdir = './log/KITTI/'

        
            with open(logdir+args.model_name+'.txt','a+') as f:
                f.write(str(epoch)+'\t')
                f.write(str(total_train_loss/len(TrainImgLoader))+'\n')
                f.close()
            #-------------- save model -----------------------------
            savefilename = args.savemodeldir+args.model_name+'_train_'+str(epoch)+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
                'optimizer': optimizer.state_dict()
            }, savefilename)


def train(model,optimizer,imgL,imgR,disp_L,args,gpu):
    imgL, imgR, disp_true = imgL.cuda(gpu), imgR.cuda(gpu), disp_L.cuda(gpu)

    model.train()

    mask =  (disp_true < args.maxdisp) * (disp_true > 0) 
    mask.detach_()
    if mask.sum() == 0:
        return float(0)

    disp_true[~mask] = 0

    optimizer.zero_grad()

    loss_func = __loss__[args.loss_func]
    regression = __disparity_estimator__[args.estimator](args.maxdisp)   # SL1 loss
    if args.model == 'GANet':
        regression = __disparity_estimator__[args.estimator](args.maxdisp+1)     # The search range of GANet is [0,192] which has a total of 193 disparity candidates


    if args.model == 'PSMNet' and args.loss_func != 'SL1':
        output1, output2, output3 = model(imgL, imgR)
        loss = 0.5 * loss_func(output1,disp_true,mask,args.maxdisp) \
            + 0.7 * loss_func(output2,disp_true,mask,args.maxdisp) \
            + 1.0 * loss_func(output3,disp_true,mask,args.maxdisp)
    elif args.model == 'PSMNet' and args.loss_func == 'SL1':
        output1, output2, output3 = model(imgL, imgR)
        output1 = regression(output1)
        output2 = regression(output2)
        output3 = regression(output3)
        loss = 0.5 * loss_func(output1[mask], disp_true[mask], reduction='mean')     \
            + 0.7 * loss_func(output2[mask], disp_true[mask], reduction='mean')      \
            + 1.0 * loss_func(output3[mask], disp_true[mask],reduction='mean')
    elif (args.model == 'GwcNet_G' or args.model == 'GwcNet_GC') and args.loss_func != 'SL1':
        output0, output1, output2, output3 = model(imgL, imgR)
        loss = 0.5 * loss_func(output0,disp_true,mask,args.maxdisp) \
            + 0.5 * loss_func(output1,disp_true,mask,args.maxdisp) \
            + 0.7 * loss_func(output2,disp_true,mask,args.maxdisp) \
            + 1.0 * loss_func(output3,disp_true,mask,args.maxdisp)
    elif (args.model == 'GwcNet_G' or args.model == 'GwcNet_GC') and args.loss_func == 'SL1':
        output0, output1, output2, output3 = model(imgL, imgR)
        output0 = regression(output0)
        output1 = regression(output1)
        output2 = regression(output2)
        output3 = regression(output3)
        loss = 0.5 * loss_func(output0[mask], disp_true[mask], reduction='mean')     \
            + 0.5 * loss_func(output1[mask], disp_true[mask], reduction='mean')     \
            + 0.7 * loss_func(output2[mask], disp_true[mask], reduction='mean')      \
            + 1.0 * loss_func(output3[mask], disp_true[mask],reduction='mean')
    elif args.model == 'GANet' and args.loss_func != 'SL1':
        output1, output2, output3 = model(imgL, imgR)
        loss = 0.2 * loss_func(output1,disp_true,mask,args.maxdisp+1) \
            + 0.6 * loss_func(output2,disp_true,mask,args.maxdisp+1) \
            + 1.0 * loss_func(output3,disp_true,mask,args.maxdisp+1)
    elif args.model == 'GANet' and args.loss_func == 'SL1':
        output1, output2, output3 = model(imgL, imgR)
        output1 = regression(output1)
        output2 = regression(output2)
        output3 = regression(output3)
        loss = 0.2 * loss_func(output1[mask], disp_true[mask], reduction='mean')     \
            + 0.6 * loss_func(output2[mask], disp_true[mask], reduction='mean')      \
            + 1.0 * loss_func(output3[mask], disp_true[mask],reduction='mean')

        
    loss.backward()
    optimizer.step()

    return loss.item()

def adjust_learning_rate(optimizer,epoch,args):
    lr = args.lr
    if args.dataset == 'sceneflow':
        if epoch >= 30:
            lr = args.lr / 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def random_int(obj=None):
    return (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295

if __name__ == '__main__':
    main()