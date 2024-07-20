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

from backbones import __models__
from datasets import __datasets__
from disparity_estimators import __disparity_estimator__
from losses import __loss__

import time


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Muti-Modal Groundtruth Distribution')
parser.add_argument('--model', default='PSMNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192,help='maxium disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, default='/data0/xp/Scence_Flow/',help='data path')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--start_model', type=int, required=True)
parser.add_argument('--end_model', type=int, required=True)
parser.add_argument('--gap', type=int, required=True)
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')


parser.add_argument('--loadmodel', help='load the weights from a specific checkpoint')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')

parser.add_argument('--estimator',default='softargmax',help='disparity regression methods',choices=__disparity_estimator__.keys())

parser.add_argument('--model_name',default='PSMNet',help='log name')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model
if args.model is not None:
    model = __models__[args.model](args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()


# disparity regression methods
regression = __disparity_estimator__[args.estimator](args.maxdisp)
if args.model == 'GANet':
    regression = __disparity_estimator__[args.estimator](args.maxdisp+1)



def test(imgL,imgR,disp_true,masknocc=None):
    
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    if masknocc is not None:
        masknocc = masknocc.cuda()
        disp_true[masknocc!=1] = 0
    mask = (disp_true < args.maxdisp) * (disp_true > 0)

    with torch.no_grad():
        output3 = model(imgL,imgR)
        output3 = regression(output3)
        img = torch.squeeze(output3,1)            

    if len(disp_true[mask]) == 0:
        loss = torch.Tensor([0]).cuda()
        loss_3px = float(0)
        loss_3px_5 = float(0)
        loss_1px = float(0)
        loss_2px = float(0)
        loss_4px = float(0)
    else:
        if args.estimator == 'argmax':
            img = img.to(torch.float32)
        loss = F.l1_loss(img[mask],disp_true[mask]) 
    
        pred_disp = img.data.cpu()
        disp_true = disp_true.data.cpu()

        true_disp = copy.deepcopy(disp_true)
        index = np.argwhere((true_disp < args.maxdisp) * (true_disp > 0))

        pred_disp.reshape(disp_true.size())

        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)
        correct_5 = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)
        loss_3px = 1-(float(torch.sum(correct))/float(len(index[0])))
        loss_3px_5 = 1-(float(torch.sum(correct_5))/float(len(index[0])))

        correct_4 = (disp_true[index[0][:], index[1][:], index[2][:]] < 4)
        loss_4px = 1-(float(torch.sum(correct_4))/float(len(index[0])))

        correct_2 = (disp_true[index[0][:], index[1][:], index[2][:]] < 2)
        loss_2px = 1-(float(torch.sum(correct_2))/float(len(index[0])))

        correct_1 = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)
        loss_1px = 1-(float(torch.sum(correct_1))/float(len(index[0])))


    return loss.item(), loss_1px, loss_2px, loss_3px, loss_3px_5, loss_4px


def main():

    min_test_loss = 1000
    best_model = 0
    for i in np.arange(args.start_model,args.end_model+1,args.gap):

        loadmodel = args.loadmodel + "_train_{}.tar".format(i)
        print('Load pretrained model')
        pretrain_dict = torch.load(loadmodel)
        model.load_state_dict(pretrain_dict['state_dict'], strict=False)
        
        # saved model name
        model_name = args.model_name

        #------------- TEST ------------------------------------------------------------
        total_test_loss = 0
        total_test_3px = 0
        total_test_3px_5 = 0
        total_test_1px = 0
        total_test_2px = 0
        total_test_4px = 0
        
        if i >= 0:
            start_time = time.perf_counter()
            for batch_idx, sample in enumerate(TestImgLoader):
                if args.dataset == 'middlebury' or args.dataset == 'eth3d':
                    test_loss, test_1px, test_2px, test_3px, test_3px_5, test_4px = test(sample['left'],sample['right'],sample['disparity'],sample['mask'])
                else:
                    test_loss, test_1px, test_2px, test_3px, test_3px_5, test_4px = test(sample['left'],sample['right'],sample['disparity'])        
                        
                total_test_loss += test_loss
                total_test_3px += test_3px
                total_test_3px_5 += test_3px_5
                total_test_1px += test_1px
                total_test_2px += test_2px
                total_test_4px += test_4px
                
            end_time = time.perf_counter()

            print('total test loss = %.6f' %(total_test_loss/len(TestImgLoader)))
            print('total test 1px = %.6f' %(total_test_1px/len(TestImgLoader)*100))
            print('total test 2px = %.6f' %(total_test_2px/len(TestImgLoader)*100))
            print('total test 3px = %.6f' %(total_test_3px/len(TestImgLoader)*100))
            print('total test 3px_5 = %.6f' %(total_test_3px_5/len(TestImgLoader)*100))
            print('total test 4px = %.6f' %(total_test_4px/len(TestImgLoader)*100))
            print('time per frame = %.6f' %((end_time - start_time)/len(TestImgLoader)/args.test_batch_size))


            if args.dataset == 'sceneflow':
                if total_test_loss/len(TestImgLoader) < min_test_loss:
                    min_test_loss = total_test_loss/len(TestImgLoader)
                    best_model = i
                print('Best EPE error = %.6f at Model %d' %(min_test_loss, best_model))
                print('--------------')
            elif args.dataset == 'kitti':
                if total_test_3px_5/len(TestImgLoader)*100 < min_test_loss:
                    min_test_loss = total_test_3px_5/len(TestImgLoader)*100
                    best_model = i
                print('Best D1 error = %.6f at Model %d' %(min_test_loss, best_model))
                print('--------------')              
            elif args.dataset == 'middlebury':
                if total_test_2px/len(TestImgLoader)*100 < min_test_loss:
                    min_test_loss = total_test_2px/len(TestImgLoader)*100
                    best_model = i
                print('Best 2px error = %.6f at Model %d' %(min_test_loss, best_model))
                print('--------------')
            elif args.dataset == 'eth3d':
                if total_test_1px/len(TestImgLoader)*100 < min_test_loss:
                    min_test_loss = total_test_1px/len(TestImgLoader)*100
                    best_model = i
                print('Best 1px error = %.6f at Model %d' %(min_test_loss, best_model))
                print('--------------')   

        #-------------- SAVE  information -------------------------------------------
        if args.dataset == 'sceneflow':
            logdir = './log/SceneFlow/'
        elif args.dataset == 'kitti':
            logdir = './log/KITTI/'
        elif args.dataset == 'middlebury':
            logdir = './log/Middlebury/'
        elif args.dataset == 'eth3d':
            logdir = './log/ETH3D/'
            
        with open(logdir+model_name+'.txt','a+') as f:
            f.write(str(i)+'\t')
            f.write(str(total_test_loss/len(TestImgLoader))+'\t')
            f.write(str(100*total_test_1px/len(TestImgLoader))+'\t')
            f.write(str(100*total_test_2px/len(TestImgLoader))+'\t')
            f.write(str(100*total_test_3px/len(TestImgLoader))+'\t')
            f.write(str(100*total_test_3px_5/len(TestImgLoader))+'\t')
            f.write(str(100*total_test_4px/len(TestImgLoader))+'\t')
            f.write(str((end_time - start_time)/len(TestImgLoader)/args.test_batch_size)+'\n')
            f.close()
    
    if args.dataset == 'sceneflow':
        with open('./log/SceneFlow/'+model_name+'.txt','a+') as f:
            f.write('%s\n' %(args.model))
            f.write('%s\n' %(args.estimator))
            f.write('Best EPE error = %.6f at Model %d\n' %(min_test_loss, best_model))
            f.write('--------------\n')
            f.close()
    elif args.dataset == 'kitti':
        with open('./log/KITTI/'+model_name+'.txt','a+') as f:
            f.write('%s\n' %(args.model))
            f.write('Best D1 error = %.6f at Model %d\n' %(min_test_loss, best_model))
            f.write('--------------\n')
            f.close()
    elif args.dataset == 'middlebury':
        with open('./log/Middlebury/'+model_name+'.txt','a+') as f:
            f.write('%s\n' %(args.model))
            f.write('Best 2px error = %.6f at Model %d\n' %(min_test_loss, best_model))
            f.write('--------------\n')
            f.close()
    elif args.dataset == 'eth3d':
        with open('./log/ETH3D/'+model_name+'.txt','a+') as f:
            f.write('%s\n' %(args.model))
            f.write('Best 1px error = %.6f at Model %d\n' %(min_test_loss, best_model))
            f.write('--------------\n')
            f.close()
            
if __name__ == '__main__':
   main()