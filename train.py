import os
import numpy as np
import random
import argparse
import datetime

# from matplotlib import pyplot as plt

import torch
import torch.cuda
from torch.optim import lr_scheduler

from models.SparseCoding import get_Drr_Dtheta, DYANEnc
from utils.utils import AverageMeter,accuracy
from models.networks import DYAN_B

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def log(log_line, f_log):
    print(log_line)
    f_log.write(log_line+'\n')

def train_D(args, writers,
            trainloader, testloader,
            mseLoss=torch.nn.MSELoss()):
    # Initialize DYAN Dictionary
    Drr, Dtheta = get_Drr_Dtheta(args.N)
    # Select Network
    if args.wiG:
        # TODO: should be Group DYAN here
        net = DYANEnc(Drr=Drr, Dtheta=Dtheta, lam=args.lam_f,
                      wiRW=args.wiRW, gpu_id=args.gpu_id).cuda(args.gpu_id)
    else:
        net = DYANEnc(Drr=Drr, Dtheta=Dtheta, lam=args.lam_f,
                      wiRW=args.wiRW, gpu_id=args.gpu_id).cuda(args.gpu_id)
    # Training assistants
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    args.dir_save = os.path.join(args.work_dir, args.name_exp,datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(args.dir_save, exist_ok=True)
    f_log = open(os.path.join(args.dir_save,'log.txt'),'w')

    net.train()
    for epoch in range(1, args.epoch_D+1):
        losses, l_mse = AverageMeter(),AverageMeter()
        for i, (data,_,_) in enumerate(trainloader):
            optimizer.zero_grad()
            # batch_size, num_clips, dim_joints, T, num_joints, num_subj
            skeletons = data.cuda(args.gpu_id)
            # => batch_size, num_clips, num_subj, T, num_joints, dim_joints
            skeletons = skeletons.transpose(2,5)
            _,_,S,T,J,D = skeletons.shape 
            skeletons = skeletons.reshape(-1,S,T,J,D)
            N,S,T,_,_ = skeletons.shape
            skeletons = skeletons.reshape(N,S,T,-1)
            if args.wiCY:
                Y = torch.cat((skeletons[:,0,:,:],skeletons[:,1,:,:]),2).cuda(args.gpu_id)
            else:
                Y = skeletons.cuda(args.gpu_id)
            _,_,R = net(Y, T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
            loss_mse = mseLoss(R, Y)
            
            loss = loss_mse
            loss.backward()
            optimizer.step()
            l_mse.update(loss_mse.data.item())
            losses.update(loss.data.item())
            if i%(trainloader.__len__()//10)==0:
                log(f'Train({epoch})|ite({i+1})|loss|{losses.value}|l_mse|{l_mse.value}',f_log)
                writers[0].add_scalar("Loss", losses.value, i+(epoch-1)*len(trainloader))
                writers[0].add_scalar("Loss_MSE", l_mse.value, i+(epoch-1)*len(trainloader))
        scheduler.step()

        torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(args.dir_save, str(epoch) + '.pth'))
        loss, loss_mse, _, _ = test_D(args, testloader, net) 
        log(f'Train({epoch})| loss |{losses.avg}| l_mse |{l_mse.avg}', f_log)
        log(f'Test ({epoch})| loss |{loss}| l_mse |{loss_mse}',f_log)
        writers[1].add_scalar("Loss", loss, epoch*len(trainloader))
        writers[1].add_scalar("Loss_MSE", loss_mse, epoch*len(trainloader))
    log(f"END Timstamp:{datetime.datetime.now().strftime('%Y:%m:%d %H:%M')}",f_log)
    f_log.close()


def test_D(args, dataloader, net,
            mseLoss=torch.nn.MSELoss()):
    losses, l_mse, l_cls, l_bi = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    with torch.no_grad():
        for _, (data,_,_) in enumerate(dataloader):
            skeletons = data.cuda(args.gpu_id)
            # batch_size, num_clips, dim_joints, T, num_joints, num_subj
            skeletons = data.cuda(args.gpu_id)
            # => batch_size, num_clips, num_subj, T, num_joints, dim_joints
            skeletons = skeletons.transpose(2,5)
            _,_,S,T,J,D = skeletons.shape 
            skeletons = skeletons.reshape(-1,S,T,J,D)
            N,S,T,_,_ = skeletons.shape
            skeletons = skeletons.reshape(N,S,T,-1)
            if args.wiCY:
                Y = torch.cat((skeletons[:,0,:,:],skeletons[:,1,:,:]),2).cuda(args.gpu_id)
            else:
                Y = skeletons.cuda(args.gpu_id)
            _,_,R = net(Y, T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
            loss_mse = mseLoss(R, Y)

            loss = loss_mse
            l_mse.update(loss_mse.data.item())
            losses.update(loss.data.item())
        
    return losses.avg, l_mse.avg, l_cls.avg, l_bi.avg


def train_cls(args, writers,
            trainloader, testloader,
            mseLoss=torch.nn.MSELoss(),
            criterion = torch.nn.CrossEntropyLoss(),
            L1Loss=torch.nn.SmoothL1Loss()):
    # Initialize DYAN Dictionary
    Drr, Dtheta = get_Drr_Dtheta(args.N)
    # Select Network
    if args.wiG:
        # TODO: should be Group DYAN here
        net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
    else:
        net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
    # Training assistants
    if args.wiF:
        for p in net.sparseCoding.parameters():
            p.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, net.parameters()), lr=1e-4, weight_decay=0.001, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    args.dir_save = os.path.join(args.work_dir, args.name_exp,datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(args.dir_save, exist_ok=True)
    f_log = open(os.path.join(args.dir_save,'log.txt'),'w')
    f_log.writelines(str(args)+'\n')

    net.train()
    for epoch in range(1, args.Epoch+1):
        losses, l_mse, l_cls, l_bi = AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter()
        accs = AverageMeter()
        for i, (data,label,_) in enumerate(trainloader):
            # batch_size, num_clips, dim_joints, T, num_joints, num_subj
            skeletons = data.cuda(args.gpu_id)
            gt_label = label.cuda(args.gpu_id)
            # => batch_size, num_clips, num_subj, T, num_joints, dim_joints
            skeletons = skeletons.transpose(2,5)
            _,n_clips,S,T,J,D = skeletons.shape 
            skeletons = skeletons.reshape(-1,S,T,J,D)
            N,S,T,_,_ = skeletons.shape
            skeletons = skeletons.reshape(N,S,T,-1)
            if args.wiCY:
                Y = torch.cat((skeletons[:,0,:,:],skeletons[:,1,:,:]),2).cuda(args.gpu_id)
            else:
                Y = skeletons.cuda(args.gpu_id)
            out_label, R, B = net(Y,T) # y: (batch_size, num_clips) x T x ((num_joints x dim_joints) x num_subj)
            out_label = torch.mean(out_label.view(-1, n_clips, args.num_class), 1)
            loss_cls = criterion(out_label, gt_label)
            loss_mse = mseLoss(R, Y)
            loss_bi = L1Loss(torch.zeros_like(B).to(B),B)
            loss = args.lam1*loss_cls + args.lam2*loss_mse + args.Alpha*loss_bi

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accs.update(accuracy(out_label, gt_label)[0].data.item())
            losses.update(loss.data.item())
            l_cls.update(loss_cls.data.item())
            l_bi.update(loss_bi.data.item())
            l_mse.update(loss_mse.data.item())
            if i%(trainloader.__len__()//10)==0:
                log(f'Train|ite({epoch}|{i+1}): |loss|{losses.value}|l_cls|{l_cls.value}'+ 
                    f'|l_bi|{l_bi.value}|l_mse|{l_mse.value}|acc|{accs.value}%', f_log)
                
        scheduler.step()
        log(f'Train({epoch})|acc|{accs.avg}%|loss|{losses.avg}|l_cls|{l_cls.avg}|l_bi|{l_bi.avg}|l_mse|{l_mse.avg}',f_log)
        writers[0].add_scalar("Loss", losses.avg, epoch)
        writers[0].add_scalar("Loss_CLS", l_cls.avg, epoch)
        writers[0].add_scalar("Loss_BI", l_bi.avg, epoch)
        writers[0].add_scalar("Loss_MSE", l_mse.avg, epoch)
        writers[0].add_scalar("Accuracy", accs.avg, epoch)
        if epoch % args.epoch_save == 0:
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(args.dir_save, str(epoch) + '.pth'))
            loss, loss_mse,loss_cls,loss_bi,acc = test_cls(args, testloader, net)
            log(f'Test ({epoch})|acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
            writers[1].add_scalar("Accuracy", acc, epoch)
            writers[1].add_scalar("Loss", loss, epoch)
            writers[1].add_scalar("Loss_CLS", loss_cls, epoch)
            writers[1].add_scalar("Loss_BI",  loss_bi,  epoch)
            writers[1].add_scalar("Loss_MSE", loss_mse, epoch)
    log(f"END Timstamp:{datetime.datetime.now().strftime('%Y:%m:%d %H:%M')}",f_log)
    f_log.close()


def test_cls(args, dataloader, net,
            withMask=False, gumbel_thresh=False,
            mseLoss=torch.nn.MSELoss(),
            criterion=torch.nn.CrossEntropyLoss(),
            L1Loss=torch.nn.SmoothL1Loss()):
    losses, l_mse, l_cls, l_bi = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        for _, (data,label,index) in enumerate(dataloader):
            skeletons = data.cuda(args.gpu_id)
            gt_label = label.cuda(args.gpu_id)
            # => batch_size, num_clips, num_subj, T, num_joints, dim_joints
            skeletons = skeletons.transpose(2,5)
            _,n_clips,S,T,J,D = skeletons.shape 
            skeletons = skeletons.reshape(-1,S,T,J,D)
            N,S,T,_,_ = skeletons.shape
            skeletons = skeletons.reshape(N,S,T,-1)
            if args.wiCY:
                Y = torch.cat((skeletons[:,0,:,:],skeletons[:,1,:,:]),2).cuda(args.gpu_id)
            else:
                Y = skeletons.cuda(args.gpu_id)
            # y: (batch_size, num_clips) x T x ((num_joints x dim_joints) x num_subj)
            out_label, R, B = net(Y,T) 
            out_label = torch.mean(out_label.view(-1, n_clips, args.num_class), 1)
            loss_cls = criterion(out_label, gt_label)
            loss_mse = mseLoss(R, Y)
            loss_bi = L1Loss(torch.zeros_like(B).to(B),B)
            loss = args.lam1*loss_cls + args.lam2*loss_mse + args.Alpha*loss_bi

            accs.update(accuracy(out_label, gt_label)[0].data.item())
            l_bi.update(loss_bi.data.item())
            l_mse.update(loss_mse.data.item())
            l_cls.update(loss_cls.data.item())
            losses.update(loss.data.item())
        
    return losses.avg, l_mse.avg, l_cls.avg, l_bi.avg, accs.avg


def get_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='CVAC Interaction')
    parser.add_argument('--work_dir', default='/data/dluo/work_dir/2312_CVAC_NTU-Inter',
        help='the work folder for storing results')
    # random seed
    parser.add_argument('--seed', default=0, type=int)
    # GPU
    parser.add_argument('--gpu_id', default=7, type=int)
    
    # Dataset
    parser.add_argument('--dataset', default='NTU')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--data_dim', default=3, type=int)
    parser.add_argument('--num_class', default=11, type=int)
    parser.add_argument('--NI', action='store_true', help='not NTU-Inter')
    parser.add_argument('--sampling', default='multi',help='sampling strategy')
    parser.add_argument('--withMask', default=False, type=str2bool)
    parser.add_argument('--maskType', default='score')
    parser.add_argument('--bs', default=8, type=int) # 32/6 -> Single/Multi
    parser.add_argument('--num_workers', default=4, type=int) # 8/4 -> Single/Multi
    # DYAN Dictionary
    parser.add_argument('--N', default=80*2, type=int, help='number of poles')
    parser.add_argument('--T', default=36, type=int, help='input clip length')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--wiRW', default=True, type=str2bool, help='Reweighted DYAN')
    # parser.add_argument('--wiD',default='/data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_wiCY_woG_woCC_woBI_woCL_T36_f0.1/20240224_1801/5.pth') # woRW
    parser.add_argument('--wiD',default='/data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f0.1_2_1_1_0.1/20240313_1559/5.pth') # wiRW
    # Architecture
    parser.add_argument('--mode', default='cls')  # D | cls
    parser.add_argument('--wiCY', default=True, type=str2bool, help='Concatenated Y')
    parser.add_argument('--wiG',  default=False, type=str2bool, help='Use GroupLASSO')
    parser.add_argument('--wiCC', default=True, type=str2bool, help='Concatenated Coefficients')
    parser.add_argument('--wiF',  default=True, type=str2bool, help='Freeze DYAN Reconstruction net')
    parser.add_argument('--wiBI', default=True, type=str2bool, help='Use Binarization Code')
    parser.add_argument('--wiCL', default=False, type=str2bool, help='Use Contrast Learning')
    parser.add_argument('--setup',default='cv')
    # Training
    parser.add_argument('--Epoch', default=100, type=int)
    parser.add_argument('--epoch_D', default=5, type=int) # before this epoch only train Reconstruction Part
    parser.add_argument('--epoch_save', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='classifier')
    parser.add_argument('--lr_2', default=1e-3, type=float, help='sparse coding')

    parser.add_argument('--th_gumbel', default=0.2, type=float, help='threshold for Gumbel Module') 
    parser.add_argument('--Alpha', default=1, type=float, help='loss_bi')
    parser.add_argument('--lam1', default=2, type=float, help='loss_cls') # 2
    parser.add_argument('--lam2', default=1, type=float, help='loss_mse')
    

    return parser


# https://stackoverflow.com/questions/48951136/plot-multiple-graphs-in-one-plot-using-tensorboard
# layout_w = {"args": {
#                 "Loss": ["Multiline", ["Loss/train", "Loss/test"]],
#                 "Loss_mse": ["Multiline", ["Loss_MSE/train", "Loss_MSE/test"]],
#                 },
#             }
# writer.add_custom_scalars(layout_w)

# layout_w = {"args": {
#                 "loss": ["Multiline", ["loss/train", "loss/test"]],
#                 "loss_cls": ["Multiline", ["loss/train_cls", "loss/test_cls"]],
#                 "loss_mse": ["Multiline", ["loss/train_mse", "loss/test_mse"]],
#                 "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
#                 },
#             }
# writer.add_custom_scalars(layout_w)