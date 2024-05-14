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
from utils.utils import AverageMeter,accuracy, sparsity, affine_aug
from models.networks import DYAN_B

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='CVAC Interaction')
    parser.add_argument('--work_dir', default='/data/dluo/work_dir/2312_CVAC_NTU-Inter',
        help='the work folder for storing results')
    parser.add_argument('--cus_n', default='', help='customized name')
    # random seed
    parser.add_argument('--seed', default=0, type=int)
    # GPU
    parser.add_argument('--gpu_id', default=7, type=int)
    # Dataset
    parser.add_argument('--dataset', default='NTU')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--setup', default='cv')
    parser.add_argument('--data_dim', default=3, type=int)
    parser.add_argument('--num_class', default=11, type=int)
    parser.add_argument('--wiL', default='0', type=str2bool, help='Limb(middle point of joints)')
    parser.add_argument('--NI', action='store_true', help='NTU-Inter')
    parser.add_argument('--sampling', default='multi',help='sampling strategy')
    parser.add_argument('--withMask', default='0', type=str2bool)
    parser.add_argument('--maskType', default='score')
    parser.add_argument('--bs', default=8, type=int) # 32/6 -> Single/Multi
    parser.add_argument('--num_workers', default=4, type=int) # 8/4 -> Single/Multi
    # DYAN Dictionary
    parser.add_argument('--N', default=80*2, type=int, help='number of poles')
    parser.add_argument('--T', default=36, type=int, help='input clip length')
    parser.add_argument('--lam_f', default=1.7, type=float) # 1e-1
    parser.add_argument('--th_d', default=1e-5, type=float)
    parser.add_argument('--wiRW', default='0', type=str2bool, help='Reweighted DYAN')
    parser.add_argument('--wiBI', default='0', type=str2bool, help='Use Binarization Code')
    parser.add_argument('--th_g', default=0.5, type=float, help='threshold for Gumbel Module')
    parser.add_argument('--te_g', default=0.01, type=float, help='temparature for Gumbel Module')
    parser.add_argument('--wiD',default='')
    # parser.add_argument('--wiD',default='/data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_wiCY_woG_woCC_woBI_woCL_T36_f0.1/20240224_1801/5.pth') # woRW
    # parser.add_argument('--wiD',default='/data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f0.1_2_1_1_0.1/20240313_1559/5.pth') # wiRW
    # Architecture
    parser.add_argument('--mode', default='D')  # D | cls
    parser.add_argument('--wiCY', default='1', type=str2bool, help='Concatenated Y')
    parser.add_argument('--wiG',  default='0', type=str2bool, help='Use GroupLASSO')
    parser.add_argument('--wiCC', default='1', type=str2bool, help='Concatenated Coefficients')
    parser.add_argument('--wiF',  default='1', type=str2bool, help='Freeze DYAN Reconstruction net')
    parser.add_argument('--wiAff',default='0', type=str2bool, help='Affine Augmentation')
    parser.add_argument('--trs', default='1.0,0.3,0.5', type=str, help='Data Augmentation setup')
    parser.add_argument('--wiCL', default='0', type=str2bool, help='Use Contrast Learning')
    
    parser.add_argument('--pret',default='')
    # parser.add_argument('--pret',default='/data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2/20240415_1632/95.pth')
    # Training
    parser.add_argument('--ep', default=100, type=int)
    parser.add_argument('--ep_D', default=20, type=int) # before this epoch only train Reconstruction Part
    parser.add_argument('--ep_save', default=5, type=int)
    parser.add_argument('--save_m', default='0', type=str2bool)
    parser.add_argument('--lr', default=1e-4, type=float, help='classifier')
    parser.add_argument('--lr_2', default=1e-3, type=float, help='Linear networks')

    parser.add_argument('--Alpha', default=1, type=float, help='loss_bi')
    parser.add_argument('--lam1', default=2, type=float, help='loss_cls') # 2
    parser.add_argument('--lam2', default=1, type=float, help='loss_mse')
    parser.add_argument('--ms', default='15,20', type=str, help='step milestones')

    return parser


def log(log_line, f_log):
    print(log_line)
    f_log.write(log_line+'\n')


def train_D(args, writers,
            trainloader, testloader):
    # Initialize DYAN Dictionary
    Drr, Dtheta = get_Drr_Dtheta(args.N)
    # Select Network
    net = DYANEnc(args, Drr, Dtheta).cuda(args.gpu_id)
    mseLoss=torch.nn.MSELoss()
    L1Loss=torch.nn.SmoothL1Loss()
    # Training assistants
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)


    args.dir_save = os.path.join(args.work_dir,
                                 args.name_exp,
                                 datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.dir_save, exist_ok=True)
    f_log = open(os.path.join(args.dir_save,'log.txt'),'w')
    for arg in vars(args):  log(f"{arg}: {str(getattr(args, arg))}",f_log)

    if args.save_m:
        torch.save({'state_dict': net.state_dict()},
                    os.path.join(args.dir_save, '0.pth'))
    loss, loss_mse, loss_bi, sp_0, sp_th = test_D(args, testloader, net)
    log(f'Test ({0})| loss |{loss}| l_mse |{loss_mse}| l_bi |{loss_bi}| Sp_0 |{sp_0}| Sp_th |{sp_th}',f_log)
    writers[1].add_scalar("Loss",       loss,     0)
    writers[1].add_scalar("Loss_MSE",   loss_mse, 0)
    writers[1].add_scalar("Loss_BI",    loss_bi,  0)
    writers[1].add_scalar("Sparsity_0", sp_0,     0)
    writers[1].add_scalar("Sparsity_th",sp_th,    0)
    
    for epoch in range(1, args.ep_D+1):
        net.train()
        losses, l_mse, l_bi = AverageMeter(), AverageMeter(), AverageMeter()
        Sp_0, Sp_th = AverageMeter(),AverageMeter()
        for i, (data,_,_) in enumerate(trainloader):
            if args.wiAff:  data = affine_aug(data, args.trs[0], args.trs[1], args.trs[2])
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
            C, _, R, B = net(Y, T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
            loss_mse = mseLoss(R, Y)
            loss_bi = L1Loss(torch.zeros_like(B).to(B),B)
            
            loss = args.lam2*loss_mse + args.Alpha*loss_bi
            # C.register_hook(lambda grad: print(grad))
            # B.register_hook(lambda grad: print(grad))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l_mse.update(loss_mse.data.item())
            l_bi.update(loss_bi.data.item())
            losses.update(loss.data.item())

            # Sparsity Indicator
            sp_0, sp_th = sparsity(C)
            Sp_0.update(sp_0.data.item())
            Sp_th.update(sp_th.data.item())
            if i%(trainloader.__len__()//10)==0:
                log(f'Train({epoch})|ite({i+1})|loss|{losses.value}|l_mse|{l_mse.value}|l_bi|{l_bi.value}',f_log)
                writers[0].add_scalar("Loss", losses.value, i+(epoch-1)*len(trainloader))
                writers[0].add_scalar("Loss_MSE", l_mse.value,  i+(epoch-1)*len(trainloader))
                writers[0].add_scalar("Loss_BI",  l_bi.value,   i+(epoch-1)*len(trainloader))
                writers[0].add_scalar("Sparsity_0", Sp_0.value, i+(epoch-1)*len(trainloader))
                writers[0].add_scalar("Sparsity_th",Sp_th.value,i+(epoch-1)*len(trainloader))
        scheduler.step()
        log(f'Train({epoch})| loss |{losses.avg}| l_mse |{l_mse.avg}| l_bi |{l_bi.avg}| Sp_0 |{Sp_0.avg}| Sp_th |{Sp_th.avg}', f_log)
        
        loss, loss_mse, loss_bi, sp_0, sp_th = test_D(args, testloader, net) 
        log(f'Test ({epoch})| loss |{loss}| l_mse |{loss_mse}| l_bi |{loss_bi}| Sp_0 |{sp_0}| Sp_th |{sp_th}',f_log)
        writers[1].add_scalar("Loss", loss, epoch*len(trainloader))
        writers[1].add_scalar("Loss_MSE", loss_mse, epoch*len(trainloader))
        writers[1].add_scalar("Loss_BI", loss_bi, epoch*len(trainloader))
        writers[1].add_scalar("Sparsity_0", sp_0, i+(epoch-1)*len(trainloader))
        writers[1].add_scalar("Sparsity_th", sp_th, i+(epoch-1)*len(trainloader))

        if args.save_m:
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(), 'args':args},
                        os.path.join(args.dir_save, str(epoch) + '.pth'))
    log(f"END Timstamp:{datetime.datetime.now().strftime('%Y:%m:%d %H:%M')}",f_log)
    f_log.close()


def test_D(args, dataloader, net):
    mseLoss=torch.nn.MSELoss()
    L1Loss=torch.nn.SmoothL1Loss()

    losses, l_mse, l_bi = AverageMeter(), AverageMeter(), AverageMeter()
    Sp_0, Sp_th = AverageMeter(),AverageMeter()
    with torch.no_grad():
        net.eval()
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
            C,_,R,B = net(Y, T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
            loss_mse = mseLoss(R, Y)
            loss_bi = L1Loss(torch.zeros_like(B).to(B),B)

            loss = args.lam2*loss_mse + args.Alpha*loss_bi
            l_mse.update(loss_mse.data.item())
            l_bi.update(loss_bi.data.item())
            losses.update(loss.data.item())
            # Sparsity Indicator
            sp_0, sp_th = sparsity(C)
            Sp_0.update(sp_0.data.item())
            Sp_th.update(sp_th.data.item())

    return losses.avg, l_mse.avg, l_bi.avg, Sp_0.avg, Sp_th.avg


def train_D_CL(args, writers, trainloader, testloader):
    # Initialize DYAN Dictionary
    Drr, Dtheta = get_Drr_Dtheta(args.N)
    # Network
    net = DYAN_B(args, Drr, Dtheta).cuda(args.gpu_id)
    mseLoss=torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    L1Loss=torch.nn.SmoothL1Loss()
    # Training assistants
    ## Freeze Linear Layer for classification
    for name, p in net.named_parameters():
        if "cls" in name: p.requires_grad = False
    # for p in net.CoefNet.cls[0].parameters():  p.requires_grad = False
    ## 04/17, training strategy from the original paper
    if args.wiF:
        optimizer = torch.optim.SGD([{'params': filter(lambda p:p.requires_grad, net.CoefNet.parameters())},
                                    {'params': net.proj.parameters(), 'lr': args.lr_2}],
                                    lr=args.lr, weight_decay=0.001, momentum=0.9)
    else:
        optimizer = torch.optim.SGD([{'params': net.sparseCoding.parameters()},
                                    {'params': filter(lambda p:p.requires_grad, net.CoefNet.parameters())},
                                    {'params': net.proj.parameters(), 'lr': args.lr_2}],
                                    lr=args.lr, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(step) for step in args.ms.split(',')], gamma=0.1)
    # LOG
    args.dir_save = os.path.join(args.work_dir,
                                 args.name_exp,
                                 datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.dir_save, exist_ok=True)
    f_log = open(os.path.join(args.dir_save,'log.txt'),'w')
    for arg in vars(args):  log(f"{arg}: {str(getattr(args, arg))}",f_log)
    # Save states for model before training
    if args.save_m:
        torch.save({'state_dict': net.state_dict()},
                    os.path.join(args.dir_save, '0.pth'))
    loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, testloader, net)
    log(f'Test ({0})|acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
    writers[1].add_scalar("Accuracy", acc,      0)
    writers[1].add_scalar("Loss",     loss,     0)
    writers[1].add_scalar("Loss_CLS", loss_cls, 0)
    writers[1].add_scalar("Loss_BI",  loss_bi,  0)
    writers[1].add_scalar("Loss_MSE", loss_mse, 0)
    
    # START Training
    for epoch in range(1, args.ep_D+1):
        net.train()
        losses, l_mse, l_cls, l_bi, l_cl = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        Sp_0, Sp_th = AverageMeter(), AverageMeter()
        accs = AverageMeter()
        for i, (data,label,_) in enumerate(trainloader):
            # Affine Augment
            data_1 = affine_aug(data)
            data_2 = affine_aug(data)
            # batch_size, num_clips, dim_joints, T, num_joints, num_subj
            skeletons_1 = data_1.cuda(args.gpu_id)
            skeletons_2 = data_2.cuda(args.gpu_id)
            gt_label = label.cuda(args.gpu_id)
            # => batch_size, num_clips, num_subj, T, num_joints, dim_joints
            skeletons_1 = skeletons_1.transpose(2,5)
            skeletons_2 = skeletons_2.transpose(2,5)
            bz,n_clips,S,T,J,D = skeletons_1.shape 
            skeletons_1 = skeletons_1.reshape(-1,S,T,J,D)
            skeletons_2 = skeletons_2.reshape(-1,S,T,J,D)
            N,S,T,_,_ = skeletons_1.shape
            skeletons_1 = skeletons_1.reshape(N,S,T,-1)
            skeletons_2 = skeletons_2.reshape(N,S,T,-1)
            if args.wiCY:
                Y1 = torch.cat((skeletons_1[:,0,:,:],skeletons_1[:,1,:,:]),2).cuda(args.gpu_id)
                Y2 = torch.cat((skeletons_2[:,0,:,:],skeletons_2[:,1,:,:]),2).cuda(args.gpu_id)
            else:
                Y1 = skeletons_1.cuda(args.gpu_id)
                Y2 = skeletons_2.cuda(args.gpu_id)
                # label, R, B, Z, C
            out_label1, R1, B1, Z1, C1 = net(Y1, T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
            _, _, _, Z2, _             = net(Y2, T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
            # Contrastive Loss
            Z = torch.cat([Z1,Z2], dim=0) # bz -> bz *2 
            simL_matrix = torch.matmul(Z, Z.T)

            L = torch.cat([torch.arange(bz) for i in range(2)], dim=0)
            L = (L.unsqueeze(0) == L.unsqueeze(1)).float().cuda(args.gpu_id)

            mask = torch.eye(L.shape[0], dtype=torch.bool).cuda(args.gpu_id)
            L = L[~mask].view(L.shape[0],-1)
            simL_matrix = simL_matrix[~mask].view(simL_matrix.shape[0], -1)
            positives = simL_matrix[L.bool()].view(L.shape[0], -1)
            negatives = simL_matrix[~L.bool()].view(simL_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            L = torch.zeros(logits.shape[0], dtype=torch.long).cuda(args.gpu_id)
            temper = 0.07 #default
            logits = logits/temper

            loss_cl = criterion(logits, L)

            out_label = torch.mean(out_label1.view(-1, n_clips, args.num_class), 1)
            loss_cls = criterion(out_label, gt_label)
            loss_mse = mseLoss(R1, Y1)
            loss_bi = L1Loss(torch.zeros_like(B1).to(B1),B1)
            
            loss = args.lam1*loss_cls + args.lam2*loss_mse + args.Alpha*loss_bi
            # C.register_hook(lambda grad: print(grad))
            # B.register_hook(lambda grad: print(grad))
            optimizer.zero_grad()
            loss_cl.backward()
            optimizer.step()

            l_cl.update(loss_cl.data.item())
            accs.update(accuracy(out_label, gt_label)[0].data.item())
            losses.update(loss.data.item())
            l_cls.update(loss_cls.data.item())
            l_mse.update(loss_mse.data.item())
            l_bi.update(loss_bi.data.item())

            # Sparsity Indicator
            sp_0, sp_th = sparsity(C1)
            Sp_0.update(sp_0.data.item())
            Sp_th.update(sp_th.data.item())
            if i%(trainloader.__len__()//10)==0:
                log(f'Train({epoch}|{i+1}): |l_cl|{l_cl.value}|loss|{losses.value}|l_cls|{l_cls.value}'+ 
                    f'|l_bi|{l_bi.value}|l_mse|{l_mse.value}|acc|{accs.value}%', f_log)
        scheduler.step()
        log(f'Train({epoch})|l_cl|{l_cl.avg}|loss |{losses.avg}|l_mse|{l_mse.avg}|l_bi|{l_bi.avg}|Sp_0|{Sp_0.avg}|Sp_th|{Sp_th.avg}', f_log)
        writers[0].add_scalar("Loss_CL", l_cl.avg, epoch)
        writers[0].add_scalar("Loss", losses.avg, epoch)
        writers[0].add_scalar("Loss_CLS", l_cls.avg, epoch)
        writers[0].add_scalar("Loss_BI", l_bi.avg, epoch)
        writers[0].add_scalar("Loss_MSE", l_mse.avg, epoch)
        writers[0].add_scalar("Accuracy", accs.avg, epoch)
        
        loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, testloader, net) 
        log(f'Test ({epoch})|acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
        writers[1].add_scalar("Accuracy", acc, epoch)
        writers[1].add_scalar("Loss", loss, epoch)
        writers[1].add_scalar("Loss_CLS", loss_cls, epoch)
        writers[1].add_scalar("Loss_BI",  loss_bi,  epoch)
        writers[1].add_scalar("Loss_MSE", loss_mse, epoch)

        if args.save_m:
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(), 'args':args},
                        os.path.join(args.dir_save, str(epoch) + '.pth'))
    log(f"END Timstamp:{datetime.datetime.now().strftime('%Y:%m:%d %H:%M')}",f_log)
    f_log.close()


def train_cls(args, writers, trainloader, testloader):
    # Initialize DYAN Dictionary
    Drr, Dtheta = get_Drr_Dtheta(args.N)
    # Select Network
    if args.wiG:
        # TODO: should be Group DYAN here
        net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
    else:
        net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
    mseLoss=torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    L1Loss=torch.nn.SmoothL1Loss()
    # Training assistants
    if args.wiF:
        for p in net.sparseCoding.parameters():  p.requires_grad = False
        if args.wiCL:
            optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, net.parameters()),
                                        lr=args.lr_2, weight_decay=0.001, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, net.parameters()),
                                        lr=args.lr, weight_decay=0.001, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.lr, weight_decay=0.001, momentum=0.9) # 1e-4
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
                                         milestones=[int(step) for step in args.ms.split(',')]) # 30,50
    # LOG
    args.dir_save = os.path.join(args.work_dir, args.name_exp,datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(args.dir_save, exist_ok=True)
    f_log = open(os.path.join(args.dir_save,'log.txt'),'w')
    for arg in vars(args):  log(f"{arg}: {str(getattr(args, arg))}",f_log)

    # START Training
    
    for epoch in range(1, args.ep+1):
        losses, l_mse, l_cls, l_bi = AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter()
        accs = AverageMeter()

        net.train()
        for i, (data,label,_) in enumerate(trainloader):
            # batch_size, num_clips, dim_joints, T, num_joints, num_subj
            if args.wiAff:  data = affine_aug(data, args.trs[0], args.trs[1], args.trs[2])
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
                log(f'Train({epoch}|ite{i+1}): |loss|{losses.value}|l_cls|{l_cls.value}'+ 
                    f'|l_bi|{l_bi.value}|l_mse|{l_mse.value}|acc|{accs.value}%', f_log)
                
        scheduler.step()
        log(f'Train({epoch})|acc|{accs.avg}%|loss|{losses.avg}|l_cls|{l_cls.avg}|l_bi|{l_bi.avg}|l_mse|{l_mse.avg}',f_log)
        writers[0].add_scalar("Loss", losses.avg, epoch)
        writers[0].add_scalar("Loss_CLS", l_cls.avg, epoch)
        writers[0].add_scalar("Loss_BI", l_bi.avg, epoch)
        writers[0].add_scalar("Loss_MSE", l_mse.avg, epoch)
        writers[0].add_scalar("Accuracy", accs.avg, epoch)
        if epoch % args.ep_save == 0:
            if args.save_m:
                torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict(),'args':args},
                            os.path.join(args.dir_save, str(epoch) + '.pth'))
            loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, testloader, net)
            log(f'Test ({epoch})|acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
            writers[1].add_scalar("Accuracy", acc, epoch)
            writers[1].add_scalar("Loss", loss, epoch)
            writers[1].add_scalar("Loss_CLS", loss_cls, epoch)
            writers[1].add_scalar("Loss_BI",  loss_bi,  epoch)
            writers[1].add_scalar("Loss_MSE", loss_mse, epoch)
    log(f"END Timstamp: {datetime.datetime.now().strftime('%Y/%m/%d %H:%M')}",f_log)
    f_log.close()


def test_cls(args, dataloader, net,
            mseLoss=torch.nn.MSELoss(),
            criterion=torch.nn.CrossEntropyLoss(),
            L1Loss=torch.nn.SmoothL1Loss()):
    losses, l_mse, l_cls, l_bi = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        net.eval()
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
            if args.wiCL and args.mode=='D':
                out_label, R, B, _, _ = net(Y,T)
            else:
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