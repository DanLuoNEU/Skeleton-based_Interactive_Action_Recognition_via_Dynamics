# Move to repo root to run this file
# Dan
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main(args):
    # Dataset
    # trainSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
    #                     split='train')
    args.bs=1
    testSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                        split='test')
    
    # trainloader = DataLoader(trainSet, batch_size=args.bs, shuffle=False,
    #                          num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testSet, batch_size=args.bs, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # # Log
    # str_net_1 = f"{'wiCY' if args.wiCY else 'woCY'}_{'wiG' if args.wiG  else 'woG'}_{'wiRW' if args.wiRW else 'woRW'}_{'wiCC' if args.wiCC else 'woCC'}_"
    # str_net_2 = f"{'wiF' if args.wiF else 'woF'}_{'wiBI' if args.wiBI else 'woBI'}_{'wiCL' if args.wiCL else 'woCL'}"
    
    
    # args.name_exp = f"{args.dataset}_{args.setup}_{args.mode}{'' if args.wiD!='' else '_woD'}_{str_net_1+str_net_2}"
    # args.name_exp = args.name_exp + f"_T{args.T}_f{args.lam_f:.0e}_d{args.th_d:.0e}_{args.lam1}_{args.lam2}_{args.Alpha}_{args.th_gumbel:.3f}"
    # print(args.name_exp)
    print('START Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M'))
    colors = ['red', 'green']
    if args.mode == 'D':
        # Initialize DYAN Dictionary
        Drr, Dtheta = get_Drr_Dtheta(args.N)
        # Select Network
        net1 = DYANEnc(args, Drr=Drr, Dtheta=Dtheta).cuda(7)
        net2 = DYANEnc(args, Drr=Drr, Dtheta=Dtheta).cuda(7)
        # Directory to save the test log

        with torch.no_grad():
            for i, (data,_,_) in enumerate(testloader):
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
                C1, _, R1, B1 = net1(Y.cuda(7), T) # y: (batch_size, num_clips) x T x (num_joints x (dim_joints) x num_subj)
                C2, _, R2, B2 = net2(Y.cuda(7), T)
                # Sparsity Indicator
                sp_0_1, sp_th_1 = sparsity(C1)
                sp_0_2, sp_th_2 = sparsity(C2)
                # c_t = np.random.random(1000).astype(float)
                # writer.add_histogram('CT', c_t, i)
                fig = plt.hist([C1[0].flatten().cpu().data.abs(),
                                C2[0].flatten().cpu().data.abs()],
                                bins=100, color=colors)
                plt.xlim(xmin=0.05, xmax = 1)
                plt.ylim(ymin=0, ymax = 20)
                plt.savefig('scripts/C_woRW-wiRW.png')
                plt.close()
    elif args.mode == 'cls':
        pass
    
    # print(args.name_exp)
    print('END Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M')) 


if __name__ == '__main__':
    from dataset.NTU_Inter import NTU
    from models.SparseCoding import get_Drr_Dtheta, DYANEnc
    from utils.utils import accuracy, sparsity
    from train import init_seed, log, get_parser
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)

    init_seed(args.seed)
    main(args)