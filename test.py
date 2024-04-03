import os
import datetime

import torch
from torch.utils.data import DataLoader

from dataset.NTU_Inter import NTU
from dataset.NTU120_Inter import NTU120
from models.SparseCoding import get_Drr_Dtheta, DYANEnc
from utils.utils import accuracy, sparsity
from train import init_seed, log, get_parser, test_D, test_cls

def main(args):
    # Dataset
    if args.dataset=='NTU':
        trainSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='train')
        testSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='test')
    elif args.dataset=='NTU120':
        trainSet = NTU120(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='train')
        testSet = NTU120(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='test')
    
    trainloader = DataLoader(trainSet, batch_size=args.bs, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testSet, batch_size=args.bs, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # # Log
    # str_net_1 = f"{'wiCY' if args.wiCY else 'woCY'}_{'wiG' if args.wiG  else 'woG'}_{'wiRW' if args.wiRW else 'woRW'}_{'wiCC' if args.wiCC else 'woCC'}_"
    # str_net_2 = f"{'wiF' if args.wiF else 'woF'}_{'wiBI' if args.wiBI else 'woBI'}_{'wiCL' if args.wiCL else 'woCL'}"
    
    
    # args.name_exp = f"{args.dataset}_{args.setup}_{args.mode}{'' if args.wiD!='' else '_woD'}_{str_net_1+str_net_2}"
    # args.name_exp = args.name_exp + f"_T{args.T}_f{args.lam_f:.0e}_d{args.th_d:.0e}_{args.lam1}_{args.lam2}_{args.Alpha}_{args.th_gumbel:.3f}"
    # print(args.name_exp)
    print('START Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M'))
    if args.mode == 'D':
        # Initialize DYAN Dictionary
        Drr, Dtheta = get_Drr_Dtheta(args.N)
        # Select Network
        if args.wiG:
            # TODO: should be Group DYAN here
            net = DYANEnc(Drr=Drr, Dtheta=Dtheta, lam=args.lam_f, thr_dif=args.th_d,
                          wiRW=args.wiRW, gpu_id=args.gpu_id).cuda(args.gpu_id)
        else:
            net = DYANEnc(Drr=Drr, Dtheta=Dtheta, lam=args.lam_f, thr_dif=args.th_d,
                          wiRW=args.wiRW, gpu_id=args.gpu_id).cuda(args.gpu_id)
        # Directory to save the test log
        
        f_log = open(os.path.join(os.path.dirname(os.path.abspath(args.wiD)),'test.txt'),'w')
        
        update_dict = net.state_dict()
        model_pret = torch.load(args.wiD, map_location=args.map_loc)
        state_dict = model_pret['state_dict']
        pret_dict = {k: v for k, v in state_dict.items() if k in update_dict}
        update_dict.update(pret_dict)
        net.load_state_dict(update_dict)
        # Load pretrained model
        print(f"Pretrained Model Loaded: {args.wiD}")

        loss, loss_mse, sp_0, sp_th = test_D(args, trainloader, net) 
        log(f'Train| loss |{loss}| l_mse |{loss_mse}| Sp_0 |{sp_0}| Sp_th(<0.05) |{sp_th}', f_log)
        loss, loss_mse, sp_0, sp_th = test_D(args, testloader, net) 
        log(f'Test| loss |{loss}| l_mse |{loss_mse}| Sp_0 |{sp_0}| Sp_th(<0.05) |{sp_th}',f_log)
    elif args.mode == 'cls':
        pass
    
    # print(args.name_exp)
    print('END Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M')) 


if __name__ == '__main__':
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)

    init_seed(args.seed)
    main(args)