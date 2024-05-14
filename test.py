import os
import datetime

import torch
from torch.utils.data import DataLoader

from dataset.NTU_Inter import NTU
from dataset.NTU120_Inter import NTU120
from models.SparseCoding import get_Drr_Dtheta, DYANEnc
from models.networks import DYAN_B
from utils.utils import accuracy, sparsity
from train import init_seed, log, get_parser, test_D, test_cls

def main(args):
    # Dataset
    if args.dataset=='NTU':
        trainSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='train', setup=args.setup,
                            limb=args.wiL)
        testSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='test', setup=args.setup,
                            limb=args.wiL)
    elif args.dataset=='NTU120':
        trainSet = NTU120(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='train', setup=args.setup,
                            limb=args.wiL)
        testSet = NTU120(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                            split='test', setup=args.setup,
                            limb=args.wiL)
    
    trainloader = DataLoader(trainSet, batch_size=args.bs, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testSet, batch_size=args.bs, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # print(args.name_exp)
    print('START Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M'))
    if args.mode == 'D':
        # Initialize DYAN Dictionary
        Drr, Dtheta = get_Drr_Dtheta(args.N)
        # Select Network
        if args.wiG:
            # TODO: should be Group DYAN here
            net = DYANEnc(args,Drr,Dtheta).cuda(args.gpu_id)
        else:
            net = DYANEnc(args,Drr,Dtheta).cuda(args.gpu_id)
        # Directory to save the test log
        net.eval()
        f_log = open(os.path.join(os.path.dirname(os.path.abspath(args.wiD)),'test.txt'),'w')
        f_log.writelines([f"{arg}: {getattr(args, arg) or ''}\n" for arg in vars(args)])
        
        update_dict = net.state_dict()
        model_pret = torch.load(args.wiD, map_location=args.map_loc)
        state_dict = model_pret['state_dict']
        pret_dict = {k: v for k, v in state_dict.items() if k in update_dict}
        update_dict.update(pret_dict)
        net.load_state_dict(update_dict)
        # Load pretrained model
        print(f"Pretrained Model Loaded: {args.wiD}")

        loss, loss_mse, loss_bi, sp_0, sp_th = test_D(args, trainloader, net) 
        log(f'Train| loss |{loss}| l_mse |{loss_mse}| Sp_0 |{sp_0}| Sp_th(<0.05) |{sp_th}', f_log)
        loss, loss_mse, loss_bi, sp_0, sp_th = test_D(args, testloader, net) 
        log(f'Test| loss |{loss}| l_mse |{loss_mse}| Sp_0 |{sp_0}| Sp_th(<0.05) |{sp_th}',f_log)
    elif args.mode == 'cls':
        acc_max = 0
        # Initialize DYAN Dictionary
        Drr, Dtheta = get_Drr_Dtheta(args.N)
        if '.pth' in args.pret:
            # Select Network
            if args.wiG:
                # TODO: should be Group DYAN here
                net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
            else:
                net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
            # if not args.wiCL:
            #     # Loading pretrained model
            #     print(f"Loading Pretrained Model: {args.pret}...")
            #     update_dict = net.state_dict()
            #     state_dict = torch.load(args.pret, map_location=args.map_loc)['state_dict']
            #     pret_dict = {k: v for k, v in state_dict.items() if k in update_dict}
            #     update_dict.update(pret_dict)
            #     net.load_state_dict(update_dict)

            net.eval()
            i_ep = args.pret.split('/')[-1].split('.')[0]
            f_log = open(os.path.join(os.path.dirname(os.path.abspath(args.pret)),f"test_{i_ep}.txt"),'w')
            for arg in vars(args):  log(f"{arg}: {str(getattr(args, arg))}",f_log)

            # loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, trainloader, net)
            # log(f'Train|acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
            loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, testloader, net)
            log(f'Test |acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
            if acc > acc_max: acc_max = acc
        else:
            dir_pret = args.pret
            list_nets = [int(name_file.split('.')[0]) for name_file in os.listdir(dir_pret) if '.pth' in name_file]
            list_nets.sort()

            f_log = open(os.path.join(dir_pret, f"test_all.txt"),'w')
            for arg in vars(args):  log(f"{arg}: {str(getattr(args, arg))}",f_log)
            for ep_save in list_nets:
                args.pret = os.path.join(dir_pret,f"{ep_save}.pth")
                # Select Network
                if args.wiG:
                    # TODO: should be Group DYAN here
                    net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
                else:
                    net = DYAN_B(args, Drr=Drr, Dtheta=Dtheta).cuda(args.gpu_id)
                # if not args.wiCL:
                #     # Loading pretrained model
                #     print(f"Loading Pretrained Model: {args.pret}...")
                #     update_dict = net.state_dict()
                #     state_dict = torch.load(args.pret, map_location=args.map_loc)['state_dict']
                #     pret_dict = {k: v for k, v in state_dict.items() if k in update_dict}
                #     update_dict.update(pret_dict)
                #     net.load_state_dict(update_dict)

                net.eval()
                log(f'Epoch {ep_save}', f_log)
                # loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, trainloader, net)
                # log(f'Train|acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
                loss, loss_mse, loss_cls, loss_bi, acc = test_cls(args, testloader, net)
                log(f'Test |acc|{acc}%|loss|{loss}|l_cls|{loss_cls}|l_bi|{loss_bi}|l_mse|{loss_mse}',f_log)
                if acc > acc_max: acc_max = acc
        print(f'ACC max: {acc_max:.04f}')

    
    # print(args.name_exp)
    print('END Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M')) 


if __name__ == '__main__':
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)

    init_seed(args.seed)
    main(args)