import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.NTU_Inter import NTU
from dataset.NTU120_Inter import NTU120
from train import init_seed, get_parser, train_D, train_D_CL, train_cls


def main(args):
    args.map_loc = "cuda:"+str(args.gpu_id)
    
    # Dataset
    args.random_rot = False if (args.wiCL and args.mode=='D') or args.wiAff else True
    if args.wiAff:  args.trs = [float(scale) for scale in args.trs.split(',')]
    if args.dataset=='NTU':
        trainSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                        split='train', setup=args.setup,
                        random_rot=args.random_rot, limb=args.wiL)
        testSet = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                        split='test', setup=args.setup,
                        limb=args.wiL)
    elif args.dataset=='NTU120':
        trainSet = NTU120(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                        split='train', setup=args.setup,
                        random_rot=args.random_rot, limb=args.wiL)
        testSet = NTU120(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN",
                        split='test', setup=args.setup,
                        limb=args.wiL)
    
    trainloader = DataLoader(trainSet, batch_size=args.bs, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testSet, batch_size=args.bs, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    args.num_clips = trainSet.num_clips
    # Log
    str_net_1 = f"{'wiCY' if args.wiCY else 'woCY'}_{'wiG' if args.wiG  else 'woG'}_{'wiRW' if args.wiRW else 'woRW'}_{'wiCC' if args.wiCC else 'woCC'}_"
    str_net_2 = f"{'wiF' if args.wiF else 'woF'}_{'wiBI' if args.wiBI else 'woBI'}_{'wiCL' if args.wiCL else 'woCL'}"
    
    
    args.name_exp = f"{args.dataset}_{args.setup}_{args.mode}_{str_net_1+str_net_2}"
    args.name_exp += f"_T{args.T}_f{args.lam_f:.1e}_d{args.th_d:.1e}_mse{args.lam2}"
    if args.wiBI:        args.name_exp += f"_bi{args.Alpha}_th{args.th_g:.1e}_te{args.te_g:.1e}"
    if args.mode=="cls": args.name_exp += f"_cls{args.lam1}"
    if args.wiD=='':     args.name_exp += '_woD'
    if args.cus_n!='':   args.name_exp += f"_{args.cus_n}"
    print(args.name_exp)
    print('START Timstamp:',datetime.datetime.now().strftime('%Y/%m/%d %H:%M'))
    writers=[SummaryWriter(log_dir=f'runs/{args.name_exp}_train'),
             SummaryWriter(log_dir=f'runs/{args.name_exp}_test')]
    if args.mode == 'D':
        if args.wiCL:
            train_D_CL(args, writers, trainloader, testloader)
        else:
            train_D(args, writers, trainloader, testloader)
    if args.mode == 'cls':
        train_cls(args, writers, trainloader, testloader)
    
    print(args.name_exp)
    print('END Timstamp:',datetime.datetime.now().strftime('%Y/%m/%d %H:%M')) 


if __name__ == '__main__':
    parser = get_parser()
    args=parser.parse_args()

    init_seed(args.seed)
    main(args)