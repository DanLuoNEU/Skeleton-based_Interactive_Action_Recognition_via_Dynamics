import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.NTU_Inter import NTU
from dataset.NTU120_Inter import NTU120
from train import init_seed, get_parser, train_D, train_cls


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
    
    trainloader = DataLoader(trainSet, batch_size=args.bs, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testSet, batch_size=args.bs, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Log
    str_net_1 = f"{'wiCY' if args.wiCY else 'woCY'}_{'wiG' if args.wiG  else 'woG'}_{'wiCC' if args.wiCC else 'woCC'}_"
    str_net_2 = f"{'wiF' if args.wiF else 'woF'}_{'wiBI' if args.wiBI else 'woBI'}_{'wiCL' if args.wiCL else 'woCL'}"
    
    
    args.name_exp = f"{args.dataset}_{args.setup}_{args.mode}{'' if args.wiD!='' else '_woD'}_{str_net_1+str_net_2}_T{args.T}_f{args.lam_f}_{args.lam1}_{args.lam2}"
    print(args.name_exp)
    writers=[SummaryWriter(log_dir=f'runs/{args.name_exp}_train'),
             SummaryWriter(log_dir=f'runs/{args.name_exp}_test')]
    if args.mode == 'D':
        train_D(args, writers, trainloader, testloader)
    elif args.mode == 'cls':
        train_cls(args, writers, trainloader, testloader)
    
    print(args.name_exp)
    print('END Timstamp:',datetime.datetime.now().strftime('%Y:%m:%d %H:%M')) 



if __name__ == '__main__':
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)

    init_seed(args.seed)
    main(args)