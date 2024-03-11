from dataset.NTU_Inter import *
from models.networks import *

def test(dataloader, net, gpu_id, 
            sampling, mode,
            withMask=False, gumbel_thresh=False):
    acc = 0
    losses, l_mse, l_cls, l_bi = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    with torch.no_grad():
        for i, (data,label,index) in enumerate(dataloader):
            skeletons = data.cuda(gpu_id)
            gt_label = label.cuda(gpu_id)
            if mode == 'dy':
                # skeletons: batch_size, num_clips, dim_joints, T, num_joints, num_subj
                #          ->batch_size, num_subj, num_clips, T, num_joints, dim_joints
                skeletons = skeletons.transpose(0,5,1,3,4,2)
                _,_,_,T,J,D = skeletons.shape
                skeletons = skeletons.reshape(-1,T,J,D)
                N,T,_,_ = skeletons.shape
                skeletons = skeletons.reshape(N,T,-1)
                loss_mse = torch.zeros(1).cuda(gpu_id)
                _,_,R = net(skeletons,T)
                loss_mse += mseLoss(R, skeletons)
                loss = loss_mse
            l_mse.update(loss_mse.data.item())
            losses.update(loss.data.item())
        
    return losses.avg, l_mse.avg, l_cls.avg, l_bi.avg, acc

# if __name__ == "__main__":
#     gpu_id = 2
#     bz = 8
#     num_workers = 4
#     'initialized params'
#     N = 80 * 2
#     P, Pall = gridRing(N)
#     Drr = abs(P)
#     Drr = torch.from_numpy(Drr).float()
#     Dtheta = np.angle(P)
#     Dtheta = torch.from_numpy(Dtheta).float()

#     mode = 'dy+bi+cl'
#     T = 36
#     dataset = 'NUCLA'
#     sampling = 'Multi'
#     withMask = False
#     gumbel_thresh = 0.505  #0.505
#     setup = 'setup1'
#     path_list = './data/CV/' + setup + '/'
#     # testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling=sampling, phase='test', cam='2,1', T=T,
#     #                           maskType='score', setup=setup)
#     # testloader = DataLoader(testSet, batch_size=bz, shuffle=True, num_workers=num_workers)

#     if mode == 'dy+bi+cl':

#         # net = Fullclassification(num_class=10, Npole=(N + 1), Drr=Drr, Dtheta=Dtheta, dim=2, dataType='2D',
#         #                          Inference=True,
#         #                          gpu_id=gpu_id, fistaLam=0.1, group=False, group_reg=0.01, useCL=False).cuda(gpu_id)
#         net = contrastiveNet(dim_embed=128, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id, dim=2,
#                                dataType='2D', fistaLam=0.1, fineTune=True, useCL=False).cuda(gpu_id)
#     else:
#         kinetics_pretrain = './pretrained/i3d_kinetics.pth'
#         net = twoStreamClassification(num_class=10, Npole=(N + 1), num_binary=(N + 1), Drr=Drr, Dtheta=Dtheta, dim=2,
#                                   gpu_id=gpu_id, inference=True, fistaLam=0.1, dataType='2D',
#                                   kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)


#     ckpt = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/' + sampling + '/' + mode + '/T36_contrastive_fineTune_all/' + '60.pth'
#     # ckpt = './pretrained/N-UCLA/' + setup + '/' + sampling + '/pretrainedRHdyan_BI_v2.pth'
#     stateDict = torch.load(ckpt, map_location="cuda:" + str(gpu_id))['state_dict']
#     # net.load_state_dict(stateDict)
#     net = load_pretrainedModel_endtoEnd(stateDict,net)
#     # pdb.set_trace()
#     Acc = testing(testloader,net, gpu_id, sampling, mode, withMask,gumbel_thresh)

#     print('Acc:%.4f' % Acc)
#     print('done')