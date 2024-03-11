# creatRealDictionary
# fista_new, DYANEnc
# fista_reweighted, DYANrhEnc
# fista_reweighted_mask,fista_reweighted_batch
# fista_reweighted_wc_batch,fista_reweighted_mask_batch
# fista_reweighted_mask_wc_batch, MaskDyanEncoder
import os
import sys
## Imports related to PyTorch
import pdb
import time
from math import sqrt
import numpy as np
import random

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable

from utils import *
from dataset.NTU_Inter import NTU
# from modelZoo.actHeat import imageFeatureExtractor

def creatRealDictionary(T, rr, theta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    for i in range(0,T):
        W1 = torch.mul(torch.pow(rr,i) , torch.cos(i*theta)).cuda(gpu_id)
        W2 = torch.mul(torch.pow(rr,i) , torch.sin(i*theta)).cuda(gpu_id)
        W = torch.cat((Wones,W1,W2),0)
        WVar.append(W.view(1,-1))
    dic = torch.cat((WVar),0)

    return dic


## Generate the grid on poles
def generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax):
    rmin2 = pow(rmin, 2)
    rmax2 = pow(rmax, 2)
    xv = np.arange(-rmax, rmax, delta)
    x, y = np.meshgrid(xv, xv, sparse=False)
    mask = np.logical_and(np.logical_and(x ** 2 + y ** 2 >= rmin2, x ** 2 + y ** 2 <= rmax2),
                          np.logical_and(np.angle(x + 1j * y) >= thetaMin, np.angle(x + 1j * y) <= thetaMax))
    px = x[mask]
    py = y[mask]
    P = px + 1j * py

    return P


def gridRing(N):
    """
    Generate Pho and Theta randomly in range
    """
    epsilon_low = 0.15
    epsilon_high = 0.15
    rmin = (1 - epsilon_low)
    rmax = (1 + epsilon_high)

    thetaMin = 0.001
    thetaMax = np.pi - 0.001
    delta = 0.001
    Npole = int(N/2)
    Pool = generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax)
    M = len(Pool)

    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P, np.conjugate(P)), axis=0)  # mirror once

    return P, Pall


def get_Drr_Dtheta(N):
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    return Drr, Dtheta


def fista_new(D, Y, lambd, maxIter, gpu_id,
              thr_dif=1e-5):
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD, 2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id)
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,linv)
    DtY = torch.mul(DtY,linv)
    Softshrink = nn.Softshrink(lambd)
    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Softshrink((Ay + DtY))

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        # pdb.set_trace()
        if torch.norm((x_old - x_new), 2)/x_old.shape[1] < thr_dif:
            x_old = x_new
            # print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old


class DYANEnc(nn.Module):
    def __init__(self, Drr=torch.zeros(0), Dtheta=torch.zeros(0),
                 lam=0.1, gpu_id=0):
        """
        Drr:    rho in dictionary
        Dtheta: theta in dictionary
        inter:  flag for interaction action recognition
        lambda: lambda for FISTA
        gpu_id: GPU ID for CUDA
        """
        super(DYANEnc, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.lam = lam
        self.gpu_id = gpu_id
    
    def forward(self,x, T):
        D = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        C = fista_new(D, x, self.lam, 100, self.gpu_id)
        R = torch.matmul(D, C)
        return C, D, R

def fista_reweighted(D, Y, lambd, w, maxIter):
    '''D: T x 161, Y: N x T x 50, w: N x 161 x 50'''
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0, 2, 1), D)
        DtY = torch.matmul(D.permute(0, 2, 1), Y)
    # eig, v = torch.eig(DtD, eigenvectors=True)
    # eig, v = torch.linalg.eig(DtD)
    # L = torch.max(eig)
    L = torch.norm(DtD, 2)

    # eigs = torch.abs(torch.linalg.eigvals(DtD))
    # L = torch.max(eigs, dim=1).values

    Linv = 1/L
    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).to(D.device) - torch.mul(DtD,Linv)
    t_old = 1

    const_xminus = torch.mul(DtY, Linv) - weightedLambd.to(D.device)
    const_xplus = torch.mul(DtY, Linv) + weightedLambd.to(D.device)

    iter = 0
    while iter < maxIter:
        iter +=1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old-1)/t_new
        y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), 2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old

class DYANrhEnc(nn.Module):
    def __init__(self, Drr=torch.zeros(0),
                       Dtheta=torch.zeros(0),
                       inter=True,
                       lam=0.1, gpu_id=0,
                       max_ite=2):
        """
        Drr:    rho in dictionary
        Dtheta: theta in dictionary
        inter:  flag for interaction action recognition
        lambda: lambda for FISTA
        gpu_id: GPU ID for CUDA
        max_iter: max iteration muber for RH DYAN
        """
        super(DYANrhEnc, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T # 36
        self.inter=inter
        self.lam = lam
        self.gpu_id = gpu_id
        self.max_ite = max_ite

    def forward(self, x, T):
        # with RH
        D = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        # print('rr:', self.rr, 'theta:', self.theta)
        # sparseCode = fista_new(dic,x,self.lam, 100,self.gpu_id)

        i = 0
        w_init = torch.ones(x.shape[0], D.shape[1], x.shape[2])
        while i < self.max_ite:
            temp = fista_reweighted(D, x, self.lam, w_init, 100)
            # vector:
            w = 1 / (torch.abs(temp) + 1e-2)
            w_init = (w/torch.norm(w)) * D.shape[1]

            # matrix:
            # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
            # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
            # pdb.set_trace()

            final = temp
            del temp
            i += 1

        sparseCode = final
        sparseCode = sparseCode.cuda(self.gpu_id)

        reconst = torch.matmul(D, sparseCode)
        return sparseCode, D, reconst


def fista_reweighted_mask(D, Y, lambd, w, maxIter):
    'D: T x 50, Y: bz x T x 1, w:bz x 161 x 1 '
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0,2,1), D)
        DtY = torch.matmul(D.permute(0,2,1), Y)

    eigs = torch.abs(torch.linalg.eigvals(DtD))
    L = torch.max(eigs, dim=1).values


    # print('max eigen:', L, 'min eigen:', torch.min(eigs), 'max D:', torch.max(D))
    # pdb.set_trace()
    # L = torch.norm(DtD, 2)
    Linv = (1 / L).unsqueeze(1).unsqueeze(2)
    # print(w)
    # pdb.set_trace()
    # w = w.permute(2, 1, 0)
    weightedLambd = (w * lambd) * Linv

    x_old = torch.zeros(DtD.shape[0], DtY.shape[1], DtY.shape[2]).to(D.device)

    y_old = x_old
    I = torch.eye(DtD.shape[1]).to(D.device).unsqueeze(0).repeat(DtD.shape[0], 1, 1)
    A = I - DtD * (Linv)
    t_old = 1

    const_xminus = DtY * Linv - weightedLambd
    const_xplus = DtY * Linv + weightedLambd

    iter = 0

    while iter < maxIter:

        iter += 1
        Ay = torch.bmm(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old - 1) / t_new
        # print(t_new)
        y_new = x_new + torch.mul(tt, (x_new - x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

        del t_new, x_new, y_new
        # torch.cuda.empty_cache()
    return x_old

def fista_reweighted_batch(D, Y, lambd, w, maxIter, gpu_id):
    # Y = Y.squeeze(0)
    # Y = Y.permute(1, 0)
    # Y = Y.unsqueeze(2)
    if len(Y.shape) == 3:
        Y = Y.permute(2, 1, 0)
        Dt = D.permute(0, 2, 1)
        w = w.permute(2, 1, 0)
    else:
        Y = Y.permute(0, 2, 1, 3)
        Dt = D.permute(0, 1, 3, 2)
        w = w.permute(0, 3, 2, 1)

    DtD = torch.matmul(Dt, D)
    DtY = torch.matmul(Dt, Y)

    eigs = torch.abs(torch.linalg.eigvals(DtD))
    L = torch.max(eigs, dim=-1).values

    # print('max eigen:', L, 'min eigen:', torch.min(eigs), 'max D:', torch.max(D))
    # pdb.set_trace()
    Linv = 1 / L
    # print(w)
    # pdb.set_trace()

    weightedLambd = (w * lambd) * Linv.unsqueeze(-1).unsqueeze(-1)

    if len(Y.shape) == 3:
        x_old = torch.zeros(DtD.shape[0], DtY.shape[1], DtY.shape[2]).to(D.device)

        I = torch.eye(DtD.shape[1]).to(D.device).unsqueeze(0).repeat(DtD.shape[0], 1, 1)

    else:
        x_old = torch.zeros(DtD.shape[0], DtD.shape[1], DtY.shape[2], DtY.shape[3]).to(D.device)
        I = torch.eye(DtD.shape[2]).to(D.device).unsqueeze(0).unsqueeze(0).repeat(DtD.shape[0],DtD.shape[1],1,1)

    y_old = x_old


    A = I - DtD * (Linv.unsqueeze(-1).unsqueeze(-1))

    t_old = 1

    const_xminus = DtY * (Linv.unsqueeze(-1).unsqueeze(-1)) - weightedLambd
    const_xplus = DtY * (Linv.unsqueeze(-1).unsqueeze(-1)) + weightedLambd

    iter = 0

    while iter < maxIter:
        iter += 1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old - 1) / t_new
        y_new = x_new + torch.mul(tt, (x_new - x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old


'''
This function is the implementation of the reweighted fista with weight of coefficient as an input.
'''
def fista_reweighted_wc_batch(D, Y, lambd, Wc, maxIter, gpu_id):
    i = 0
    w_init = Wc
    # w_init = torch.ones(1, dic.shape[1], x.shape[2]).cuda(self.gpu_id)
    while i < 2:
        temp = fista_reweighted_batch(D, Y, lambd, w_init, maxIter, gpu_id)
        'for vector:'
        w = Wc / (torch.abs(temp) + 1e-2)
        w_init = (w / torch.norm(w)) * D.shape[1]

        'for matrix:'
        # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
        # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
        # pdb.set_trace()
        final = temp
        del temp
        i += 1
    sparseCode = final

    reconst = torch.matmul(D, sparseCode)
    return sparseCode, D, reconst


def fista_reweighted_mask_batch(D, Y, lambd, w, M, maxIter, gpu_id):
    YM = M * Y
    if len(Y.shape) == 3:
        DM = D * M.permute(2, 1, 0)
    else:
        DM = D * M.permute(0, 2, 1, 3)
    return fista_reweighted_batch(DM, YM, lambd, w, maxIter, gpu_id)


def fista_reweighted_mask_wc_batch(D, Y, lambd, Wc, M, maxIter, gpu_id):
    i = 0
    w_init = Wc
    # w_init = torch.ones(1, dic.shape[1], x.shape[2]).cuda(self.gpu_id)
    while i < 2:
        temp = fista_reweighted_mask_batch(D, Y, lambd, w_init, M, maxIter, gpu_id)
        'for vector:'
        if len(Y.shape) == 3:
            w = Wc / ((torch.abs(temp) + 1e-2).permute(2, 1, 0))
            w_init = (w / torch.norm(w)) * D.shape[-1]
        else:
            w = Wc/((torch.abs(temp) + 1e-2).permute(0,3,2,1))
            w_init = (w/torch.norm(w)) * D.shape[-1] * D.shape[-2]

        'for matrix:'
        # w = torch.pinverse((temp + 1e-2*torch.ones(temp.shape).cuda(self.gpu_id)))
        # w_init = (w/torch.norm(w,'fro')) * (temp.shape[1]*temp.shape[-1])
        # pdb.set_trace()
        final = temp
        del temp
        i += 1
    sparseCode = final

    reconst = torch.matmul(D, sparseCode)
    return sparseCode, D, reconst


class MaskDyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, lam, gpu_id):
        super(MaskDyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T
        self.lam = lam
        self.gpu_id = gpu_id

    def forward(self, x, T, M):
        dic = creatRealDictionary(T, self.rr, self.theta, self.gpu_id)
        if len(x.shape) == 3:
            dic = dic.repeat(x.shape[2], 1, 1)
            Wc = torch.ones(1, dic.shape[2], x.shape[-2]).cuda(self.gpu_id)
        else:
            dic = dic.repeat(x.shape[0], x.shape[2], 1, 1)
            Wc = torch.ones(x.shape[0], x.shape[-1], dic.shape[-1], x.shape[-2]).cuda(self.gpu_id)
        # print('rr:', self.rr, 'theta:', self.theta)


        sparseCode, dic, reconst = fista_reweighted_mask_wc_batch(dic, x, self.lam, Wc, M, 100, self.gpu_id)
        if len(x.shape) == 3:
            sparseCode = sparseCode.permute(2,1,0)
            dic = torch.mean(dic,0)
        else:
            sparseCode = sparseCode.squeeze(-1).permute(0,2,1)
            dic = torch.mean(torch.mean(dic,0),0)


        return sparseCode, dic, reconst


if __name__ == '__main__':
    modelRoot = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/'
    saveModel = modelRoot + 'Single/regularDYAN/'
    if not os.path.exists(saveModel):
        os.makedirs(saveModel)
    lam = 0.1
    N = 80 * 2
    Epoch = 100
    gpu_id = 4
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    withMask = False
    net = DYANEnc(Drr, Dtheta, lam, gpu_id)
    net.cuda(gpu_id)
    setup = 'setup1'  # v1,v2 train, v3 test;
    path_list = '../data/CV/' + setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'

    # trainSet = NTUdataset()(root_list=path_list, dataType='2D', sampling='Single', phase='train', cam='2,1', T=36,
    #                            maskType='score',
    #                            setup=setup)
    # # #

    # trainloader = DataLoader(trainSet, batch_size=24, shuffle=True, num_workers=8)

    # testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling='Single', phase='test', cam='2,1', T=36,
    #                           maskType='score', setup=setup)
    # testloader = DataLoader(testSet, batch_size=16, shuffle=True, num_workers=8)

    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001,
                                momentum=0.9)

    net.train()

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    mseLoss = torch.nn.MSELoss()

    # Loss = []
    for epoch in range(0, Epoch+1):
        print('training epoch:', epoch)
        lossVal = []
        start_time = time.time()
        # for i, sample in enumerate(trainloader):
        #     # print('sample:', i)
        #     optimizer.zero_grad()

        #     skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

        #     t = skeletons.shape[1]
        #     input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

        #     sparseCode,_,output_skeletons = net.forward2(input_skeletons, t)

        #     loss = mseLoss(output_skeletons, input_skeletons)
        #     loss.backward()
        #     # print('rr.grad:', net.rr.grad, 'theta.grad:', net.theta.grad)
        #     optimizer.step()

        #     lossVal.append(loss.data.item())
        # end_time = time.time()
        # print('epoch:', epoch, 'loss:', np.mean(np.asarray(lossVal)), 'time(h):', (end_time - start_time) / 3600)

        # if epoch % 20 == 0:
        #     torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
        #                 'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')
        #     with torch.no_grad():
        #         ERROR = torch.zeros(testSet.__len__(), 1)

        #         for i, sample in enumerate(testloader):
        #             skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

        #             t = skeletons.shape[1]
        #             input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

        #             _,_,output_skeletons = net.forward2(input_skeletons, t)

        #             error = torch.norm(output_skeletons - input_skeletons).cpu()
        #             ERROR[i] = error

        #         print('epoch:', epoch, 'error:', torch.mean(ERROR))

        # scheduler.step()
    print('done')