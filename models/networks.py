# GroupNorm,binaryCoding,binarizeSparseCode,classificationGlobal
# classificationWBinarization,ClsSparseCode,Fullclassification
# fusionLayers,twoStreamClassification,MLP,contrastiveNet
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.SparseCoding import gridRing, DYANEnc
# from models.sparseGroupLasso import *
from utils.utils import dim_out
# from models.actRGB import *
from models.gumbel_module import GumbelSigmoid
from scipy.spatial import distance


class CoefNet(nn.Module):
    def __init__(self, args, num_jts=25):
        super(CoefNet, self).__init__()
        self.Npole = args.N+1
        self.njts = num_jts
        self.dim_ske = args.data_dim
        self.wiCC = args.wiCC
        self.useCL = args.wiCL
        self.num_class = args.num_class

        
        self.conv1 = nn.Conv1d(self.Npole, 256, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=256, eps=1e-05, affine=True)
        if self.wiCC:
            w_1 = dim_out(num_jts*self.dim_ske*2,1,1,0)
        else:
            w_1 = dim_out(num_jts*self.dim_ske,1,1,0)
        
        self.conv2 = nn.Conv1d(256, 512, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=512, eps=1e-5, affine=True)
        w_2 = dim_out(w_1,1,1,0)
        
        self.conv3 = nn.Conv1d(512, 1024, 3, stride=self.dim_ske, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=1024, eps=1e-5, affine=True)
        w_3 = dim_out(w_2,self.dim_ske,2,1)
        
        self.pool = nn.AvgPool1d(kernel_size=(self.njts))
        self.conv_mut = nn.Conv1d(1024, 1024, 2, stride=1, padding=0)
        self.bn_mut = nn.BatchNorm1d(num_features=1024, eps=1e-5, affine=True)
        w_p = int(w_3/self.njts)

        # 25,2 -> 23,4 
        self.conv4 = nn.Conv2d(self.Npole + 1024, 1024, (3, 1), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=1024, eps=1e-5, affine=True)
        if self.wiCC:
            h_4, w_4 = dim_out(num_jts,3,1,0), dim_out(args.data_dim*2,1,1,1)
        else:
            h_4, w_4 = dim_out(num_jts,3,1,0), dim_out(args.data_dim,1,1,1)
        # 23,4 -> 21,6
        self.conv5 = nn.Conv2d(1024, 512, (3, 1), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-5, affine=True)
        h_5, w_5 = dim_out(h_4,3,1,0), dim_out(w_4,1,1,1)
        # 21,6 -> 10,2
        self.conv6 = nn.Conv2d(512, 256, (3, 3), stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=256, eps=1e-5, affine=True)
        h_6, w_6 = dim_out(h_5,3,2,0), dim_out(w_5,3,2,0)
        # if not args.NI:
        #     self.fc = nn.Linear(256*h_6*w_6, 1024) #njts = 25
        # else:
        #     self.fc = nn.Linear(256*10*2, 1024) #njts = 25
        self.fc = nn.Linear(256*h_6*w_6, 1024)
        # self.fc = nn.Linear(7168,1024) #njts = 34
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

        # # self.linear = nn.Sequential(nn.Linear(256*10*2,1024),nn.LeakyReLU(),nn.Linear(1024,512),nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU())
        # if self.useCL == False:
        #     self.cls = nn.Sequential(nn.Linear(128,128),nn.LeakyReLU(),nn.Linear(128,self.num_class))
        # else:
        #     self.cls = nn.Sequential(nn.Linear(128, self.num_class))
        self.relu = nn.LeakyReLU()

        # Initialize model weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu' )
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self,x):
        # (num_bs x num_clips) x num_poles x (num_joints x dim_joint x num_subj)
        inp = x

        bz, num_p, N = inp.shape
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x_gl = self.pool(self.relu(self.bn3(self.conv3(x))))

        if self.wiCC:
            x_mut = self.relu(self.bn_mut(self.conv_mut(x_gl)))
            x_gl = torch.cat((x_gl, x_mut), 2)
            N_conv = x_gl.shape[-1]
            x_new = torch.cat((x_gl.repeat(1,1,int(N/N_conv)),inp),1).reshape(bz,1024+self.Npole, self.njts, -1)
        else:
            x_new = torch.cat((x_gl.repeat(1,1,N),inp),1).reshape(bz,1024+self.Npole, self.njts, -1)


        x_out = self.relu(self.bn4(self.conv4(x_new)))
        x_out = self.relu(self.bn5(self.conv5(x_out)))
        x_out = self.relu(self.bn6(self.conv6(x_out)))

        # MLP
        x_out = x_out.view(bz,-1)  #flatten
        x_out = self.relu(self.fc(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out)) #last feature before cls

        # out = self.cls(x_out)

        return x_out


class DYAN_B(nn.Module):
    def __init__(self, args, Drr=torch.zeros(0), Dtheta=torch.zeros(0),
                 num_joints=25):
        super(DYAN_B, self).__init__()
        # self.Npole = args.N+1
        # self.Drr = Drr
        # self.Dtheta = Dtheta
        self.wiCC = args.wiCC
        self.wiP = False
        
        def weight_init(m):
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

        # self.sparseCoding = DYANEnc(args, self.Drr, self.Dtheta)
        self.sparseCoding = DYANEnc(args, Drr, Dtheta)
        self.CoefNet = CoefNet(args, num_jts=num_joints)
        dim_out = self.CoefNet.fc3.out_features
        if self.wiCC:
            self.cls = nn.Sequential(nn.Linear(dim_out, args.num_class))
        else:
            self.cls = nn.Sequential(nn.Linear(dim_out*2, args.num_class))
        self.cls.apply(weight_init)
        
        # For Contrastive Learning
        if args.wiCL:
            if args.mode == 'D':
                # Projection Function for Feature Similarity
                self.wiP = True
                self.num_clips = args.num_clips
                self.proj = nn.Sequential(nn.Linear(dim_out, dim_out),
                                            nn.LeakyReLU())
                self.proj.apply(weight_init)
            # Loading pretrained model
            print(f"Loading Pretrained Model: {args.pret}...")
            update_dict = self.state_dict()
            state_dict = torch.load(args.pret, map_location=args.map_loc)['state_dict']
            pret_dict = {k: v for k, v in state_dict.items() if k in update_dict}
            update_dict.update(pret_dict)
            self.load_state_dict(update_dict)
            

    def forward_wiCC(self, feat_in):
        """"""
        f_cls = self.CoefNet(feat_in)

        return f_cls

    def forward_woCC(self, feat_in):
        """"""
        D = feat_in.shape[-1]
        lastFeat_1 = self.CoefNet(feat_in[:,:,:int(D/2)])
        lastFeat_2 = self.CoefNet(feat_in[:,:,int(D/2):])
        f_cls = torch.cat((lastFeat_1,lastFeat_2),1)

        return f_cls

    def forward(self, x, T):
        """
            x: batch_size x num_clips,  T, dim_joints x num_joints x num_subj
            C: N(161) x D(25x3x2)
            R: T x D
        """
        C, D, R, B = self.sparseCoding(x, T)
        # With Binarization part or not
        if self.sparseCoding.wiBI:  feat_in = B
        else:                       feat_in = C
        # Concatenate Coefficients or not
        if self.wiCC:               f_cls = self.forward_wiCC(feat_in)
        else:                       f_cls = self.forward_woCC(feat_in)
        # If there is a projection for different augmented input
        if self.wiP:
            # Projected features
            Z = self.proj(f_cls)
            # Reshape projecte feature
            bz = x.shape[0]//self.num_clips
            Z = torch.mean(Z.reshape(bz, self.num_clips, Z.shape[-1]), dim=1)
            Z = F.normalize(Z, dim=1)
            # Action Label classification
            label = self.cls(f_cls)

            return label, R, B, Z, C
        else:
            label = self.cls(f_cls)

            return label, R, B


class binaryCoding(nn.Module):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(161, 64, kernel_size=(3,3), padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(64, 32, kernel_size=(3,3), padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 64, kernel_size=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 500),
            # nn.Linear(64*26*8, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, num_binary)
        )

        for m in self.modules():
            # if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
            #     init.xavier_normal(m.weight.data)
            #     m.bias.data.fill_(0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class binarizeSparseCode(nn.Module):
    def __init__(self, num_binary, Drr, Dtheta, gpu_id, Inference, fistaLam):
        super(binarizeSparseCode, self).__init__()
        self.k = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.fistaLam = fistaLam
        # self.sparsecoding = sparseCodingGenerator(self.Drr, self.Dtheta, self.PRE, self.gpu_id)
        # self.binaryCoding = binaryCoding(num_binary=self.k)
        self.sparseCoding = DYANEnc(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
        self.binaryCoding = GumbelSigmoid()

    def forward(self, x, T):
        sparseCode, Dict = self.sparseCoding(x, T)
        # sparseCode = sparseCode.permute(2,1,0).unsqueeze(3)
        # # sparseCode = sparseCode.reshape(1, T, 20, 2)
        # binaryCode = self.binaryCoding(sparseCode)

        # reconstruction = torch.matmul(Dict, sparseCode)
        binaryCode = self.binaryCoding(sparseCode, force_hard=True, temperature=0.1, inference=self.Inference)

        # temp = sparseCode*binaryCode
        return binaryCode, sparseCode, Dict
    

class classificationWBinarization(nn.Module):
    def __init__(self, num_class, Npole, num_binary, dataType, useCL):
        super(classificationWBinarization, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.dataType = dataType
        self.useCL = useCL
        self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
        # self.Classifier = DYANEnc(num_class=self.num_class, Npole=Npole,dataType=self.dataType, useCL=self.useCL)
        self.Classifier = DYANEnc()

    def forward(self, x):
        'x is coefficients'
        inp = x.reshape(x.shape[0], x.shape[1], -1).permute(2,1,0).unsqueeze(-1)
        binaryCode = self.BinaryCoding(inp)
        binaryCode = binaryCode.t().reshape(self.num_binary, x.shape[-2], x.shape[-1]).unsqueeze(0)
        label, _ = self.Classifier(binaryCode)

        return label,binaryCode


# class Fullclassification(nn.Module):
#     def __init__(self, num_class, Npole, Drr, Dtheta,dim, dataType, Inference, gpu_id, fistaLam, group, group_reg,useCL):
#         super(Fullclassification, self).__init__()
#         self.num_class = num_class
#         self.Npole = Npole
#         # self.bi_thresh = 0.505
#         self.num_binary = Npole
#         self.Drr = Drr
#         self.Dtheta = Dtheta
#         self.Inference = Inference
#         self.gpu_id = gpu_id
#         self.dim = dim
#         self.useCL = useCL
#         self.dataType = dataType
#         self.useGroup = group
#         self.fistaLam = fistaLam
#         # self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
#         self.groups = np.linspace(0, 160, 161, dtype=np.int)
#         self.group_reg = group_reg
#         self.group_regs = torch.ones(len(self.groups)) * self.group_reg

#         self.BinaryCoding = GumbelSigmoid()

#         if self.useGroup:
#             self.sparseCoding = GroupLassoEncoder(self.Drr, self.Dtheta, self.fistaLam, self.groups, self.group_regs,
#                                                  self.gpu_id)
#             # self.sparseCoding = MaskDyanEncoder(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
#         else:
#             self.sparseCoding = DYANEnc(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
#         self.Classifier = classificationGlobal(num_class=self.num_class, Npole=self.Npole, dataType=self.dataType, useCL=self.useCL)

#     def forward(self, x,bi_thresh):
#         # sparseCode, Dict, R = self.sparseCoding.forward2(x, T) # w.o. RH
#         # bz, dims = x.shape[0], x.shape[-1]
#         T = x.shape[1]

#         if self.useGroup:
#             sparseCode,Dict, _ = self.sparseCoding(x, T)
#             # print('group lasso reg weights, l1, l2:', self.fistaLam, self.group_reg)
#         else:
#             sparseCode, Dict, _ = self.sparseCoding(x, T) # w.RH

#         'for GUMBEL'
#         binaryCode = self.BinaryCoding(sparseCode**2, bi_thresh, force_hard=True, temperature=0.1, inference=self.Inference)
#         temp1 = sparseCode * binaryCode
#         # temp = binaryCode.reshape(binaryCode.shape[0], self.Npole, int(x.shape[-1]/self.dim), self.dim)
#         Reconstruction = torch.matmul(Dict, temp1)
#         sparseFeat = binaryCode
#         # sparseFeat = torch.cat((binaryCode, sparseCode),1)
#         label, lastFeat = self.Classifier(sparseFeat)
#         # print('sparseCode:', sparseCode)

#         return label, lastFeat, binaryCode, Reconstruction

# class fusionLayers(nn.Module):
#     def __init__(self, num_class, in_chanel_x, in_chanel_y):
#         super(fusionLayers, self).__init__()
#         self.num_class = num_class
#         self.in_chanel_x = in_chanel_x
#         self.in_chanel_y = in_chanel_y
#         self.cat = nn.Linear(self.in_chanel_x + self.in_chanel_y, 128)
#         self.cls = nn.Linear(128, self.num_class)
#         self.relu = nn.LeakyReLU()
#     def forward(self, feat_x, feat_y):
#         twoStreamFeat = torch.cat((feat_x, feat_y), 1)
#         out = self.relu(self.cat(twoStreamFeat))
#         label = self.cls(out)
#         return label, out

# class twoStreamClassification(nn.Module):
#     def __init__(self, num_class, Npole, num_binary, Drr, Dtheta, dim, gpu_id, inference, fistaLam, dataType, kinetics_pretrain):
#         super(twoStreamClassification, self).__init__()
#         self.num_class = num_class
#         self.Npole = Npole
#         self.num_binary = num_binary
#         self.Drr = Drr
#         self.Dtheta = Dtheta
#         # self.PRE = PRE
#         self.gpu_id = gpu_id
#         self.dataType = dataType
#         self.dim = dim
#         self.kinetics_pretrain = kinetics_pretrain
#         self.Inference = inference

#         self.fistaLam = fistaLam
#         self.withMask = False

#         # self.dynamicsClassifier = Fullclassification(self.num_class, self.Npole,
#         #                         self.Drr, self.Dtheta, self.dim, self.dataType, self.Inference, self.gpu_id, self.fistaLam,self.withMask)


#         self.dynamicsClassifier = contrastiveNet(dim_embed=128, Npole=self.Npole, Drr=self.Drr, Dtheta=self.Dtheta, Inference=True, gpu_id=self.gpu_id, dim=2, dataType=self.dataType, fistaLam=fistaLam, fineTune=True)
#         self.RGBClassifier = RGBAction(self.num_class, self.kinetics_pretrain)

#         self.lastPred = fusionLayers(self.num_class, in_chanel_x=512, in_chanel_y=128)

#     def forward(self,skeleton, image, rois, fusion, bi_thresh):
#         # stream = 'fusion'
#         bz = skeleton.shape[0]
#         if bz == 1:
#             skeleton = skeleton.repeat(2,1,1,1)
#             image = image.repeat(2,1,1,1,1)
#             rois = rois.repeat(2,1,1,1,1)
#         label1,lastFeat_DIR, binaryCode, Reconstruction = self.dynamicsClassifier(skeleton, bi_thresh)
#         label2, lastFeat_CIR = self.RGBClassifier(image, rois)

#         if fusion:
#             label = {'RGB':label1, 'Dynamcis':label2}
#             feats = lastFeat_DIR
#         else:
#             # label = 0.5 * label1 + 0.5 * label2
#             label, feats= self.lastPred(lastFeat_DIR, lastFeat_CIR)
#         if bz == 1 :
#             nClip = int(label.shape[0]/2)
#             return label[0:nClip], binaryCode[0:nClip], Reconstruction[0:nClip], feats
#         else:
#             return label, binaryCode, Reconstruction, feats


# class MLP(nn.Module):
#     def __init__(self,  dim_in):
#         super(MLP, self).__init__()

#         self.layer1 = nn.Linear(dim_in, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.layer2 = nn.Linear(512, 128)
#         self.bn2 = nn.BatchNorm1d(128)
#         # self.gelu = nn.GELU()
#         self.relu = nn.LeakyReLU()
#         self.sig = nn.Sigmoid()

#     def forward(self,x):
#         x_out = self.relu(self.bn1(self.layer1(x)))
#         x_out = self.relu(self.bn2(self.layer2(x_out)))
#         # x_out = self.sig(x_out)

#         return x_out

# class contrastiveNet(nn.Module):
#     def __init__(self, dim_embed, Npole, Drr, Dtheta, Inference, gpu_id, dim, dataType, fistaLam,fineTune,useCL):
#         super(contrastiveNet, self).__init__()

#         # self.dim_in = dim_in
#         self.Npole = Npole
#         self.dim_embed = dim_embed
#         self.Drr = Drr
#         self.Dtheta = Dtheta
#         self.Inference = Inference
#         self.gpu_id = gpu_id
#         self.dim_data = dim
#         self.dataType = dataType
#         self.fistaLam = fistaLam
#         # self.withMask = False
#         self.useGroup = False
#         self.group_reg = 0.01
#         self.num_class = 10
#         self.fineTune = fineTune
#         self.useCL = useCL
#         # self.BinaryCoding = GumbelSigmoid()
#         # self.Classifier = classificationGlobal(num_class=self.num_class, Npole=self.Npole, dataType=self.dataType)
#         # self.mlpHead = MLP(self.dim_in)
#         # self.sparseCoding = DYANEnc(self.Drr, self.Dtheta, self.fistaLam, self.gpu_id)
#         self.backbone = Fullclassification(self.num_class, self.Npole, self.Drr, self.Dtheta, self.dim_data, self.dataType, self.Inference, self.gpu_id, self.fistaLam,self.useGroup, self.group_reg, self.useCL)
#         # if self.useCL == False:
#         #     dim_mlp = self.backbone.Classifier.cls.in_features
#         # else:
#         dim_mlp = self.backbone.Classifier.cls[0].in_features
#         self.proj = nn.Linear(dim_mlp,self.dim_embed)
#         self.relu = nn.LeakyReLU()
#         # if self.fineTune == False:
#         #     'remove projection layer'
#         #     # self.backbone.Classifier.cls = nn.Sequential(nn.Linear(dim_mlp, 512), nn.BatchNorm1d(512), nn.LeakyReLU(),
#         #     #                                          nn.Linear(512, dim_mlp), nn.BatchNorm1d(dim_mlp), nn.LeakyReLU(),
#         #     #                                          self.backbone.Classifier.cls)
#         #     self.backbone.Classifier.cls = nn.Sequential(self.backbone.Classifier.cls)
#         # else:
#         #     self.backbone.Classifier.cls = nn.Sequential(,nn.LeakyReLU(),self.backbone.Classifier.cls)

#     def forward(self, x, bi_thresh):
#         'x: affine skeleton'
#         bz = x.shape[0]
#         if len(x.shape) == 3:
#             x = x.unsqueeze(0)
#         x = x.reshape(x.shape[0]* x.shape[1], x.shape[2], x.shape[3])
#         nClip = int(x.shape[0]/bz)
#         if self.fineTune == False:
#             if bz < 2:
#                 x = x.repeat(2,1,1,1)
#                 bz = x.shape[0]
#             x1 = x[:,0]
#             x2 = x[:,1]
#             # _, lastFeat1, _, _ = self.backbone(x1, x1.shape[1])
#             # _, lastFeat2, _,_ = self.backbone(x2, x2.shape[1])
#             #
#             #
#             # z1 = F.normalize(self.mlpHead(lastFeat1), dim=1)
#             #
#             # z2 = F.normalize(self.mlpHead(lastFeat2),dim=1)

#             _, lastFeat1,_,_ = self.backbone(x1, bi_thresh)
#             _, lastFeat2,_,_ = self.backbone(x2, bi_thresh)

#             embedding1 = self.relu(self.proj(lastFeat1))
#             embedding2 = self.relu(self.proj(lastFeat2))


#             embed1 = torch.mean(embedding1.reshape(bz, nClip, embedding1.shape[-1]),1)
#             embed2 = torch.mean(embedding2.reshape(bz, nClip, embedding2.shape[-1]),1)
#             z1 = F.normalize(embed1, dim=1)
#             z2 = F.normalize(embed2, dim=1)

#             features = torch.cat([z1,z2], dim=0)
#             labels = torch.cat([torch.arange(bz) for i in range(2)], dim=0)
#             labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda(self.gpu_id)

#             simL_matrix = torch.matmul(features, features.T)
#             mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu_id)
#             labels = labels[~mask].view(labels.shape[0],-1)
#             simL_matrix = simL_matrix[~mask].view(simL_matrix.shape[0], -1)
#             positives = simL_matrix[labels.bool()].view(labels.shape[0], -1)
#             negatives = simL_matrix[~labels.bool()].view(simL_matrix.shape[0], -1)

#             logits = torch.cat([positives, negatives], dim=1)
#             labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.gpu_id)
#             temper = 0.07 #default
#             logits = logits/temper

#             return logits, labels
#         else:

#             return self.backbone(x, bi_thresh)



if __name__ == '__main__':
    gpu_id = 7

    N = 2*80
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    # net = twoStreamClassification(num_class=10, Npole=161, num_binary=161, Drr=Drr, Dtheta=Dtheta,
    #                               dim=2, gpu_id=gpu_id,inference=True, fistaLam=0.1, dataType='2D',
    #                               kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
    # x = torch.randn(5, 36, 50).cuda(gpu_id)
    # xImg = torch.randn(5, 20, 3, 224, 224).cuda(gpu_id)
    # T = x.shape[1]
    # xRois = xImg
    # label, _, _ = net(x, xImg, xRois, T, False)




    print('check')






