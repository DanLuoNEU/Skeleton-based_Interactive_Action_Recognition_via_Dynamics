# DataLoader for Interaction Recognition
import os
import time
import numpy as np
import random

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.tools import valid_crop_resize, random_rot

class NTU(Dataset):
    def __init__(self, data_dir="", mutual=True,
                 setup='cv', split='train',
                 random_choose=False, random_shift=False,random_move=False, random_rot=False,
                 window_size=120, normalization=False, debug=False, use_mmap=True,
                 limb=False, bone=False, vel=False, entity_rearrangement=False, label_path=None,
                 p_interval=[0.5,1], T_D=36):
        """
        data_dir:
        mutual: returned sequence from one video is with the same timestamps or not
        label_path:
        setup: cs | cv
        split:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        bone: use bone modality or not
        vel: use motion modality or not
        entity_rearrangement: If true, use entity rearrangement (interactive actions)
        """
        self.debug = debug
        self.data_path = os.path.join(data_dir,f"data/ntu/NTU60_{setup.upper()}.npz")
        self.mutual = mutual
        if self.mutual:
            self.num_classes = 11
        else:
            self.num_classes = 60
        self.label_path = label_path
        self.split = split
        self.setup = setup
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval if self.split == 'train' else [0.95]
        self.random_rot = random_rot
        self.limb = limb
        self.bone = bone
        self.vel = vel
        self.entity_rearrangement = entity_rearrangement
        self.load_data()
        if normalization:
            self.get_mean_map()
        self.num_clips = 10
        self.T_D=T_D
        
    def load_data(self):
        # data: N C V T M
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            
            # Mutual Actions
            index_needed = np.where((self.label > 48) & (self.label < 60))
            self.label = self.label[index_needed]
            self.data = self.data[index_needed] # (11864, 300, 150)Xset
            self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)

            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]

            # Mutual Actions
            index_needed = np.where((self.label > 48) & (self.label < 60))
            self.label = self.label[index_needed]
            self.data = self.data[index_needed] # (11864, 300, 150)Xset
            self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)

            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        # N: amount of all sample videos
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def sample_seq(self, data_i):
        """ multi sample for now
        Input
            data_i: (C,       T,      V,            , M
                     dim_ske, T(120), num_joints(25), num_subj(2))
        Return
            data_i: (num_clips, dim_ske, T_D, num_joints(25), num_subj(2))
        """
        C, T, V, M  = data_i.shape

        stride = T/self.num_clips

        ske_Clips=[]
        if self.mutual:
            ids_seq=[]
            for i in range(self.num_clips):
                id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                ids_seq.append(id_sample)
            for id in ids_seq:
                if id - int(self.T_D/2) < 0 and T - (id + int(self.T_D/2)) >= 0:
                    temp = np.expand_dims(data_i[:,0:self.T_D], 0)
                elif id - int(self.T_D/2) >= 0 and T - (id+int(self.T_D/2)) >= 0:
                    temp = np.expand_dims(data_i[:,id-int(self.T_D/2):id+int(self.T_D/2)], 0)
                elif id - int(self.T_D/2) >= 0 and T - (id+int(self.T_D/2)) < 0:
                    temp = np.expand_dims(data_i[:,T-self.T_D:], 0)
                ske_Clips.append(temp)
            ske_Clips = np.concatenate((ske_Clips), 0)
        else:
            for i_M in range(M):
                ids_seq=[]
                for i in range(self.num_clips):
                    id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                    ids_seq.append(id_sample)
                ske_M = []
                for id in ids_seq:
                    if id - int(self.T_D/2) < 0 and T - (id + int(self.T_D/2)) >= 0:
                        temp = np.expand_dims(data_i[:,0:self.T_D,:,i_M], (0,4))
                    elif id - int(self.T_D/2) >= 0 and T - (id+int(self.T_D/2)) >= 0:
                        temp = np.expand_dims(data_i[:,id-int(self.T_D/2):id+int(self.T_D/2),:,i_M], (0,4))
                    elif id - int(self.T_D/2) >= 0 and T - (id+int(self.T_D/2)) < 0:
                        temp = np.expand_dims(data_i[:,T-self.T_D:,:,i_M], (0,4))
                    ske_M.append(temp)
                ske_M = np.concatenate((ske_M), 0)
                ske_Clips.append(ske_M)
            ske_Clips = np.concatenate((ske_Clips), 4)
        
        return ske_Clips

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        """
        data_numpy: 
        """
        data_numpy = self.data[index]   # (T,M x V x C | T(120), num_subj(2) x num_joints(25) x dim_ske(3))
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = random_rot(data_numpy)
        if self.entity_rearrangement:
            data_numpy = data_numpy[:,:,:,torch.randperm(data_numpy.size(3))]
        if self.limb:
            limb_pairs = ((11,10),(10,9),(5,6),(6,7),      # Arm R+L
                          (17,18),(18,19),(13,14),(14,15), # Leg R+L
                          (21,4))                          # Head & Shoulder Center
            num_limbs = len(limb_pairs)
            C,T,V,M = data_numpy.shape
            limb_data_numpy = np.zeros((C,T,V+num_limbs,M), dtype=data_numpy.dtype)
            limb_data_numpy[:,:,:V,:] = data_numpy
            for i_limb,(l1,l2) in enumerate(limb_pairs):
                limb_data_numpy[:,:,V+i_limb,:] = (data_numpy[:, :, l1 - 1] + data_numpy[:, :, l2 - 1])/2
            data_numpy = limb_data_numpy
        if self.bone:
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        # Multi Clips
        data_numpy=self.sample_seq(data_numpy)
        
        return data_numpy, label, index
    
if __name__ == '__main__':
    print(time.asctime( time.localtime(time.time())))
    dataset = NTU(data_dir="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/21_CTR-GCN")
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=8, 
                                             shuffle=True,drop_last=True, pin_memory=True)
    
    for id_b,(data,label,index) in enumerate(dataloader):
        print(time.asctime( time.localtime(time.time()) ))
        print(id_b)
        pass