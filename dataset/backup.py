# DataLoader for Interaction Recognition
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.tools import valid_crop_resize, random_rot

class NTUDataset(Dataset):
    def __init__(self, data_dir="", label_path=None, p_interval=[0.5,1], setup='cv', split='train',
                 random_choose=False, random_shift=False,random_move=False, random_rot=True,
                 window_size=120, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False, entity_rearrangement=True):
        """
        data_path:
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
        self.label_path = label_path
        self.split = split
        self.setup = setup
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        if self.split == 'train':
            self.p_interval = p_interval
        else:
            self.p_interval = [0.95]
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.entity_rearrangement = entity_rearrangement
        self.load_data()
        if normalization:
            self.get_mean_map()

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
            index_needed = np.where(((self.label > 48) & (self.label < 60)) | (self.label > 104))
            self.label = self.label[index_needed]
            self.data = self.data[index_needed] # (11864, 300, 150)Xset
            self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)
            self.label = np.where((self.label > 104), self.label-94, self.label)

            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]

            # Mutual Actions
            index_needed = np.where(((self.label > 48) & (self.label < 60)) | (self.label > 104))
            self.label = self.label[index_needed]
            self.data = self.data[index_needed] # (11864, 300, 150)Xset
            self.label = np.where(((self.label > 48) & (self.label < 60)), self.label-49, self.label)
            self.label = np.where((self.label > 104), self.label-94, self.label)

            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = random_rot(data_numpy)
        if self.entity_rearrangement:
            data_numpy = data_numpy[:,:,:,torch.randperm(data_numpy.size(3))]
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

        return data_numpy, label, index


# # IRNet 
# import os
# import glob
# import numpy as np
# import pandas as pd

# import torch
# from torch.utils.data import Dataset

# ### Summarized names
# ACTIONS = [
#     "drink water","eat meal/snack","brushing teeth","brushing hair",
#     "drop","pickup","throw","sitting down","standing up (from sitting position)",
#     "clapping","reading","writing","tear up paper","wear jacket","take off jacket",
#     "wear a shoe","take off a shoe","wear on glasses","take off glasses",
#     "put on a hat/cap","take off a hat/cap","cheer up","hand waving",
#     "kicking something","put something inside pocket / take out something from pocket",
#     "hopping (one foot jumping)","jump up","make a phone call/answer phone",
#     "playing with phone/tablet","typing on a keyboard (pc/laptop)",
#     "pointing to something with finger","taking a selfie","check time (from watch)",
#     "rub two hands together","nod head/bow (Japanese)","shake head",
#     "wipe face (using napkin/towel)","salute (policeman)",
#     "put the palms together (Indians/Thai) / make a bow with hands folded in front (Chinese)",
#     "cross hands in front (say stop)","sneeze/cough","staggering",
#     "falling","touch head (headache)","touch chest (stomach-ache/heart pain)",
#     "touch back (backache)","touch neck (neck-ache)","nausea or vomiting condition",
#     "use a fan (with hand or paper)/feeling warm","Punch/slapping","Kicking",
#     "Pushing","PattingOnBack","PointingFinger","Hugging","GiveSomething",
#     "TouchingPocket","Handshaking","WalkingTowards","WalkingApart"]

# # https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt
# # "302 of the captured samples in the "NTU RGB+D" dataset have missing or incomplete skeleton data.
# # If you are working on skeleton-based analysis, please ignore these files in your training and testing procedures."

# IGNORE_LIST = [
#     'S001C002P005R002A008','S001C002P006R001A008','S001C003P002R001A055','S001C003P002R002A012','S001C003P005R002A004',
#     'S001C003P005R002A005','S001C003P005R002A006','S001C003P006R002A008','S002C002P011R002A030','S002C003P008R001A020',
#     'S002C003P010R002A010','S002C003P011R002A007','S002C003P011R002A011','S002C003P014R002A007','S003C001P019R001A055',
#     'S003C002P002R002A055','S003C002P018R002A055','S003C003P002R001A055','S003C003P016R001A055','S003C003P018R002A024',
#     'S004C002P003R001A013','S004C002P008R001A009','S004C002P020R001A003','S004C002P020R001A004','S004C002P020R001A012',
#     'S004C002P020R001A020','S004C002P020R001A021','S004C002P020R001A036','S005C002P004R001A001','S005C002P004R001A003',
#     'S005C002P010R001A016','S005C002P010R001A017','S005C002P010R001A048','S005C002P010R001A049','S005C002P016R001A009',
#     'S005C002P016R001A010','S005C002P018R001A003','S005C002P018R001A028','S005C002P018R001A029','S005C003P016R002A009',
#     'S005C003P018R002A013','S005C003P021R002A057','S006C001P001R002A055','S006C002P007R001A005','S006C002P007R001A006',
#     'S006C002P016R001A043','S006C002P016R001A051','S006C002P016R001A052','S006C002P022R001A012','S006C002P023R001A020',
#     'S006C002P023R001A021','S006C002P023R001A022','S006C002P023R001A023','S006C002P024R001A018','S006C002P024R001A019',
#     'S006C003P001R002A013','S006C003P007R002A009','S006C003P007R002A010','S006C003P007R002A025','S006C003P016R001A060',
#     'S006C003P017R001A055','S006C003P017R002A013','S006C003P017R002A014','S006C003P017R002A015','S006C003P022R002A013',
#     'S007C001P018R002A050','S007C001P025R002A051','S007C001P028R001A050','S007C001P028R001A051','S007C001P028R001A052',
#     'S007C002P008R002A008','S007C002P015R002A055','S007C002P026R001A008','S007C002P026R001A009','S007C002P026R001A010',
#     'S007C002P026R001A011','S007C002P026R001A012','S007C002P026R001A050','S007C002P027R001A011','S007C002P027R001A013',
#     'S007C002P028R002A055','S007C003P007R001A002','S007C003P007R001A004','S007C003P019R001A060','S007C003P027R002A001',
#     'S007C003P027R002A002','S007C003P027R002A003','S007C003P027R002A004','S007C003P027R002A005','S007C003P027R002A006',
#     'S007C003P027R002A007','S007C003P027R002A008','S007C003P027R002A009','S007C003P027R002A010','S007C003P027R002A011',
#     'S007C003P027R002A012','S007C003P027R002A013','S008C002P001R001A009','S008C002P001R001A010','S008C002P001R001A014',
#     'S008C002P001R001A015','S008C002P001R001A016','S008C002P001R001A018','S008C002P001R001A019','S008C002P008R002A059',
#     'S008C002P025R001A060','S008C002P029R001A004','S008C002P031R001A005','S008C002P031R001A006','S008C002P032R001A018',
#     'S008C002P034R001A018','S008C002P034R001A019','S008C002P035R001A059','S008C002P035R002A002','S008C002P035R002A005',
#     'S008C003P007R001A009','S008C003P007R001A016','S008C003P007R001A017','S008C003P007R001A018','S008C003P007R001A019',
#     'S008C003P007R001A020','S008C003P007R001A021','S008C003P007R001A022','S008C003P007R001A023','S008C003P007R001A025',
#     'S008C003P007R001A026','S008C003P007R001A028','S008C003P007R001A029','S008C003P007R002A003','S008C003P008R002A050',
#     'S008C003P025R002A002','S008C003P025R002A011','S008C003P025R002A012','S008C003P025R002A016','S008C003P025R002A020',
#     'S008C003P025R002A022','S008C003P025R002A023','S008C003P025R002A030','S008C003P025R002A031','S008C003P025R002A032',
#     'S008C003P025R002A033','S008C003P025R002A049','S008C003P025R002A060','S008C003P031R001A001','S008C003P031R002A004',
#     'S008C003P031R002A014','S008C003P031R002A015','S008C003P031R002A016','S008C003P031R002A017','S008C003P032R002A013',
#     'S008C003P033R002A001','S008C003P033R002A011','S008C003P033R002A012','S008C003P034R002A001','S008C003P034R002A012',
#     'S008C003P034R002A022','S008C003P034R002A023','S008C003P034R002A024','S008C003P034R002A044','S008C003P034R002A045',
#     'S008C003P035R002A016','S008C003P035R002A017','S008C003P035R002A018','S008C003P035R002A019','S008C003P035R002A020',
#     'S008C003P035R002A021','S009C002P007R001A001','S009C002P007R001A003','S009C002P007R001A014','S009C002P008R001A014',
#     'S009C002P015R002A050','S009C002P016R001A002','S009C002P017R001A028','S009C002P017R001A029','S009C003P017R002A030',
#     'S009C003P025R002A054','S010C001P007R002A020','S010C002P016R002A055','S010C002P017R001A005','S010C002P017R001A018',
#     'S010C002P017R001A019','S010C002P019R001A001','S010C002P025R001A012','S010C003P007R002A043','S010C003P008R002A003',
#     'S010C003P016R001A055','S010C003P017R002A055','S011C001P002R001A008','S011C001P018R002A050','S011C002P008R002A059',
#     'S011C002P016R002A055','S011C002P017R001A020','S011C002P017R001A021','S011C002P018R002A055','S011C002P027R001A009',
#     'S011C002P027R001A010','S011C002P027R001A037','S011C003P001R001A055','S011C003P002R001A055','S011C003P008R002A012',
#     'S011C003P015R001A055','S011C003P016R001A055','S011C003P019R001A055','S011C003P025R001A055','S011C003P028R002A055',
#     'S012C001P019R001A060','S012C001P019R002A060','S012C002P015R001A055','S012C002P017R002A012','S012C002P025R001A060',
#     'S012C003P008R001A057','S012C003P015R001A055','S012C003P015R002A055','S012C003P016R001A055','S012C003P017R002A055',
#     'S012C003P018R001A055','S012C003P018R001A057','S012C003P019R002A011','S012C003P019R002A012','S012C003P025R001A055',
#     'S012C003P027R001A055','S012C003P027R002A009','S012C003P028R001A035','S012C003P028R002A055','S013C001P015R001A054',
#     'S013C001P017R002A054','S013C001P018R001A016','S013C001P028R001A040','S013C002P015R001A054','S013C002P017R002A054',
#     'S013C002P028R001A040','S013C003P008R002A059','S013C003P015R001A054','S013C003P017R002A054','S013C003P025R002A022',
#     'S013C003P027R001A055','S013C003P028R001A040','S014C001P027R002A040','S014C002P015R001A003','S014C002P019R001A029',
#     'S014C002P025R002A059','S014C002P027R002A040','S014C002P039R001A050','S014C003P007R002A059','S014C003P015R002A055',
#     'S014C003P019R002A055','S014C003P025R001A048','S014C003P027R002A040','S015C001P008R002A040','S015C001P016R001A055',
#     'S015C001P017R001A055','S015C001P017R002A055','S015C002P007R001A059','S015C002P008R001A003','S015C002P008R001A004',
#     'S015C002P008R002A040','S015C002P015R001A002','S015C002P016R001A001','S015C002P016R002A055','S015C003P008R002A007',
#     'S015C003P008R002A011','S015C003P008R002A012','S015C003P008R002A028','S015C003P008R002A040','S015C003P025R002A012',
#     'S015C003P025R002A017','S015C003P025R002A020','S015C003P025R002A021','S015C003P025R002A030','S015C003P025R002A033',
#     'S015C003P025R002A034','S015C003P025R002A036','S015C003P025R002A037','S015C003P025R002A044','S016C001P019R002A040',
#     'S016C001P025R001A011','S016C001P025R001A012','S016C001P025R001A060','S016C001P040R001A055','S016C001P040R002A055',
#     'S016C002P008R001A011','S016C002P019R002A040','S016C002P025R002A012','S016C003P008R001A011','S016C003P008R002A002',
#     'S016C003P008R002A003','S016C003P008R002A004','S016C003P008R002A006','S016C003P008R002A009','S016C003P019R002A040',
#     'S016C003P039R002A016','S017C001P016R002A031','S017C002P007R001A013','S017C002P008R001A009','S017C002P015R001A042',
#     'S017C002P016R002A031','S017C002P016R002A055','S017C003P007R002A013','S017C003P008R001A059','S017C003P016R002A031',
#     'S017C003P017R001A055','S017C003P020R001A059']

# TRAIN_SUBJECTS = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]

# class NTUdataset(Dataset):
#     def __init__(self, data_dir='',
#                  setup='CV', split='train'):
#         """
#         :param data_dir: /path/to/IRNet/preprocessed_skeleton_data
#             Folder structure
#             |-data/ntu    
#                 |-descs.csv
#                 |-skl.csv
#         :param setup: CV | CS
#         :param split: training set or test set
#         """
#         self.data_dir = os.path.join(data_dir,"data/ntu")
#         self.setup = setup
#         self.split = split
#         self.load_data(setup, split)
        
#     def load_data(self, setup, split):
#         # data: N C V T M
#         # Rerturn:
#         #   self.data: num_seq x dim_ske x length_seq x num_joints x num_ske
#         ground_truth = pd.read_csv(self.data_dir+'/descs.csv',
#                                    index_col=False, header=None).T
#         ground_truth.columns = ['setup','camera','subject','duplicate','action',
#                                 'start_frame_pt','end_frame_pt',]
#         ground_truth = ground_truth[ground_truth.action >= 50]
#         ground_truth.action = ground_truth.action - 1
#         ground_truth['DATA_DIR'] = self.data_dir

#         if split == 'CS':
#             if split == 'train':
#                 gt_split = ground_truth[ground_truth.subject.isin(TRAIN_SUBJECTS)]
#             else:
#                 gt_split = ground_truth[~ground_truth.subject.isin(TRAIN_SUBJECTS)]
#         elif split == 'CV':
#             if split == 'train':
#                 gt_split = ground_truth[ground_truth.camera != 1]
#             else:
#                 gt_split = ground_truth[ground_truth.camera == 1]        
#         # self.X, self.Y = data_io.get_data(gt_split)
#     def __len__(self):
#         return len(self.X)
#     def __iter__(self):
#         return self
#     def __getitem__(self, index):
#         return data_numpy, label, index

# # CTR_GCN way to do this
# import tools
# ntu_pairs = (
#     (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
#     (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
#     (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
#     (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
# )
# class NTUdataset(Dataset):
#     def __init__(self, data_root='./', setup='CV', 
#                  debug=False, p_interval=1, split='test', 
#                  random_choose=False, random_shift=False, 
#                  random_move=False, random_rot=False, 
#                  window_size=-1, normalization=False,
#                  use_mmap=False, bone=False, vel=False):
#         """
#         :param data_path: /path/to/CTR-GCN/preprocessed_skeleton_data
#         :param setup: CV | CS
#         :param split: training set or test set
#         :param random_choose: If true, randomly choose a portion of the input sequence
#         :param random_shift: If true, randomly pad zeros at the begining or end of sequence
#         :param random_move:
#         :param random_rot: rotate skeleton around xyz axis
#         :param window_size: The length of the output sequence
#         :param normalization: If true, normalize input sequence
#         :param debug: If true, only use the first 100 samples
#         :param use_mmap: If true, use mmap mode to load data, which can save the running memory
#         :param bone: use bone modality or not
#         :param vel: use motion modality or not
#         :param only_label: only load label for ensemble score compute
#         """
#         self.debug = debug
#         self.setup = setup
#         self.data_path = os.path.join(data_root,f'data/ntu/NTU60_{self.setup}.npz')
#         self.split = split
#         self.random_choose = random_choose
#         self.random_shift = random_shift
#         self.random_move = random_move
#         self.window_size = window_size
#         self.normalization = normalization
#         self.use_mmap = use_mmap
#         self.p_interval = p_interval
#         self.random_rot = random_rot
#         self.bone = bone
#         self.vel = vel
#         self.load_data()
#         if normalization:
#             self.get_mean_map()
        
#     def load_data(self):
#         # data: N C V T M
#         # Rerturn:
#         #   self.data: num_seq x dim_ske x length_seq x num_joints x num_ske
#         npz_data = np.load(self.data_path)
#         "keys:['x_train','x_test','y_train','y_test']"
#         if self.split == 'train':
#             self.label = np.where(npz_data['y_train'] > 0)[1]
#             self.list_mutual = np.where(self.label >= 50)
#             self.data = npz_data['x_train'][self.list_mutual,][0]
#             self.label = self.label[self.list_mutual]
#             self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
#         elif self.split == 'test':
#             self.label = list(np.where(npz_data['y_test'] > 0))
#             self.list_mutual = np.where(self.label[1] >= 50)
#             self.label[0] = self.label[0][self.list_mutual]
#             self.label[1] = self.label[1][self.list_mutual]
#             self.label = tuple(self.label)
#             self.data = npz_data['x_test'][self.list_mutual,][0]
#             self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
#         else:
#             raise NotImplementedError('data split only supports train/test')
#         N, T, _ = self.data.shape
#         self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    
#     def get_mean_map(self):
#         "get normalization data"
#         data = self.data
#         N, C, T, V, M = data.shape
#         self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
#         self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

#     def __len__(self):
#         return len(self.label)

#     def __iter__(self):
#         return self

#     def __getitem__(self, index):
#         data_numpy = self.data[index]
#         label = self.label[index]
#         data_numpy = np.array(data_numpy)
#         valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
#         # reshape Tx(MVC) to CTVM
#         data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
#         if self.random_rot:
#             data_numpy = tools.random_rot(data_numpy)
#         if self.bone:
#             bone_data_numpy = np.zeros_like(data_numpy)
#             for v1, v2 in ntu_pairs:
#                 bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
#             data_numpy = bone_data_numpy
#         if self.vel:
#             data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
#             data_numpy[:, -1] = 0

#         return data_numpy, label, index

#     def top_k(self, score, top_k):
#         rank = score.argsort()
#         hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
#         return sum(hit_top_k) * 1.0 / len(hit_top_k)
        
# import sys
# sys.path.append('../')
# sys.path.append('.')
# import os
# import numpy as np
# import random
# from PIL import Image, ImageFilter, ImageEnhance

# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# CVAC Dataset
# class NTU_dataset(Dataset):
#     """Northeastern-UCLA Dataset Skeleton Dataset, cross view experiment,
#         Access input skeleton sequence, GT label
#         When T=0, it returns the whole
#     """

#     def __init__(self, root_list, nanList, dataType, sampling, phase, cam, T, maskType, test_view):
#         self.root_list = root_list
#         self.image_root = '/data/NTU-RGBD/frames/'
#         self.dataType = dataType
#         self.clips = 6
#         self.nanList = nanList
#         self.maskType = maskType
#         self.phase = phase
#         self.sampling = sampling
#         self.test_view = test_view
#         self.T = T
#         if dataType == '3D':
#             self.root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
#         else:
#             self.root_skeleton = "/data/NTU-RGBD/poses_60"


#         # self.root_list = root_list
#         self.view = []
#         self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         for name_cam in cam.split(','):
#             self.view.append(name_cam)

#         self.name_list = []
#         if self.phase == 'train':
#             for view in self.view:
#                 list_file = os.path.join(self.root_list, f"skeleton_all_{view}.list")
#                 list_txt = np.loadtxt(list_file, dtype=str)
#                 for item in list_txt:
#                     if item[0] not in self.nanList:
#                         self.name_list.append(item[0])

#         else:
#             list_file = os.path.join(self.root_list, f"skeleton_all_{test_view}.list")
#             list_txt = np.loadtxt(list_file, dtype=str)
#             for item in list_txt:
#                 if item[0] not in self.nanList:
#                     self.name_list.append(item[0])

#     def __len__(self):
#         return len(self.name_list)
#         # return 50

#     def get_uniNorm(self, skeleton):

#         'skeleton: T X 25 x 2, norm[0,1], (x-min)/(max-min)'
#         # nonZeroSkeleton = []
#         if self.dataType == '2D':
#             dim = 2
#         else:
#             dim = 3
#         normSkeleton = np.zeros_like(skeleton)
#         visibility = np.zeros(skeleton.shape)
#         bbox = np.zeros((skeleton.shape[0], 4))
#         for i in range(0, skeleton.shape[0]):
#             nonZeros = []
#             ids = []
#             normPose = np.zeros_like((skeleton[i]))
#             for j in range(0, skeleton.shape[1]):
#                 point = skeleton[i,j]

#                 if point[0] !=0 and point[1] !=0:

#                     nonZeros.append(point)
#                     ids.append(j)

#             nonzeros = np.concatenate((nonZeros)).reshape(len(nonZeros), dim)
#             minX, minY = np.min(nonzeros[:,0]), np.min(nonzeros[:,1])
#             maxX, maxY = np.max(nonzeros[:,0]), np.max(nonzeros[:,1])
#             normPose[ids,0] = (nonzeros[:,0] - minX)/(maxX-minX)
#             normPose[ids,1] = (nonzeros[:,1] - minY)/(maxY-minY)
#             if dim == 3:
#                 minZ, maxZ = np.min(nonzeros[:,2]), np.max(nonzeros[:,2])
#                 normPose[ids,2] = (nonzeros[:,1] - minZ)/(maxZ-minZ)
#             normSkeleton[i] = normPose
#             visibility[i,ids] = 1
#             bbox[i] = np.asarray([minX, minY, maxX, maxY])

#         return normSkeleton, visibility, bbox

#     def get_rgbList(self, name_sample):
#         image_path = os.path.join(self.image_root, name_sample)

#         imgId = []
#         imageList = []

#         for item in os.listdir(image_path):
#             if item.find('.jpg') != -1:
#                 id = int(item.split('_')[1].split('.jpg')[0])
#                 imgId.append(id)

#         imgId.sort()

#         for i in range(0, len(imgId)):
#             for item in os.listdir(image_path):
#                 if item.find('.jpg') != -1:
#                     if int(item.split('_')[1].split('.jpg')[0]) == imgId[i]:
#                         imageList.append(item)

#         'make sure it is sorted'
#         return imageList, image_path

#     def get_rgb_data(self, data_path, imageList):
#         imgSize = []
#         imgSequence = []
#         imgSequenceOrig = []

#         for i in range(0, len(imageList)):
#             img_path = os.path.join(data_path, imageList[i])
#             # orig_image = cv2.imread(img_path)
#             # imgSequenceOrig.append(np.expand_dims(orig_image,0))

#             input_image = Image.open(img_path)
#             imgSize.append(input_image.size)
#             imgSequenceOrig.append(np.expand_dims(input_image, 0))


#             img_tensor = self.transform(input_image)

#             imgSequence.append(img_tensor.unsqueeze(0))

#         imgSequence = torch.cat((imgSequence), 0)
#         imgSequenceOrig = np.concatenate((imgSequenceOrig), 0)

#         return imgSequence, imgSize, imgSequenceOrig

#     def pose_to_heatmap(self, poses, image_size, outRes):
#         ''' Pose to Heatmap
#         Argument:
#             joints: T x njoints x 2
#         Return:
#             heatmaps: T x 64 x 64
#         '''
#         GaussSigma = 1

#         T = poses.shape[0]
#         H = image_size[0]
#         W = image_size[1]
#         heatmaps = []
#         for t in range(0, T):
#             pts = poses[t]  # njoints x 2
#             out = np.zeros((pts.shape[0], outRes, outRes))

#             for i in range(0, pts.shape[0]):
#                 pt = pts[i]
#                 if pt[0] == 0 and pt[1] == 0:
#                     out[i] = np.zeros((outRes, outRes))
#                 else:
#                     newPt = np.array([outRes * (pt[0] / W), outRes * (pt[1] / H)])
#                     out[i] = DrawGaussian(out[i], newPt, GaussSigma)
#             # out_max = np.max(out, axis=0)
#             # heatmaps.append(out_max)
#             heatmaps.append(out)   # heatmaps = 20x64x64
#         stacked_heatmaps = np.stack(heatmaps, axis=0)
#         min_offset = -1 * np.amin(stacked_heatmaps)
#         stacked_heatmaps = stacked_heatmaps + min_offset
#         max_value = np.amax(stacked_heatmaps)
#         if max_value == 0:
#             return stacked_heatmaps
#         stacked_heatmaps = stacked_heatmaps / max_value

#         return stacked_heatmaps

#     def getROIs(self, orignImages, bboxes):
#         assert orignImages.shape[0] == bboxes.shape[0]
#         ROIs = []
#         for i in range(0, orignImages.shape[0]):
#             image = orignImages[i]
#             bbox = bboxes[i]
#             x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
#             #
#             # W = int(x_max - x_min)
#             # H = int(y_max - y_min)

#             crop_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
#             crop_image_tensor = self.transform(crop_image)
#             ROIs.append(crop_image_tensor.unsqueeze(0))
#         ROIs = torch.cat((ROIs), 0)
#         return ROIs

#     def paddingSeq(self, skeleton, normSkeleton, imageSequence, ROIs, visibility):
#         Tadd = abs(skeleton.shape[0] - self.T)

#         last = np.expand_dims(skeleton[-1, :, :], 0)
#         copyLast = np.repeat(last, Tadd, 0)
#         skeleton_New = np.concatenate((skeleton, copyLast), 0)  # copy last frame Tadd times

#         lastNorm = np.expand_dims(normSkeleton[-1, :, :], 0)
#         copyLastNorm = np.repeat(lastNorm, Tadd, 0)
#         normSkeleton_New = np.concatenate((normSkeleton, copyLastNorm), 0)

#         lastMask = np.expand_dims(visibility[-1,:,:], 0)
#         copyLastMask = np.repeat(lastMask, Tadd, 0)
#         visibility_New = np.concatenate((visibility, copyLastMask), 0)

#         lastImg = imageSequence[-1, :, :, :].unsqueeze(0)
#         copyLastImg = lastImg.repeat(Tadd, 1, 1, 1)
#         imageSequence_New = torch.cat((imageSequence, copyLastImg), 0)

#         lastROI = ROIs[-1, :,:,:].unsqueeze(0)
#         copyLastROI = lastROI.repeat(Tadd, 1, 1, 1)
#         ROIs_New = torch.cat((ROIs, copyLastROI), 0)

#         return skeleton_New, normSkeleton_New, imageSequence_New, ROIs_New, visibility_New

#     def get_data(self, name_sample):
#         imagesList, image_path = self.get_rgbList(name_sample)
#         jsonList, imgList = alignDataList(self.root_skeleton, name_sample, imagesList,'NTU')

#         assert len(imgList) == len(jsonList)

#         imageSequence, imageSize, imageSequence_orig = self.get_rgb_data(image_path, imgList)


#         if self.dataType == '2D':
#             skeleton, usedID, confidence = getJsonData(self.root_skeleton, name_sample, jsonList)
#             imageSequence = imageSequence[usedID]
#             imageSequence_orig = imageSequence_orig[usedID]

#         else:
#             skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()['skel_body0']
#             confidence = np.ones_like(skeleton)
#         #


#         T_sample, num_joints, dim = skeleton.shape
#         normSkeleton, binaryMask, bboxes = self.get_uniNorm(skeleton)

#         if self.maskType == 'binary':
#             visibility = binaryMask
#         else:
#             visibility = confidence  # mask is from confidence score

#         # visibility = binaryMask # mask is 0/1
#         ROIs = self.getROIs(imageSequence_orig, bboxes)

#         if self.T == 0:
#             skeleton_input = skeleton
#             imageSequence_input = imageSequence
#             normSkeleton_input = normSkeleton
#             ROIs_input = ROIs
#             visibility_input = visibility
#             # imgSequence = np.zeros((T_sample, 3, 224, 224))
#             details = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': range(T_sample)}
#         else:
#             if T_sample <= self.T:
#                 skeleton_input = skeleton
#                 normSkeleton_input = normSkeleton
#                 imageSequence_input = imageSequence
#                 ROIs_input = ROIs
#                 visibility_input = visibility
#             else:
#                 # skeleton_input = skeleton[0::self.ds, :, :]
#                 # imageSequence_input = imageSequence[0::self.ds]

#                 stride = T_sample / self.T
#                 ids_sample = []
#                 for i in range(self.T):
#                     id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
#                     ids_sample.append(id_sample)

#                 skeleton_input = skeleton[ids_sample, :, :]
#                 imageSequence_input = imageSequence[ids_sample]
#                 normSkeleton_input = normSkeleton[ids_sample, :, :]
#                 ROIs_input = ROIs[ids_sample]
#                 visibility_input = visibility[ids_sample, :, :]

#             if skeleton_input.shape[0] != self.T:
#                 skeleton_input, normSkeleton_input, imageSequence_input, ROIs_input, visibility_input \
#                     = self.paddingSeq(skeleton_input, normSkeleton_input, imageSequence_input, ROIs_input,
#                                       visibility_input)

#         imgSize = (1980, 1080)

#         # normSkeleton, _ = self.get_uniNorm(skeleton_input)
#         heatmap_to_use = self.pose_to_heatmap(skeleton_input, imgSize, 64)
#         skeletonData = {'normSkeleton': normSkeleton_input, 'unNormSkeleton': skeleton_input,
#                         'visibility': visibility_input}
#         # print('heatsize:', heatmap_to_use.shape[0], 'imgsize:', imageSequence_input.shape[0], 'skeleton size:', normSkeleton.shape[0])
#         assert heatmap_to_use.shape[0] == self.T
#         assert normSkeleton_input.shape[0] == self.T
#         assert imageSequence_input.shape[0] == self.T

#         return heatmap_to_use, imageSequence_input, skeletonData, ROIs_input

#     def get_data_multiSeq(self,  name_sample):
#         imagesList, data_path = self.get_rgbList(name_sample)
#         jsonList, imgList = alignDataList(self.root_skeleton, name_sample, imagesList,'NTU')

#         assert len(imgList) == len(jsonList)
#         imageSequence, _, imageSequence_orig = self.get_rgb_data(data_path, imgList)

#         if self.dataType == '2D':
#             skeleton, usedID, confidence = getJsonData(self.root_skeleton, name_sample, jsonList)
#             imageSequence = imageSequence[usedID]
#             imageSequence_orig = imageSequence_orig[usedID]
#         else:
#             skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
#                 'skel_body0']
#             confidence = np.ones_like(skeleton)

#         normSkeleton, binaryMask, bboxes = self.get_uniNorm(skeleton)
#         ROIs = self.getROIs(imageSequence_orig, bboxes)

#         if self.maskType == 'binary':
#             visibility = binaryMask
#         else:
#             visibility = confidence  # mask is from confidence score

#         T_sample, num_joints, dim = normSkeleton.shape
#         stride = T_sample / self.clips
#         ids_sample = []

#         for i in range(self.clips):
#             id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
#             ids_sample.append(id_sample)
#         if T_sample <= self.T:
#             skeleton_input, normSkeleton_input, imageSequence_inp, ROIs_inp, visibility_input = self.paddingSeq(skeleton, normSkeleton,
#                                                                                     imageSequence, ROIs,visibility)
#             temp = np.expand_dims(normSkeleton_input, 0)
#             inpSkeleton_all = np.repeat(temp, self.clips, 0)

#             tempMask = np.expand_dims(visibility_input, 0)
#             visibility_input = np.repeat(tempMask, self.clips, 0)

#             tempImg = np.expand_dims(imageSequence_inp, 0)
#             imageSequence_input = np.repeat(tempImg, self.clips, 0)

#             temp_skl = np.expand_dims(skeleton_input, 0)
#             skeleton_all = np.repeat(temp_skl, self.clips, 0)

#             heatmaps = self.pose_to_heatmap(skeleton_input, (640, 480), 64)
#             tempHeat = np.expand_dims(heatmaps, 0)
#             heatmap_to_use = np.repeat(tempHeat, self.clips, 0)

#             temRoi= np.expand_dims(ROIs_inp, 0)
#             ROIs_input = np.repeat(temRoi, self.clips, 0)

#         else: # T_sample > self.T

#             inpSkeleton_all = []
#             imageSequence_input = []
#             visibility_input = []
#             heatmap_to_use = []
#             skeleton_all = []
#             ROIs_input = []
#             heatmaps = self.pose_to_heatmap(skeleton, (640, 480), 64)
#             for id in ids_sample:

#                 if (id - int(self.T / 2)) <= 0 < (id + int(self.T / 2)) < T_sample:

#                     temp = np.expand_dims(normSkeleton[0:self.T], 0)
#                     tempImg = np.expand_dims(imageSequence[0:self.T], 0)
#                     temp_skl = np.expand_dims(skeleton[0:self.T], 0)
#                     tempHeat = np.expand_dims(heatmaps[0:self.T], 0)
#                     temRoi = np.expand_dims(ROIs[0:self.T], 0)
#                     tempMask = np.expand_dims(visibility[0:self.T], 0)

#                 elif 0 < (id-int(self.T/2)) <= (id + int(self.T / 2)) < T_sample:
#                     temp = np.expand_dims(normSkeleton[id - int(self.T / 2):id + int(self.T / 2)], 0)
#                     tempImg = np.expand_dims(imageSequence[id - int(self.T / 2):id + int(self.T / 2)], 0)
#                     temp_skl = np.expand_dims(skeleton[id - int(self.T / 2):id + int(self.T / 2)], 0)
#                     tempHeat = np.expand_dims(heatmaps[id - int(self.T / 2):id + int(self.T / 2)], 0)
#                     temRoi = np.expand_dims(ROIs[id - int(self.T / 2):id + int(self.T / 2)], 0)
#                     tempMask = np.expand_dims(visibility[id - int(self.T / 2):id + int(self.T / 2)],0)

#                 elif (id - int(self.T/2)) < T_sample <= (id+int(self.T / 2)):

#                     temp = np.expand_dims(normSkeleton[T_sample - self.T:], 0)
#                     tempImg = np.expand_dims(imageSequence[T_sample - self.T:], 0)
#                     temp_skl = np.expand_dims(skeleton[T_sample - self.T:], 0)
#                     tempHeat = np.expand_dims(heatmaps[T_sample - self.T:], 0)
#                     temRoi = np.expand_dims(ROIs[T_sample - self.T:], 0)
#                     tempMask = np.expand_dims(visibility[T_sample - self.T:], 0)

#                 else:
#                     temp = np.expand_dims(normSkeleton[T_sample - self.T:], 0)
#                     tempImg = np.expand_dims(imageSequence[T_sample - self.T:], 0)
#                     temp_skl = np.expand_dims(skeleton[T_sample - self.T:], 0)
#                     tempHeat = np.expand_dims(heatmaps[T_sample - self.T:], 0)
#                     temRoi = np.expand_dims(ROIs[T_sample - self.T:], 0)
#                     tempMask = np.expand_dims(visibility[T_sample - self.T:], 0)

#                 inpSkeleton_all.append(temp)
#                 skeleton_all.append(temp_skl)
#                 imageSequence_input.append(tempImg)
#                 heatmap_to_use.append(tempHeat)
#                 ROIs_input.append(temRoi)
#                 visibility_input.append(tempMask)

#             inpSkeleton_all = np.concatenate((inpSkeleton_all), 0)
#             imageSequence_input = np.concatenate((imageSequence_input), 0)
#             skeleton_all = np.concatenate((skeleton_all), 0)
#             heatmap_to_use = np.concatenate((heatmap_to_use), 0)
#             ROIs_input = np.concatenate((ROIs_input), 0)
#             visibility_input = np.concatenate((visibility_input), 0)


#         skeletonData = {'normSkeleton':inpSkeleton_all, 'unNormSkeleton': skeleton_all, 'visibility':visibility_input}
#         return heatmap_to_use, imageSequence_input, skeletonData, ROIs_input

#     def __getitem__(self, index):
#         """
#         Return:
#             skeletons: FloatTensor, [T, num_joints, 2]
#             label_action: int, label for the action
#             info: dict['sample_name', 'T_sample', 'time_offset']
#         """

#         if self.phase == 'test':
#             name_sample = self.name_list[index]
#             view = self.test_view
#         else:
#             name_sample = self.name_list[index]
#             # view = self.view[index]
#         if self.sampling == 'Single':
#             heat_maps, images, skeletons, rois = self.get_data(name_sample)

#         else:
#             heat_maps, images, skeletons, rois = self.get_data_multiSeq(name_sample)

#         # label_action = self.action_list[name_sample[:3]]
#         label_action = int(name_sample[-2:]) - 1
#         dicts = {'heat': heat_maps, 'input_images': images, 'input_skeletons': skeletons,
#                  'action': label_action, 'sample_name':name_sample, 'input_rois':rois}

#         return dicts
#         # return images
    
if __name__ == '__main__':
    dataset = NTUdataset(data_root="/data/dluo/datasets/NTU-RGBD/nturgbd_skeletons/19_IRN/")


    # with open('/data/NTU-RGBD/ntu_rgb_missings_60.txt', 'r') as f:
    #     nanList = f.readlines()
    #     nanList = [line.rstrip() for line in nanList]
    # root_list = "/data/NTU-RGBD/list/"
    # DS = NTU_CrossView(root_list, nanList, dataType='2D', sampling='Multi', phase='train', cam='C002,C003', T=36, maskType='score', test_view='C001')

    # dataloader = torch.utils.data.DataLoader(DS, batch_size=2, shuffle=False,
    #                                          num_workers=1, pin_memory=True)

    # for i,sample in enumerate(dataloader):
    #     print('sample:', i, sample['sample_name'])
    #     heatmaps = sample['heat']
    #     images = sample['input_images']
    #     inp_skeleton = sample['input_skeletons']['normSkeleton']
    #     visibility = sample['input_skeletons']['visibility']
    #     label = sample['action']
    #     ROIs = sample['input_rois']

    #     print(inp_skeleton.shape)