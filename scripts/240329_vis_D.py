import os
import sys
import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt

import torch

def sortKey(s):
    return int(s.split('/')[-1].split('.')[0])

def main(args):
    
    name_gif = args[1]
    dir_pth = args[2]
    # dir_pth = "/data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_woD_wiCY_woG_woRW_wiCC_wiF_woBI_woCL_T36_f1.0e-01_d1e-05_2_1_1_0.100_ori/20240329_004205"
    print("Generating Dictionary Poles change visualizaiton:", name_gif+'.gif')
    print("Loading models from:", dir_pth)
    list_name = glob.glob(dir_pth+"/*.pth")
    list_name.sort(key=sortKey)

    frames = []
    r_0 = 0
    theta_0 = 0
    for i, path_pth in enumerate(list_name):
        name_pth = path_pth.split('/')[-1]
        state_dict = torch.load(path_pth, map_location='cpu')['state_dict']
        r = state_dict['rr'].numpy()
        theta = state_dict['theta'].numpy()
        ax = plt.subplot(1, 1, 1, projection='polar')
        if name_pth == '0.pth':
            r_0 = r
            theta_0 = theta
            
            ax.scatter(0, 1, c='red', s = 5)
        else:
            ax.scatter(0, 1, c='green', s = 5)
            ax.scatter( theta, r, c='green', s = 5)
            ax.scatter(-theta, r, c='green', s = 5)
        if i == len(list_name)-1:
            ax.set_rmax(1.2)
            ax.set_title(f"Dictionary @Epoch {name_pth.split('.')[0]}", va='top')
            plt.draw()
            plt.savefig(os.path.join(dir_pth, f'Dict_epoch{i}.png'))
        ax.scatter( theta_0, r_0, c='red', s = 5)
        ax.scatter(-theta_0, r_0, c='red', s = 5)
        ax.set_rmax(1.2)
        ax.set_title(f"Dictionary @Epoch {name_pth.split('.')[0]}", va='top')
        plt.draw()
        if (i == 0):
            plt.savefig(os.path.join(dir_pth, f'Dict_epoch{i}.png'))

        frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()
    
    # Save frames as .gif using imageio
    imageio.mimsave(os.path.join(dir_pth, name_gif+'.gif'),
                    frames, fps=3)
    print(name_gif+'.gif','saved under:', dir_pth)

if __name__ == '__main__':
    main(sys.argv)