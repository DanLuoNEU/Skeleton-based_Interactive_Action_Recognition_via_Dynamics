# Environment Set Up
```
pip install -r requirements.txt
```

# Train
Scripts for different experiments tra with HyperParameters
## I. Original 
### 1. Train the learnable Dictionary for reconstruction
```bash
python main.py --bs 8 --num_workers 4 --mode D --lam_f 0.1 --wiRW False --wiBI False --wiCY true --wiCC true --wiF True --wiCL False --ep_D 20 --ms 10,15 --save_m true --gpu_id 7
``` 
### 2. Freeze the Dictionary Learning
```bash
python main.py --bs 8 --num_workers 4 --mode cls --lam_f 0.1 --wiRW False --wiBI False --wiCY true --wiCC true --wiF True --wiCL False --wiD /path/to/best_loss_mse.pth --ep_D 80 --ms 30,50 --save_m true --gpu_id 7
```
## II. wiRW and wiBI
### 1. Train the learnable Dictionary for reconstruction
```bash
python main.py --bs 8 --num_workers 4 --mode D --lam_f 1.7 --wiRW True --wiBI True --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL False --ep_D 25 --ms 15,20 --save_m true --gpu_id 7
``` 
### 2. Freeze the Dictionary Learning
```bash
python main.py --bs 8 --num_workers 4 --mode cls --lam_f 1.7 --wiRW True --wiBI True --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL False --wiD /path/to/best_loss_mse.pth --ep_D 80 --ms 30,50 --save_m true --gpu_id 7
```

## III. Contrastive Learning
### 1. Train Classification Part with Contrastive Learning Loss only
```bash
python main.py --bs 8 --num_workers 4 --mode D --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL true --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2/20240415_1632/95.pth --ep_D 25 --ms 15,20 --save_m true --gpu_id 7
``` 

### 2. DownStreaming Task
```bash
python main.py --bs 8 --num_workers 4 --mode cls --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL true --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_wiCY_woG_wiRW_wiCC_wiF_wiBI_wiCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_woD/20240417_234930/22.pth --ep 25 --ms 15,20 --save_m true --ep_save 1 --gpu_id 7
```
## IV. Contrastive Learning with Limb modality
### 1. Train Classification Part with Additional Modality
```bash
python main.py --cus_n wiL --bs 8 --num_workers 4 --mode D --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL False --ep_D 30 --ms 3,6 --save_m true --gpu_id 7
```

### 2. DownStreaming Task
```bash
python main.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --wiD /path/to/best_loss_mse.pth --ep 100 --ms 30,50 --save_m true --gpu_id 7
```
### 3. Train Classification Part with Contrastive Learning Loss only
```bash
python main.py --cus_n wiL --bs 4 --num_workers 4 --mode D --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL true --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --ep_D 25 --ms 15,20 --save_m true --gpu_id 7
```

### 4. Classification
```bash
python main.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL true --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_D_wiCY_woG_wiRW_wiCC_wiF_wiBI_wiCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_woD_wiL/20240423_144208/23.pth --ep 25 --ms 15,20 --save_m true --ep_save 1 --gpu_id 7
```

# Test
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
