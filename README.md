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

# Test(Dataset Needed: NTU60-Cross View)
## 1. Coefficients from Each Person as Input [[Pretrained Model]()]
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
## 2. Concatenated Coefficients(CC) from 2 people as Input [[Pretrained Model]()]
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
## 3. CC + Reweighted-Heuristic(RH) [[Pretrained Model]()]
```bash
python test.py --bs 8 --num_workers 4 --wiL 0 --mode cls --lam_f 1.7 --wiRW 1 --wiBI 0 --wiCY 1 --wiCC 1 --wiF 1 --wiCL 0 --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_woBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_cls2/20240425_1631/65.pth --gpu_id 7
```
## 4. CC + Binary Code(BI) [[Pretrained Model]()]
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
## 5. CC + Contrastive Learning(CL) [[Pretrained Model]()]
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
## 6. CC + RH + BI [[Pretrained Model]()]
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
## 7. CC + RH + BI + CL [[Pretrained Model]()]
```bash
python test.py --cus_n wiL --bs 8 --num_workers 4 --mode cls --wiL true --lam_f 1.7 --wiRW true --wiBI true --th_g 0.5 --te_g 0.01 --wiCY true --wiCC true --wiF True --wiCL false --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_woCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_wiL/20240422_2155/65.pth --gpu_id 6
```
## 8. CC + RH + BI + CL + Limb(L)[[Pretrained Model]()]( 2147MiB GRAM )
```bash
python test.py --cus_n wiL --bs 4 --num_workers 4 --wiL 1 --mode cls --wiL 1 --lam_f 1.7 --wiRW 1 --wiBI 1 --th_g 0.5 --te_g 0.01 --wiCY 1 --wiCC 1 --wiF 1 --wiCL 1 --pret /data/dluo/work_dir/2312_CVAC_NTU-Inter/W13/4_wiL/bz4/NTU_cv_cls_wiCY_woG_wiRW_wiCC_wiF_wiBI_wiCL_T36_f1.7e+00_d1.0e-05_mse1_bi1_th5.0e-01_te1.0e-02_cls2_woD_wiL/20240426_1545/47.pth --gpu_id 7
```