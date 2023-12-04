# A Pytorch general Medical image Segmentation Framework
The main contributors to the project are [Chen Yunqing(myself)](https://github.com/QingYunA) and [Chen Cheng](https://scholar.google.com.hk/citations?user=UIh2arMAAAAJ).

This repository is a general Medical image Segmentation Framework and highly based on [https://github.com/MontaEllis/Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation)

中文说明: [ZH_README](https://github.com/QingYunA/General-Medical-Image-Segmentation-CNN-Framework/blob/main/ZH_README.md)

***
## The model have completed
**If the following model has been helpful to you, we hope you can cite the corresponding reference.**
| Publication Date | Model Name | The First and Last Authors |  Title | Reference|
| :---: | :---: | :---: | :---: | :---: |
| 2016-10 |  3D U-Net  | Özgün Çiçek and Ronneberger, Olaf | 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation | [MICCAI2016](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)|
| 2016-10 | 3D V-Net | Fausto Milletari and Seyed-Ahmad Ahmadi | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | [3DV2016](https://ieeexplore.ieee.org/abstract/document/7785132) |
| 2017-09 | 3D DenseVoxelNet  | Lequan Yu and Jing Qin & Pheng-Ann Heng | Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets | [MICCAI2017](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_33) |
| 2017-09 | 3D DenseNet | Toan Duc Bui and Taesup Moon | 3D Densely Convolutional Networks for Volumetric Segmentation | [arxiv](https://arxiv.org/abs/1709.03199) |
| 2017-05 | 3D HighResNet | Wenqi Li and M. Jorge Cardoso & Tom Vercauteren | On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task | [IPMI2017](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28) |
| 2017-05 | 3D Residual U-Net | Kisuk Lee and H. Sebastian Seung | Superhuman Accuracy on the SNEMI3D Connectomics Challenge | [arxiv](https://arxiv.org/abs/1706.00120) |
| 2021-10 |  CSR-Net   | Cheng Chen and Ruoxiu Xiao | CSR-Net: Cross-Scale Residual Network for multi-objective scaphoid fracture segmentation | [CIBM2021](https://www.sciencedirect.com/science/article/pii/S0010482521005709) |
| 2022 | UNETR | Ali Hatamizadeh and Daguang Xu | UNETR: Transformers for 3D Medical Image Segmentation | [CVPR2022](https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html) |
| 2023 | IS | Chen Cheng and Ruoxiu Xiao | Integration- and separation-aware adversarial model for cerebrovascular segmentation from TOF-MRA | [CMPB](https://www.sciencedirect.com/science/article/abs/pii/S0169260723001414)|

## Usage

### Requirements

The recommend python and package version:

* python 3.10.0
* pytorch 1.13.1 (do not use torch2.0, it will cause program failed to start)

### Train

here we use an example(Traning 3D Unet) to teach you how use this repository

```BASH
python train.py config=unet
```
To specify the folder name for the current save, you can modify the corresponding parameter using config.XXX=XXX.
```BASH
python train.py config=unet config.name=unet-1
```
All files during the model training process will be saved in ./logs/unet-1

all parameter can be setted in `conf/unet.yaml`
#### Global configuration
Considering that many settings are common to all configuration files, such as `data_path`, `num_epochs`, etc., to avoid repetitive work, we have placed these common parameters in `conf/config.yaml`. All configuration files will have these properties from `config.yaml`.

Taking `num_workers` (defaulted to 18 in `config.yaml`) as an example, the priority of parameter overriding is as follows:
Command line argument `config.num_workers=20` > `num_workers=18` in `data_3d.yaml` > Default value `num_workers=18` in `config.yaml`.

#### File Structure
Traning logs will be saved like this:
```
./logs/ying_tof (Corresponding saved folder: ./logs/config.name)
└── 2023-11-24 (Date: year-month-day)
    └── 17-05-02 (Time: hour-minute-second)
        ├── .hydra (Configuration save files)
        │   ├── config.yaml (Configuration for this running)
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── train_3d.log (Log during training)
        └── train_tensorboard (Tensorboard folder)
            └── events.out.tfevents.1700816703.medai.115245.0 (file for Tensorboard)
```
### Predict

run the code

```BASH
python predict.py config=unet config.ckpt=XXX
```
`WARNING`: ckpt must be the absolute path of the model, not the relative path
```
./results/ (Root folder for results)
└── unet (Model name: unet)
    └── 2023-12-04 (Date: year-month-day)
        └── 17-39-30 (Time: hour-minute-second)
            ├── metrics.csv (CSV file containing metrics)
            ├── pred_file (Folder for prediction files)
            │   ├── pred-0000.mhd (Prediction file 0 in MHD format)
            │   ├── pred-0000.zraw (Prediction file 0 in ZRAW format)
            │   ├── pred-0001.mhd (Prediction file 1 in MHD format)
            │   ├── pred-0001.zraw (Prediction file 1 in ZRAW format)
            │   ├── pred-0002.mhd (Prediction file 2 in MHD format)
            │   └── ... (Additional prediction files)
            └── predict.log (Log file for prediction)
```


## Create your own configuration

create new file in path `/conf/config`, file name ends with `.yaml`.

For example, if you wanna use 3D Vnet, you can create `vnet.yaml`, and then set all parameters you wanna set.

### how to use the parameters in `yaml` file

`config.PARAM`: replace `PARAM` with the parameters you wanna use

### Modify train.py and predict.py to find your model

in train.py, add these codes after the last model

```Python
elif config.network == 'NETWORK':
    from models.three_d.NETWORK import NETWORK
    model = NETWORK()
```

`NETWORK`: means the network you wanna use
![](https://s2.loli.net/2023/10/26/LEQt8p7TufXxqyb.png)

## Related Works
if this code is helpful for you, you can cite these for us. Thanks！
```
 C Chen and R Xiao. Generative consistency for semi-supervised cerebrovascular segmentation from TOF-MRA, 2022, IEEE Transactions on Medical Imaging 42 (2), 346-353
 C Chen and R Xiao. All answers are in the images: A review of deep learning for cerebrovascular segmentation, 2023, Computerized Medical Imaging and Graphics, 102229
 C Chen and J Zhao. Understanding the brain with attention: A survey of transformers in brain sciences, 2023, Brain-X, 1 (3): e29.
```
