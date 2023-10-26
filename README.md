# A Pytorch general Medical image Segmentation Framework
The main contributors to the project are [Chen Yunqing(myself)](https://github.com/QingYunA) and [Chen Cheng](https://scholar.google.com.hk/citations?user=UIh2arMAAAAJ).

This repository is a general Medical image Segmentation Framework and highly based on [https://github.com/MontaEllis/Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation)

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
torcnrun --nproc_per_node 1 --master_port 12345 train.py --gpus 0 -o ./logs/3d-unet --conf ./conf/unet.yml
```

**torchrun args**

`nproc_per_node` : depends on your gpu nums(e.g. if you have 4 gpus in one machine, set it to `4` )

`master_port` : the port address this program used, it can set freely

**train args**


in the most of situations, you will only used these args below:
`--gpus` : specify gpu you wanna used.(e.g. if you have 4 gpus, but you only wanna use gpu 0 and gpu 3 , set it to `0,3` )

`-o` : the output directory of logs (include checkpoint file, terminal logs etc. )

`--conf` : specify the configuration you wanna use.

### Predict

run the code

```BASH
torcnrun --nproc_per_node 1 --masaer_port 12345 predict.py --gpus 0 -o ./results/3d-unet --conf ./conf/unet.yml -k ./logs/3d-unet
```

**predict args**

`-o` : the output directory of prediction results (include metrics.csv, result file, terminal logs)

`-k`: load model from this path

## Create your own configuration

create new file in path `/conf/`, file name ends with `.yml`.

For example, if you wanna use 3D Vnet, you can create `vnet.yml`, and then set all parameters you wanna set.

**how to use the parameters in `yml` file**

`conf.PARAM`: replace `PARAM` with the parameters you wanna use

### Modify train.py and predict.py to find your model

in train.py, add these codes after the last model

```Python
elif conf.network == 'NETWORK':
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
