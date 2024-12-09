# 一个 PyTorch 通用的医学图像分割框架

该项目的主要贡献者是 [陈云青](https://github.com/QingYunA) 和 [陈诚](https://scholar.google.com.hk/citations?user=UIh2arMAAAAJ)。

该项目是一个通用的医学图像分割框架，主要基于 [https://github.com/MontaEllis/Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation)。

---

## 已完成的模型

**如果以下模型对您有帮助，请引用相应的参考文献。**

| 发表日期 |     模型名称     |                 第一和最后作者                 |                                                         标题                                                         |                                                                      参考文献                                                                      |
| :------: | :---------------: | :--------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| 2016-10 |     3D U-Net     |      Özgün Çiçek 和 Ronneberger, Olaf      |                        3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation                        |                                     [MICCAI2016](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)                                     |
| 2016-10 |     3D V-Net     |     Fausto Milletari 和 Seyed-Ahmad Ahmadi     |                 V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation                 |                                           [3DV2016](https://ieeexplore.ieee.org/abstract/document/7785132)                                           |
| 2017-09 | 3D DenseVoxelNet |     Lequan Yu 和 Jing Qin & Pheng-Ann Heng     |                Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets                |                                     [MICCAI2017](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_33)                                     |
| 2017-09 |    3D DenseNet    |          Toan Duc Bui 和 Taesup Moon          |                             3D Densely Convolutional Networks for Volumetric Segmentation                             |                                                       [arxiv](https://arxiv.org/abs/1709.03199)                                                       |
| 2017-05 |   3D HighResNet   | Wenqi Li 和 M. Jorge Cardoso & Tom Vercauteren | On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task |                                      [IPMI2017](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28)                                      |
| 2017-05 | 3D Residual U-Net |        Kisuk Lee 和 H. Sebastian Seung        |                               Superhuman Accuracy on the SNEMI3D Connectomics Challenge                               |                                                       [arxiv](https://arxiv.org/abs/1706.00120)                                                       |
| 2021-10 |      CSR-Net      |           Cheng Chen 和 Ruoxiu Xiao           |               CSR-Net: Cross-Scale Residual Network for multi-objective scaphoid fracture segmentation               |                                    [CIBM2021](https://www.sciencedirect.com/science/article/pii/S0010482521005709)                                    |
|   2022   |       UNETR       |         Ali Hatamizadeh 和 Daguang Xu         |                                 UNETR: Transformers for 3D Medical Image Segmentation                                 | [CVPR2022](https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html) |
|   2023   |        IS        |              陈程 和 Ruoxiu Xiao              |           Integration- and separation-aware adversarial model for cerebrovascular segmentation from TOF-MRA           |                                    [CMPB](https://www.sciencedirect.com/science/article/abs/pii/S0169260723001414)                                    |

## 使用方法

### 要求

推荐的 Python 和包版本：

* Python 3.10.0
* PyTorch 1.13.1（不要使用 torch2.0，否则会导致程序无法启动）

### 训练

这里我们使用一个示例（训练3D Unet）来教您如何使用这个存储库。

```BASH
python train.py config=unet
```
要指定当前保存的文件夹名称，您可以使用 config.XXX=XXX 修改相应的参数。
```BASH
python train.py config=unet config.name=unet-1
```
模型训练过程中的所有文件将保存在 ./logs/unet-1 中。

所有参数都可以在 `conf/unet.yaml` 中进行设置。
#### 全局配置
考虑到许多设置对所有配置文件都是通用的，例如 `data_path`、`num_epochs` 等，为了避免重复工作，我们将这些通用参数放在了 `conf/config.yaml` 中。所有配置文件都将从 `config.yaml` 中继承这些属性。

以 `num_workers`（在 `config.yaml` 中默认为 18）为例，参数覆盖的优先级如下：
命令行参数 `config.num_workers=20` > `data_3d.yaml` 中的 `num_workers=18` > `config.yaml` 中的默认值 `num_workers=18`。

#### 文件结构
训练日志将按照以下方式保存：
```
./logs/ying_tof (对应保存的文件夹: ./logs/config.name)
└── 2023-11-24 (日期: 年-月-日)
    └── 17-05-02 (时间: 时-分-秒)
        ├── .hydra (配置保存文件)
        │   ├── config.yaml (当前运行的配置)
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── train_3d.log (训练期间的日志)
        └── train_tensorboard (Tensorboard 文件夹)
            └── events.out.tfevents.1700816703.medai.115245.0 (Tensorboard 文件)
```
### 预测

运行以下代码

```BASH
python predict.py config=unet config.ckpt=XXX
```
`警告`：ckpt 必须是模型的绝对路径，而不是相对路径。
```
./results/ (结果的根文件夹)
└── unet (模型名称: unet)
    └── 2023-12-04 (日期: 年-月-日)
        └── 17-39-30 (时间: 时-分-秒)
            ├── metrics.csv (包含指标的 CSV 文件)
            ├── pred_file (预测文件夹)
            │   ├── pred-0000.mhd (MHD 格式的预测文件 0)
            │   ├── pred-0000.zraw (ZRAW 格式的预测文件 0)
            │   ├── pred-0001.mhd (MHD 格式的预测文件 1)
            │   ├── pred-0001.zraw (ZRAW 格式的预测文件 1)
            │   ├── pred-0002.mhd (MHD 格式的预测文件 2)
            │   └── ... (其他预测文件)
            └── predict.log (预测的日志文件)
```

## 创建自己的配置

在路径 `/conf/` 中创建一个新文件，文件名以 `.yaml` 结尾。

例如，如果您想使用 3D Vnet，可以创建 `vnet.yaml` ，然后设置您想要设置的所有参数。

**如何使用 `yaml` 文件中的参数**

`config.PARAM` ：用您想要使用的参数替换 `PARAM`

### 修改 train.py 和 predict.py 来找到您的模型

在 train.py 中，在最后一个模型之后添加以下代码

```Python
elif config.network == 'NETWORK':
    from models.three_d.NETWORK import NETWORK
    model = NETWORK()
```

`NETWORK` ：表示您想要使用的网络
![](https://s2.loli.net/2023/10/26/LEQt8p7TufXxqyb.png)

## 相关工作

如果这段代码对您有帮助，请为我们引用以下文献。谢谢！

```
 C Chen 和 R Xiao. Generative consistency for semi-supervised cerebrovascular segmentation from TOF-MRA, 2022, IEEE Transactions on Medical Imaging 42 (2), 346-353
 C Chen 和 R Xiao. All answers are in the images: A review of deep learning for cerebrovascular segmentation, 2023, Computerized Medical Imaging and Graphics, 102229
 C Chen 和 J Zhao. Understanding the brain with attention: A survey of transformers in brain sciences, 2023, Brain-X, 1 (3): e29.
```
