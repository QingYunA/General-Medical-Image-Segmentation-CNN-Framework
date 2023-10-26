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

这里我们以一个示例（训练 3D Unet）来教你如何使用该存储库

```BASH
torcnrun --nproc_per_node 1 --master_port 12345 train.py --gpus 0 -o ./logs/3d-unet --conf ./conf/unet.yml
```

**torchrun 参数**

`nproc_per_node` ：取决于您的 GPU 数量（例如，如果一台机器上有 4 个 GPU，请将其设置为 `4` ）

`master_port` ：该程序使用的端口地址，可以自由设置

**train 参数**

在大多数情况下，您只会使用以下参数：

`--gpus` ：指定要使用的 GPU（例如，如果您有 4 个 GPU，但只想使用 GPU 0 和 GPU 3，请将其设置为 `0,3` ）

`-o` ：日志的输出目录（包括checkpoint文件、终端日志等）

`--conf` ：指定要使用的配置。

### 预测

运行以下代码

```BASH
torcnrun --nproc_per_node 1 --masaer_port 12345 predict.py --gpus 0 -o ./results/3d-unet --conf ./conf/unet.yml -k ./logs/3d-unet
```

**predict 参数**

`-o` ：预测结果的输出目录（包括 metrics.csv、结果文件、终端日志）

`-k` ：从该路径加载模型

## 创建自己的配置

在路径 `/conf/` 中创建一个新文件，文件名以 `.yml` 结尾。

例如，如果您想使用 3D Vnet，可以创建 `vnet.yml` ，然后设置您想要设置的所有参数。

**如何使用 `yml` 文件中的参数**

`conf.PARAM` ：用您想要使用的参数替换 `PARAM`

### 修改 train.py 和 predict.py 来找到您的模型

在 train.py 中，在最后一个模型之后添加以下代码

```Python
elif conf.network == 'NETWORK':
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
