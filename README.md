# Medical-Template

This is a template for medical image segmentation by Serein.
***

## Usage

### Set Data Path in conf.yml

all paramters are in `conf.yml`

### Train

```shell
torchrun --nproc_per_node 1 --master_port 12365 train.py --gpus 1 -o ./logs/test_conf/ --conf_path ./conf/conf.yml --use_scheduler
```

`-o` : logs will output in this path
`--network` : choose which network
`--use_sheduler` : determine whether to use scheduler
`--patch_size` : patch size of input image
