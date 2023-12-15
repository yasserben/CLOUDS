## Getting Started with CLOUDS

This document provides a brief intro of the usage of CLOUDS.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Training & Evaluation in Command Line

We provide a script `train_net.py`, that is made to train all the configs provided in CLOUDS.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](./datasets/README.md).

Below is an example of how to train CLOUDS on GTA5 :

#### Warmup on GTA5 (using ConvNext-L)
```
python train_net.py --num-gpus 2 \
--config-file configs/warmup/dataset/train_gta.yaml OUTPUT_DIR /path/to/output_directory
``` 
#### Joint Training on GTA5 and generated dataset (using ConvNext-L)
```
python train_net.py --num-gpus 2 \
--config-file configs/joint_training/gta/train_jt_gta.yaml OUTPUT_DIR /path/to/output_directory
``` 

You can do the same thing for SYNTHIA and Cityscapes using ConvNext-L, ResNet-50 and ResNet-101.

#### Evaluation of the model's performance
```
python train_net.py --eval-only --config-file /path/to/config_file \
MODEL.WEIGHTS /path/to/checkpoint_file
```

For more options, see `python train_net.py -h`.