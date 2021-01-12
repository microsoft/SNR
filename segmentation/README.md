# SNR-Semantic Segmentation

This is the repo for reproducing the results for domain generalization, and unsupervised domain adaptation of semantic segmentation task, in the paper 'Style Normalization and Restitution for Domain Generalization and Adaptation'. We use the codebase from 'Domain Adaptation for Semantic Segmentation with Maximum Squares Loss' (https://github.com/ZJULearning/MaxSquareLoss).


## Requirements
The code is implemented with Python(3.6) and Pytorch(1.0.0).

Install the newest Pytorch from https://pytorch.org/.

To install the required python packages, run

```python
pip install -r requirements.txt
```

## Setup

#### GTA5-to-Cityscapes:

- Download [**GTA5 datasets**](https://download.visinf.tu-darmstadt.de/data/from_games/), which contains 24,966 annotated images with 1914×1052 resolution taken from the GTA5 game. We use the sample code for reading the label maps and a split into training/validation/test set from [here](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip). In the experiments, we resize GTA5 images to 1280x720.
- Download [**Cityscapes**](https://www.cityscapes-dataset.com/), which contains 5,000 annotated images with 2048 × 1024 resolution taken from real urban street scenes. We resize Cityscapes images to 1024x512 (or 1280x640 which yields sightly better results but consumes more time). 
- Download the **[checkpoint](https://drive.google.com/open?id=1KP37cQo_9NEBczm7pvq_zEmmosdhxvlF)** pretrained on GTA5.
- If you want to pretrain the model by yourself, download [**the model**](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth) pretrained on ImageNet.

#### SYNTHIA-to-Cityscapes:

- Download [**SYNTHIA-RAND-CITYSCAPES**](http://synthia-dataset.net/download/808/) consisting of 9,400 1280 × 760 synthetic images. We resize the images to 1280x760.
- Download the [**checkpoint**](https://drive.google.com/open?id=1wLffQRljXK1xoqRY64INvb2lk2ur5fEL) pretrained on SYNTHIA.

#### Cityscapes-to-CrossCity

- Download [**NTHU dataset**](https://yihsinchen.github.io/segmentation_adaptation_dataset/), which consists of images with 2048 × 1024 resolution from four different cities: Rio, Rome, Tokyo, and Taipei. We resize the images to 1024x512, which are the same as Cityscapes.
- Download the **[checkpoint](https://drive.google.com/open?id=1QMpj7sPqsVwYldedZf8A5S2pT-4oENEn)** pretrained on Cityscapes.

Put all datasets into "datasets" folder and all checkpoints into "pretrained_model" folder.


## Training

### GTA5-to-Cityscapes:

Domain Generalization Setting: Pretrain the model on the source domain (GTA5), this is also our SNR shceme on the domain generalization setting. 

Otherwise, download the [checkpoint](https://drive.google.com/open?id=1KP37cQo_9NEBczm7pvq_zEmmosdhxvlF) pretrained on GTA5 in "Setup" section.

```
python3 tools/train_source.py --gpu "0" --dataset 'gta5' --checkpoint_dir "./log/gta5_pretrain/" --iter_max 200000 --iter_stop 80000 --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --crop_size "1280,720"
```

Unsupervised Domain Adaptation Setting: Please refer to the scheme of SNR-MS (ours) in Section 4.2- of our paper.

Then in the next adaptation step, set `--pretrained_ckpt_file "./log/gta5_pretrain/gta5final.pth"`.

- SNR integrated backbone:


```
python3 tools/solve_gta5.py --gpu "0" --backbone "deeplabv2_multi_snr" --dataset 'cityscapes' --checkpoint_dir "./log/YOUR_SAVED_PATH/" --pretrained_ckpt_file "./pretrained_model/GTA5_source.pth" --round_num 5 --target_mode "maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --lambda_target 0.1
```


Eval:

```
python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./log/YOUR_SAVED_PATH" --image_summary True --flip True
```

To have a look at predicted examples, run tensorboard as follows:

```
tensorboard --logdir=./log/eval_city  --port=6009
```

### Citation

If you use the code in your research, please cite:

```
@article{jin2021style,
  title={Style Normalization and Restitution for DomainGeneralization and Adaptation},
  author={Jin, Xin and Lan, Cuiling and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2101.00588},
  year={2021}
}
```
