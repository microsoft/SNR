# SNR-DG
This is the repo for reproducing the results for domain generalization of classification task in the paper 'Style Normalization and Restitution for Domain Generalization and Adaptation'. We use Epi-FCR (https://github.com/HAHA-DL/Episodic-DG) as our code framework to validate the effectiveness of SNR for PACS dataset.
## Datasets preparation
Please download the data from https://drive.google.com/open?id=0B6x7gtvErXgfUU1WcGY5SzdwZVk and use the official train/val split.
### ImageNet pretrained model
We use the pytorch pretrained ResNet-18 model from https://download.pytorch.org/models/resnet18-5c106cde.pth

## Enviroments

> pytorch 1.0.0 \
> Python 3.7.3 \
> Ubuntu 16.04.6

## Run

> Baseline: 

sh run_main_agg.sh #data_folder #model_path 

> Our SNR with the proposed dual causality loss: 

sh run_main_agg_snr_causality.sh #data_folder #model_path 


### Correspondence with the paper

Please refer to the Section 4.1 of our arxiv paper SNR-extension. 


### Reference
If you consider using this code or its derivatives, please consider citing:

```
@article{jin2021style,
  title={Style Normalization and Restitution for DomainGeneralization and Adaptation},
  author={Jin, Xin and Lan, Cuiling and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2101.00588},
  year={2021}
}
```

### Note

When working with a different enviroment, you can get different results and need to tune the hyper parameters yourself.

