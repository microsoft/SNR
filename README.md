# Introduction
This repository holds the codes and methods for the following paper:
> Style Normalization and Restitution for Domain Generalization and Adaptation

For many practical computer vision applications, the learned models usually have high performance on the datasets used for training but suffer from significant performance degradation when deployed in new environments, where there are usually style differences between the training images and the testing images. An effective domain generalizable model is expected to be able to learn feature representations that are both generalizable and discriminative. 

In this work, we design a novel Style Normalization and Restitution module (SNR) to simultaneously ensure both high generalization and discrimination capability of the networks. In the SNR module, particularly, we filter out the style variations (e.g, illumination, color contrast) by performing Instance Normalization (IN) to obtain style normalized features, where the discrepancy among different samples and domains is reduced. However, such a process is task-ignorant and inevitably removes some task-relevant discriminative information, which could hurt the performance. To remedy this, we propose to distill task-relevant discriminative features from the residual (i.e, the difference between the original feature and the style normalized feature) and add them back to the network to ensure high discrimination. Moreover, for better disentanglement, we enforce a dual causality loss constraint in the restitution step to encourage the better separation of task-relevant and task-irrelevant features. 

<p align="center">
  <img src="imgs/pipeline.png" alt="pipeline" width="800">
</p>



We validate the effectiveness of our SNR on different computer vision tasks, including classification, semantic segmentation, and object detection. Please refer to the sub-folder of each task for more details.

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
