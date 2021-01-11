# SNR-Object detection 

This is the repo for reproducing the results for domain generalization and unsupervised domain adaptation of object detection task in the paper 'Style Normalization and Restitution for Domain Generalization and Adaptation'. We use the codebase from 'Domain Adaptive Faster R-CNN for Object Detection in the Wild' (https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch/blob/master/LICENSE_maskrcnn_benchmark).


## Installation

Please follow the instruction in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) to install and use Domain-Adaptive-Faster-RCNN-PyTorch.

## Datasets

Please refer to the semantic segmentation sub-section for more details.


## Example Usage
(Optional) If pretrain the model only on the source domain, this is also our SNR shceme on the domain generalization setting, you can change the backbone from DA Faster R-CNN to the naive Faster R-CNN. 

(Note) Please refer to the scheme of 'SNR-DA Faster R-CNN (ours)' in Section 4.3- of our arxiv paper SNR-extension.
An example of Domain Adaptive Faster R-CNN with FPN adapting from **Cityscapes** dataset to **Foggy Cityscapes** dataset is provided:
1. Follow the example in [Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) to download dataset and generate coco style annoation files
2. Symlink the path to the Cityscapes and Foggy Cityscapes dataset to `datasets/` as follows:
    ```bash
    # symlink the dataset
    cd ~/github/Domain-Adaptive-Faster-RCNN-PyTorch
    ln -s /<path_to_cityscapes_dataset>/ datasets/cityscapes
    ln -s /<path_to_foggy_cityscapes_dataset>/ datasets/foggy_cityscapes
    ```
3. Train the SNR module embedded Domain Adaptive Faster R-CNN:
    ```
    python tools/train_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_SNR_cityscapes_to_foggy_cityscapes.yaml"
    ```
4. Test the trained model:
    ```
    python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_SNR_cityscapes_to_foggy_cityscapes.yaml" MODEL.WEIGHT <path_to_store_weight>/model_final.pth
    ```
### Citation

If you use this code in your research, please cite:

```
@article{jin2021style,
  title={Style Normalization and Restitution for DomainGeneralization and Adaptation},
  author={Jin, Xin and Lan, Cuiling and Zeng, Wenjun and Chen, Zhibo},
  journal={arXiv preprint arXiv:2101.00588},
  year={2021}
}
```