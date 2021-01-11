# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from graphs.models.deeplab_multi import DeeplabMulti, DeeplabMulti_SNR

def get_model(args):
    if args.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    elif args.backbone == "deeplabv2_multi_snr":
        model = DeeplabMulti_SNR(num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    return model, params