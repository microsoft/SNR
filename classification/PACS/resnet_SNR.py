# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

__all__ = ['ResNet_SNR', 'resnet18_snr', 'resnet18_snr_nas', 'resnet18_snr_causality', 'resnet18_in']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv1x1nonLinear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1nonLinear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x


class ResNet_SNR(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_SNR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)
        self.IN3 = nn.InstanceNorm2d(256, affine=True)
        self.IN4 = nn.InstanceNorm2d(512, affine=True)

        # SE for selection:
        self.style_reid_laye1 = ChannelGate_sub(64, num_gates=64, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye2 = ChannelGate_sub(128, num_gates=128, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye3 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye4 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):

        end_points = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)  # torch.Size([64, 256, 64, 32])
        x_1_ori = x_1
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye1(x_style_1)
        x_1 = x_IN_1 + x_style_1_reid_useful
        x_1_useless = x_IN_1 + x_style_1_reid_useless


        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2_ori = x_2
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2_reid_useful, x_style_2_reid_useless, selective_weight_useful_2 = self.style_reid_laye2(x_style_2)
        x_2 = x_IN_2 + x_style_2_reid_useful
        x_2_useless = x_IN_2 + x_style_2_reid_useless

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3_ori = x_3
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3_reid_useful, x_style_3_reid_useless, selective_weight_useful_3 = self.style_reid_laye3(x_style_3)
        x_3 = x_IN_3 + x_style_3_reid_useful
        x_3_useless = x_IN_3 + x_style_3_reid_useless

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])

        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        end_points['Feature'] = x_4

        x_4 = self.fc(x_4)

        end_points['Predictions'] = F.softmax(input=x_4, dim=-1)

        return x_4, end_points


class ResNet_SNR_NAS(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_SNR_NAS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)
        self.IN3 = nn.InstanceNorm2d(256, affine=True)
        self.IN4 = nn.InstanceNorm2d(512, affine=True)

        # SE for selection:
        self.style_reid_layer1 = ChannelGate_sub(64, num_gates=64, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_layer2 = ChannelGate_sub(128, num_gates=128, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_layer3 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_layer4 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        # For NAS:
        self.gamma_layer1 = torch.nn.Parameter(torch.ones(1) * 0.5).cuda()
        self.gamma_layer2 = torch.nn.Parameter(torch.ones(1) * 0.5).cuda()
        self.gamma_layer3 = torch.nn.Parameter(torch.ones(1) * 0.5).cuda()
        self.gamma_layer4 = torch.nn.Parameter(torch.ones(1) * 0.5).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):

        end_points = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)  # torch.Size([64, 256, 64, 32])
        x_1_ori = x_1
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_layer1(x_style_1)
        x_1 = x_IN_1 + F.sigmoid(self.gamma_layer1) * x_style_1_reid_useful
        x_1_useless = x_IN_1 + x_style_1_reid_useless


        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2_ori = x_2
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2_reid_useful, x_style_2_reid_useless, selective_weight_useful_2 = self.style_reid_layer2(x_style_2)
        x_2 = x_IN_2 + F.sigmoid(self.gamma_layer2) * x_style_2_reid_useful
        x_2_useless = x_IN_2 + x_style_2_reid_useless

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3_ori = x_3
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3_reid_useful, x_style_3_reid_useless, selective_weight_useful_3 = self.style_reid_layer3(x_style_3)
        x_3 = x_IN_3 + F.sigmoid(self.gamma_layer3) * x_style_3_reid_useful
        x_3_useless = x_IN_3 + x_style_3_reid_useless

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])

        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        end_points['Feature'] = x_4

        x_4 = self.fc(x_4)

        end_points['Predictions'] = F.softmax(input=x_4, dim=-1)

        return x_4, end_points


class ResNet_SNR_Causality(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_SNR_Causality, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # FC layers for stage-1-2-3:
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(256, num_classes)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)
        self.IN3 = nn.InstanceNorm2d(256, affine=True)
        self.IN4 = nn.InstanceNorm2d(512, affine=True)

        # SE for selection:
        self.style_reid_laye1 = ChannelGate_sub(64, num_gates=64, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye2 = ChannelGate_sub(128, num_gates=128, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye3 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye4 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):

        end_points = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)  # torch.Size([64, 256, 64, 32])
        x_1_ori = x_1
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye1(x_style_1)
        x_1 = x_IN_1 + x_style_1_reid_useful
        x_1_useless = x_IN_1 + x_style_1_reid_useless


        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2_ori = x_2
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2_reid_useful, x_style_2_reid_useless, selective_weight_useful_2 = self.style_reid_laye2(x_style_2)
        x_2 = x_IN_2 + x_style_2_reid_useful
        x_2_useless = x_IN_2 + x_style_2_reid_useless

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3_ori = x_3
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3_reid_useful, x_style_3_reid_useless, selective_weight_useful_3 = self.style_reid_laye3(x_style_3)
        x_3 = x_IN_3 + x_style_3_reid_useful
        x_3_useless = x_IN_3 + x_style_3_reid_useless

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])
        x_4_featmap = x_4

        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        end_points['Feature'] = x_4

        x_4 = self.fc(x_4)

        end_points['Predictions'] = F.softmax(input=x_4, dim=-1)

        if self.training:
            return x_4, end_points, \
                   F.softmax(self.fc1(self.global_avgpool(x_IN_1).view(x_IN_1.size(0), -1))), \
                   F.softmax(self.fc1(self.global_avgpool(x_1).view(x_1.size(0), -1))), \
                   F.softmax(self.fc1(self.global_avgpool(x_1_useless).view(x_1_useless.size(0), -1))), \
                   F.softmax(self.fc2(self.global_avgpool(x_IN_2).view(x_IN_2.size(0), -1))), \
                   F.softmax(self.fc2(self.global_avgpool(x_2).view(x_2.size(0), -1))), \
                   F.softmax(self.fc2(self.global_avgpool(x_2_useless).view(x_2_useless.size(0), -1))), \
                   F.softmax(self.fc3(self.global_avgpool(x_IN_3).view(x_IN_3.size(0), -1))), \
                   F.softmax(self.fc3(self.global_avgpool(x_3).view(x_3.size(0), -1))), \
                   F.softmax(self.fc3(self.global_avgpool(x_3_useless).view(x_3_useless.size(0), -1))), \
                   self.fc3(self.global_avgpool(x_IN_3).view(x_IN_3.size(0), -1)), \
                   self.fc3(self.global_avgpool(x_3).view(x_3.size(0), -1)), \
                   self.fc3(self.global_avgpool(x_3_useless).view(x_3_useless.size(0), -1))
        else:
            return x_4, end_points, \
                   x_4_featmap, \
                   x_IN_1, x_1, x_1_useless, \
                   x_IN_2, x_2, x_2_useless, \
                   x_IN_3, x_3, x_3_useless


class ResNet_IN(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_IN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(64, affine=True)
        self.IN2 = nn.InstanceNorm2d(128, affine=True)
        self.IN3 = nn.InstanceNorm2d(256, affine=True)
        self.IN4 = nn.InstanceNorm2d(512, affine=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):

        end_points = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)  # torch.Size([64, 256, 64, 32])
        x_1 = self.IN1(x_1)

        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2 = self.IN2(x_2)

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3 = self.IN3(x_3)

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])

        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        end_points['Feature'] = x_4

        x_4 = self.fc(x_4)

        end_points['Predictions'] = F.softmax(input=x_4, dim=-1)

        if self.training:
            return x_4, end_points
        else:
            return x_4, end_points, x_3


def resnet18_snr(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SNR(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_snr_nas(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SNR_NAS(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_snr_causality(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_SNR_Causality(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18_in(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IN(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model