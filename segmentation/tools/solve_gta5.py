# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import ceil, floor
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch.utils.data as data
from torch.autograd import Variable

import sys
sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.loss import *
from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader, GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset

from tools.train_source import *

class UDATrainer(Trainer):
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)
        if self.args.source_dataset == 'synthia':
            source_data_set = SYNTHIA_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split=args.source_split,
                                    base_size=args.base_size,
                                    crop_size=args.crop_size,
                                    class_16=args.class_16)
        else:
            source_data_set = GTA5_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split=args.source_split,
                                    base_size=args.base_size,
                                    crop_size=args.crop_size)
        self.source_dataloader = data.DataLoader(source_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        if self.args.source_dataset == 'synthia':
            source_data_set = SYNTHIA_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split='val',
                                    base_size=args.base_size,
                                    crop_size=args.crop_size,
                                    class_16=args.class_16)
        else:
            source_data_set = GTA5_Dataset(args, 
                                    data_root_path=args.source_data_path,
                                    list_path=args.source_list_path,
                                    split='val',
                                    base_size=args.base_size,
                                    crop_size=args.crop_size)
        self.source_val_dataloader = data.DataLoader(source_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        print(self.args.source_dataset, self.args.target_dataset)
        target_data_set = City_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split=args.split,
                                base_size=args.target_base_size,
                                crop_size=args.target_crop_size,
                                class_16=args.class_16)
        self.target_dataloader = data.DataLoader(target_data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        target_data_set = City_Dataset(args, 
                                data_root_path=args.data_root_path,
                                list_path=args.list_path,
                                split='val',
                                base_size=args.target_base_size,
                                crop_size=args.target_crop_size,
                                class_16=args.class_16)
        self.target_val_dataloader = data.DataLoader(target_data_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=self.args.pin_memory,
                                            drop_last=True)
        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        if self.args.target_mode == "hard":
            self.target_loss = nn.CrossEntropyLoss(ignore_index= -1)
        elif self.args.target_mode == "entropy":
            self.target_loss = softCrossEntropy(ignore_index= -1)
        elif self.args.target_mode == "IW_entropy":
            self.target_loss = IWsoftCrossEntropy(ignore_index= -1, num_class=self.args.num_classes, ratio=self.args.IW_ratio)
        elif self.args.target_mode == "maxsquare":
            self.target_loss = MaxSquareloss(ignore_index= -1, num_class=self.args.num_classes)
        elif self.args.target_mode == "IW_maxsquare":
            self.target_loss = IW_MaxSquareloss(ignore_index= -1, num_class=self.args.num_classes, ratio=self.args.IW_ratio)

        self.causality_loss = PixelWiseCausalityLoss()

        self.current_round = self.args.init_round
        self.round_num = self.args.round_num
        
        self.target_loss.to(self.device)
        self.causality_loss.to(self.device)

        self.target_hard_loss = nn.CrossEntropyLoss(ignore_index= -1)

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)
        
        if not self.args.continue_training:
            self.best_MIou = 0
            self.best_iter = 0
            self.current_iter = 0
            self.current_epoch = 0

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))
        
        self.args.iter_max = self.dataloader.num_iterations*self.args.epoch_each_round*self.round_num
        print(self.args.iter_max, self.dataloader.num_iterations)

        # train
        #self.validate() # check image summary
        #self.validate_source()
        self.train_round()

        self.writer.close()
    
    def train_round(self):
        for r in range(self.current_round, self.round_num):
            print("\n############## Begin {}/{} Round! #################\n".format(self.current_round+1, self.round_num))
            print("epoch_each_round:", self.args.epoch_each_round)
            
            self.epoch_num = (self.current_round+1)*self.args.epoch_each_round

            # generate threshold
            self.threshold = self.args.threshold

            self.train()

            self.current_round += 1
        
    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch+1, self.epoch_num))
        self.logger.info("Training one epoch...")
        self.Eval.reset()
        
        # Initialize your average meters
        loss_seg_value = 0
        loss_target_value = 0
        loss_seg_value_2 = 0
        loss_target_value_2 = 0
        iter_num = self.dataloader.num_iterations

        # Set the model to be in training mode (for batchnorm and dropout)
        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()

        batch_idx = 0
        for batch_s, batch_t in tqdm_epoch:
            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            ##########################
            # source supervised loss #
            ##########################
            # train with source
            x, y, _ = batch_s
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]

            y = torch.squeeze(y, 1)
            loss = self.loss(pred, y)

            loss_ = loss
            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                loss_ += loss_2
                loss_seg_value_2 += loss_2.cpu().item() / iter_num

            loss_.backward()
            loss_seg_value += loss.cpu().item() / iter_num
            
            ###############
            # target loss #
            ###############
            # train with target
            x, _, _ = batch_t
            if self.cuda:
                x = Variable(x).to(self.device)

            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]
                pred_P_2 = F.softmax(pred_2, dim=1)
            pred_P = F.softmax(pred, dim=1)

            if self.args.target_mode == "hard":
                label = torch.argmax(pred_P.detach(), dim=1)
                if self.args.multi: label_2 = torch.argmax(pred_P_2.detach(), dim=1)
            else:
                label = pred_P
                if self.args.multi: label_2 = pred_P_2

            maxpred, argpred = torch.max(pred_P.detach(), dim=1)
            if self.args.multi: maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)

            if self.args.target_mode == "hard":
                mask = (maxpred > self.threshold)
                label = torch.where(mask, label, torch.ones(1).to(self.device, dtype=torch.long)*self.ignore_index)
            
            loss_target = self.args.lambda_target*self.target_loss(pred, label)

            loss_target_ = loss_target

            ######################################
            # Multi-level Self-produced Guidance #
            ######################################
            if self.args.multi:
                pred_c = (pred_P+pred_P_2)/2
                maxpred_c, argpred_c = torch.max(pred_c, dim=1)
                mask = (maxpred > self.threshold) | (maxpred_2 > self.threshold)

                label_2 = torch.where(mask, argpred_c, torch.ones(1).to(self.device, dtype=torch.long)*self.ignore_index)
                loss_target_2 = self.args.lambda_seg * self.args.lambda_target*self.target_hard_loss(pred_2, label_2)
                loss_target_ += loss_target_2
                loss_target_value_2 += loss_target_2 / iter_num

            loss_target_.backward()
            loss_target_value += loss_target / iter_num

            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_idx % 400 == 0:
                if self.args.multi:
                    self.logger.info("epoch{}-batch-{}:loss_seg={:.3f}-loss_target={:.3f}; loss_seg_2={:.3f}-loss_target_2={:.3f}; mask={:.3f}".format(self.current_epoch,
                                                                           batch_idx, loss.item(), loss_target.item(), loss_2.item(), loss_target_2.item(), mask.float().mean().item()))
                else:
                    self.logger.info("epoch{}-batch-{}:loss_seg={:.3f}-loss_target={:.3f}".format(self.current_epoch,
                                                                           batch_idx, loss.item(), loss_target.item()))
            batch_idx += 1

            self.current_iter += 1
        
        self.writer.add_scalar('train_loss', loss_seg_value, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, loss_seg_value))
        self.writer.add_scalar('target_loss', loss_target_value, self.current_epoch)
        tqdm.write("The average target_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_target_value))
        if self.args.multi:
            self.writer.add_scalar('train_loss_2', loss_seg_value_2, self.current_epoch)
            tqdm.write("The average loss_2 of train epoch-{}-:{}".format(self.current_epoch, loss_seg_value_2))
            self.writer.add_scalar('target_loss_2', loss_target_value_2, self.current_epoch)
            tqdm.write("The average target_loss_2 of train epoch-{}-:{:.3f}".format(self.current_epoch, loss_target_value_2))
        tqdm_epoch.close()
        
        #eval on source domain
        self.validate_source()

def add_UDA_train_args(arg_parser):
    arg_parser.add_argument('--source_dataset', default='gta5', type=str,
                            choices=['gta5', 'synthia'],
                            help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str,
                            help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0, 
                            help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=1,
                            help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2,
                            help="epoch each round")                 
    arg_parser.add_argument('--target_mode', type=str, default="maxsquare",
                            choices=['maxsquare', 'IW_maxsquare', 'entropy', 'IW_entropy', 'hard'],
                            help="the loss function on target domain")
    arg_parser.add_argument('--lambda_target', type=float, default=1,
                            help="lambda of target loss")
    arg_parser.add_argument('--gamma', type=float, default=0, 
                            help='parameter for scaled entorpy')
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2, 
                            help='the ratio of image-wise weighting factor')
    arg_parser.add_argument('--threshold', type=float, default=0.95,
                            help="threshold for Self-produced guidance")
    return arg_parser

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset)+"2"+str(args.target_dataset)+"_"+args.target_mode

    agent = UDATrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()