import sys
import os
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import torch
import torch.nn as nn
from TrainHelper import train_and_evaluate

class LossHelper:
    def __init__(self, criterion_name, device):
        self.criterion = criterion_name
        self.device = device

    def set_loss_function(self):

        # Defining variables according to number of images in dataset: 416 normal cases, 120, benign cases, 561 malignant cases, 
        self.total_images = 1190
        self.num_normal = 416
        self.num_benign = 120
        self.num_malignant = 561

        # Class weights based on inverse class frequencies
        self.weight_normal_inverse = self.total_images/self.num_normal
        self.weight_benign_inverse = self.total_images/self.num_benign
        self.weight_malignant_inverse = self.total_images/self.num_malignant

        # Regular class weights
        self.weight_normal = self.num_normal/self.total_images
        self.weight_benign = self.num_benign/self.total_images
        self.weight_malignant = self.num_malignant/self.total_images

        if self.criterion == "Cross Entropy Loss":
            self.criterion = nn.CrossEntropyLoss()

        elif self.criterion == "Cross Entropy Loss Weighted":
            cross_entropy_loss_weights = torch.tensor([self.weight_benign, self.weight_malignant, self.weight_normal]).to(self.device)

            self.criterion = nn.CrossEntropyLoss(weight = cross_entropy_loss_weights)

        elif self.criterion == "Multi Margin Loss":
            self.criterion = nn.MultiMarginLoss()

        else:
            raise ValueError("Invalid loss function name. Please enter one of the following: Cross Entropy Loss, Cross Entropy Loss Weighted, Multi Margin Loss")

        return self.criterion