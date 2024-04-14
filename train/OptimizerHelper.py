import sys
import os
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import torch.optim as optim

class OptimizerHelper:
    def __init__(self, model, optimizer_name, learning_rate):
        self.model = model
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate

    def set_optimizer(self):
        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = 1e-4, betas = (0.9, 0.999), eps = 1e-8, amsgrad = True)

        elif self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = 0.9, weight_decay = 1e-4)

        elif self.optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr = self.learning_rate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01, amsgrad = False)
        
        elif self.optimizer_name == "Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr = self.learning_rate, lr_decay = 0, weight_decay = 0, initial_accumulator_value = 0, eps = 1e-10)

        elif self.optimizer_name == "RMSProp":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.learning_rate, alpha = 0.99, eps = 1e-8, weight_decay = 1e-4, momentum = 0, centered = False)

        else:
            raise ValueError("Invalid optimizer name. Please enter one of the following: Adam, SGD, AdamW, Adagrad, RMSprop")

        return self.optimizer




