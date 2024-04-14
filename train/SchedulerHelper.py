import sys
import os
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import torch.optim as optim

class SchedulerHelper:
    def __init__(self, optimizer, scheduler_name):
        self.optimizer = optimizer
        self.scheduler_name = scheduler_name

    def set_scheduler(self):
        if self.scheduler_name == "Step LR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 5, gamma = 0.1)

        elif self.scheduler_name == "Exponential LR":
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        elif self.scheduler_name == "Cosine Annealing LR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 10)

        elif self.scheduler_name == "Reduce LR on Plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.1, patience = 7)
        
        else:
            raise ValueError("Invalid scheduler name. Please enter one of the following: Step LR, Exponential LR, Cosine Annealing LR, Reduce LR on Plateau")

        return self.scheduler




