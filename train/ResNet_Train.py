import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import constants
from OptimizerHelper import OptimizerHelper
from LossHelper import LossHelper
from SchedulerHelper import SchedulerHelper
from TrainHelper import data_prep, train_and_evaluate, gen_confusion_matrix, view_classification_report, save_log
from plots import plot_accuracy, plot_loss
# from models.LSTMModel import LSTM

def train_resnet(model_name, filename, optimizer_name, criterion_name, scheduler_name, learning_rate, isEarlyStopping): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtaining train_dataset, validation_dataset and test_dataset
    train_dataset, validation_dataset, test_dataset = data_prep("../data")

    # Creating data loaders for train, validation and test
    train_loader = DataLoader(train_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = False)

    # Load a pre-trained ResNet model
    # resnet_model = models.resnet152(pretrained=True)
    resnet_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')

    # Modify the model for the 3 classes: benign, malignant, normal
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, 3)
    resnet_model = resnet_model.to(device)
    
    # Defining loss function
    loss_helper = LossHelper(criterion_name = criterion_name, device = device)
    criterion = loss_helper.set_loss_function()

    # Defining optimizer
    optimizer_helper = OptimizerHelper(model = resnet_model, optimizer_name = optimizer_name, learning_rate = learning_rate)
    optimizer = optimizer_helper.set_optimizer()
    #print(optimizer)

    # Defining Learning Rate Scheduler
    scheduler_helper = SchedulerHelper(optimizer = optimizer, scheduler_name = scheduler_name)
    scheduler = scheduler_helper.set_scheduler()

    # Call the training, validation and testing functions with appropriate arguments
    # For epochs, use this 1 value
    train_accuracies, test_accuracies, validation_accuracies, train_losses, test_losses, validation_losses, all_test_labels, all_test_predictions = train_and_evaluate(model = resnet_model, model_name = model_name, filename = filename, train_loader = train_loader, validation_loader = validation_loader, test_loader = test_loader, criterion = criterion, optimizer = optimizer, scheduler = scheduler, epochs = 30, lr = learning_rate, is_early_stopping = isEarlyStopping, early_stopping_patience = constants.CNN_ES_PATIENCE)

    # print(f"Training accuracies: {train_accuracies}")
    # print(f"Testing accuracies: {test_accuracies}")
    # print(f"Validation accuracies: {validation_accuracies}")
    # print(f"Training loses: {train_losses}")
    # print(f"Validation loses: {validation_losses}")
    # print(f"Testing loses: {test_losses}")

    # Display classification report
    view_classification_report(all_test_labels, all_test_predictions)

    # Plotting Training, Validation and Test Accuracies
    plot_accuracy(model_name, filename, train_accuracies, test_accuracies, validation_accuracies)
    
    # Plotting Training, Validation and Test Losses
    plot_loss(model_name, filename, train_losses, validation_losses, test_losses)

    # Generate confusion matrix
    gen_confusion_matrix(filename, all_test_labels, all_test_predictions)

    # Save txt file of different metrics
    save_log(model_name, filename, train_accuracies, test_accuracies, validation_accuracies, train_losses, validation_losses, test_losses, all_test_labels, all_test_predictions)


if __name__ == '__main__':
    train_resnet(model_name = "ResNet_Model", filename = "ResNet", optimizer_name = "Adam", scheduler_name = "Reduce LR on Plateau", criterion_name = "Cross Entropy Loss Weighted", learning_rate = 0.001, isEarlyStopping = False)