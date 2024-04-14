import sys
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import constants
from OptimizerHelper import OptimizerHelper
from LossHelper import LossHelper
from SchedulerHelper import SchedulerHelper
from TrainHelper import data_prep, train_and_evaluate, gen_confusion_matrix, view_classification_report, save_log
from models.MLP import MLP
from plots import plot_accuracy, plot_loss
# from models.LSTMModel import LSTM

def train_MLP(model_name, filename, optimizer_name, criterion_name, scheduler_name, input_features, learning_rate, dropout_rate, hidden_units, output_features, isEarlyStopping): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtaining train_dataset, validation_dataset and test_dataset
    train_dataset, validation_dataset, test_dataset = data_prep("../data")

    # Creating data loaders for train, validation and test
    train_loader = DataLoader(train_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = False)

    # Initializing model and moving it to device
    mlp_model = MLP(input_features=input_features, dropout_rate=dropout_rate, hidden_units=hidden_units, output_features=output_features)
    mlp_model.to(device)
    
    # Defining loss function
    loss_helper = LossHelper(criterion_name = criterion_name, device = device)
    criterion = loss_helper.set_loss_function()

    # Defining optimizer
    optimizer_helper = OptimizerHelper(model = mlp_model, optimizer_name = optimizer_name, learning_rate = learning_rate)
    optimizer = optimizer_helper.set_optimizer()
    #print(optimizer)

    # Defining Learning Rate Scheduler
    scheduler_helper = SchedulerHelper(optimizer = optimizer, scheduler_name = scheduler_name)
    scheduler = scheduler_helper.set_scheduler()

    # Call the training, validation and testing functions with appropriate arguments
    # For epochs, use this 1 value
    train_accuracies, test_accuracies, validation_accuracies, train_losses, test_losses, validation_losses, all_test_labels, all_test_predictions = train_and_evaluate(model = mlp_model, model_name = model_name, filename = filename, train_loader = train_loader, validation_loader = validation_loader, test_loader = test_loader, criterion = criterion, optimizer = optimizer, scheduler = scheduler, epochs = 2, lr = learning_rate, is_early_stopping = isEarlyStopping, early_stopping_patience = constants.CNN_ES_PATIENCE)

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
    experiments = [
        ("Adam", 0.001, 0.25, [512, 256], False, "Reduce LR on Plateau", "Cross Entropy Loss", "Baseline"),
        ("Adam", 0.01, 0.25, [512, 256], False, "Reduce LR on Plateau", "Cross Entropy Loss", "HighLR"),
        ("Adam", 0.0001, 0.25, [512, 256], False, "Reduce LR on Plateau", "Cross Entropy Loss", "LowLR"),
        ("Adam", 0.001, 0.5, [512, 256], False, "Reduce LR on Plateau", "Cross Entropy Loss", "HighDropout"),
        ("Adam", 0.001, 0.25, [512, 256, 128], False, "Reduce LR on Plateau", "Cross Entropy Loss", "ExtraLayer"),
        ("Adam", 0.001, 0.25, [512, 256], True, "Reduce LR on Plateau", "Cross Entropy Loss", "EarlyStopping")]

    for optimizer_name, learning_rate, dropout_rate, hidden_units, isEarlyStopping, scheduler_name, criterion_name, experiment_name in experiments:
        filename = f"MLP_{experiment_name}"
        train_MLP(
            model_name="MLP Model",
            filename=filename,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            criterion_name=criterion_name,
            input_features=224*224*3,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            output_features=3,
            isEarlyStopping=isEarlyStopping,
        )
