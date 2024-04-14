import sys
import os
from os.path import dirname, abspath
parent_dir_path = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir_path)

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, multilabel_confusion_matrix, classification_report
from tqdm import tqdm

import constants
import seaborn as sns
from PreProcessor import DataPreprocessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_log(model_name, filename, train_accuracies, test_accuracies, validation_accuracies, train_losses, validation_losses, test_losses, all_test_labels, all_test_predictions):
    
    # Define the file path where you want to save the output
    output_file = "output.txt"

    # Define the directory path where image will be stored
    log_directory = f"./metrics/{model_name}/"
    os.makedirs(log_directory, exist_ok=True) # Create directory if it does not exist
    log_path = f"{log_directory}{filename}.txt"

    # Redirect standard output to the file
    with open(log_path, "w") as f:
        sys.stdout = f  # Redirect stdout to the file

        # Print the output
        print(f"Training accuracies: {train_accuracies}")
        print(f"Testing accuracies: {test_accuracies}")
        print(f"Validation accuracies: {validation_accuracies}")
        print(f"Training loses: {train_losses}")
        print(f"Validation loses: {validation_losses}")
        print(f"Testing loses: {test_losses}")

        for i in range(4):
            print("\n")

        # Call the classification report function
        view_classification_report(all_test_labels, all_test_predictions)

    # Restore standard output
    sys.stdout = sys.__stdout__


def gen_confusion_matrix(filename, all_test_labels, all_test_predictions):

    # Calculate confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_predictions)

    # Define class names (if available)
    class_names = ['Benign (Class 0)', 'Malignant (Class 1)', 'Normal (Class 2)'] 

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')


    # Define the directory path where image will be stored
    cf_directory = f"../assets/confusion_matrix/"
    os.makedirs(cf_directory, exist_ok=True) # Create directory if it does not exist
    cf_path = f"{cf_directory}{filename}.png"

    # Save the confusion matrix plot to the specified path
    # print('Confusion Matrix stored in: ', cf_path)
    plt.savefig(cf_path)


def view_classification_report(all_test_labels, all_test_predictions):
    print("\n")
    print(classification_report(all_test_labels, all_test_predictions))
    #print(classification_report(all_test_labels, all_test_predictions, zero_division = 1))


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach(), 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    return train_loss / len(train_loader), train_accuracy


def train_googlenet(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Perform forward pass through the network with given inputs
        # Note: Inception v3 model returns outputs in the form of (output, aux_logits) during training
        outputs, aux_outputs = model(inputs)
        
        # Calculates loss of main output and loss of auxiliary output
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        
        # Combine main loss and auxiliary loss to get final loss of GoogleNet (a weighted average of the two losses)
        loss = loss1 + 0.4 * loss2

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach(), 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    return train_loss / len(train_loader), train_accuracy

def validate(model, validation_loader, criterion, device):
    model.eval()
    validation_loss = 0
    correct_validation = 0
    total_validation = 0
    
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_validation += labels.size(0)
            correct_validation += (predicted == labels).sum().item()
    
    validation_accuracy = 100 * correct_validation / total_validation
    return validation_loss / len(validation_loader), validation_accuracy


def test(model, test_loader, criterion, device):
    
    model.eval()

    test_loss = 0
    correct_test = 0
    total_test = 0

    all_test_labels = []
    all_test_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct_test / total_test
    return test_loss / len(test_loader), test_accuracy, np.array(all_test_labels), np.array(all_test_predictions)


def train_and_evaluate(model, model_name, filename, train_loader, validation_loader, test_loader, criterion, optimizer, scheduler, epochs, lr, is_early_stopping, early_stopping_patience):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_accuracies = []
    test_accuracies = []
    validation_accuracies = []

    train_losses = []
    test_losses = []
    validation_losses = []

    best_val_loss = float('inf')
    best_model_state = deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):

        if model_name == "GoogleNet_Model":
            train_loss, train_accuracy = train_googlenet(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)
        else:
            train_loss, train_accuracy = train(model = model, train_loader = train_loader, criterion = criterion, optimizer = optimizer, device = device)


        train_losses.append(train_loss/len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation
        validation_loss, validation_accuracy = validate(model = model, validation_loader = validation_loader, criterion = criterion, device = device)

        validation_losses.append(validation_loss/len(validation_loader))
        validation_accuracies.append(validation_accuracy)

        # Step the learning rate scheduler based on validation loss
        scheduler.step(validation_loss)

        # Early stopping code
        if is_early_stopping:
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_model_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stopping_patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs!')
                    break

        # Testing    
        test_loss, test_accuracy, all_test_labels, all_test_predictions = test(model = model, test_loader = test_loader, criterion = criterion, device = device)

        test_losses.append(test_loss/len(test_loader))
        test_accuracies.append(test_accuracy)

        # Printing metrics for current epoch
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Validation Loss: {validation_loss/len(validation_loader):.4f}, "
                  f"Test Loss: {test_loss/len(test_loader):.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, "
                  f"Validation Accuracy: {validation_accuracy:.2f}%, "
                  f"Test Accuracy: {test_accuracy:.2f}%")

        print("--------------------------------------------------------------")


    ## AFTER THE EPOCH LOOP

    if is_early_stopping:

        # Find the index of the number in the list and add 1 to get the index starting from 1
        corresponding_epoch_number = validation_losses.index(best_val_loss) + 1 if best_val_loss in validation_losses else None


        print(f"Best validation loss obtained for epoch {corresponding_epoch_number}: {best_val_loss}")

    best_val_loss = min(validation_losses) # Find the maximum value in the list
    best_val_loss_epoch = validation_losses.index(best_val_loss) + 1 # Find the index of the maximum value and add 1 to get epoch number
    print(f"Best validation loss obtained for epoch {best_val_loss_epoch}: {best_val_loss}")

    best_test_accuracy = max(test_accuracies) # Find the maximum value in the list
    best_test_accuracy_epoch = test_accuracies.index(best_test_accuracy) + 1 # Find the index of the maximum value and add 1 to get epoch number
    print(f"Best test accuracy obtained for epoch {best_test_accuracy_epoch}: {best_test_accuracy} \n")

    # Load the best model state (necessary if early stopping has happened)
    model.load_state_dict(best_model_state)

    model_output_directory = f"./weights/{model_name}/"
    os.makedirs(model_output_directory, exist_ok=True) # Create directory if it does not exist
    model_output_path = f"{model_output_directory}{filename}.pt"

    # Save the model
    #print(f'Saving {model_name} model to {model_output_path}')
    torch.save(model.state_dict(), model_output_path)

    return train_accuracies, test_accuracies, validation_accuracies, train_losses, test_losses, validation_losses, all_test_labels, all_test_predictions


def data_prep(dir_path):

    preprocessor = DataPreprocessor(dataset_path = dir_path, transform_type = constants.DEFAULT_DATASET_TRANSFORM_TYPE, random_seed = constants.RANDOM_SEED)
    train_dataset, validation_dataset, test_dataset = preprocessor.process_dataset()

    return train_dataset, validation_dataset, test_dataset


def data_prep_googlenet(dir_path):

    preprocessor = DataPreprocessor(dataset_path = dir_path, transform_type = constants.DEFAULT_DATASET_TRANSFORM_TYPE, random_seed = constants.RANDOM_SEED)
    train_dataset, validation_dataset, test_dataset = preprocessor.process_dataset_googlenet()

    return train_dataset, validation_dataset, test_dataset