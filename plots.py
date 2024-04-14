# Importing plotting/visualization and numerical computation libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_accuracy(model_name, filename, train_accuracies, test_accuracies, validation_accuracies):

    # Plotting Training, Validation and Test Accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training, Validation and Test Accuracies of {model_name}')
    plt.legend()
    plt.grid(True)

    # Define the directory path where image will be stored
    plot_accuracy_directory = f"../assets/accuracy_and_loss_graphs/{model_name}/"
    os.makedirs(plot_accuracy_directory, exist_ok=True) # Create directory if it does not exist
    plot_accuracy_path = f"{plot_accuracy_directory}{filename}_accuracy.png"

    plt.savefig(plot_accuracy_path)

def plot_loss(model_name, filename, train_losses, validation_losses, test_losses):
    
    # Plotting Training, Validation and Test Accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training, Validation and Test Losses of {model_name}')
    plt.legend()
    plt.grid(True)

    # Define the directory path where image will be stored
    plot_loss_directory = f"../assets/accuracy_and_loss_graphs/{model_name}/"
    os.makedirs(plot_loss_directory, exist_ok=True) # Create directory if it does not exist
    plot_loss_path = f"{plot_loss_directory}{filename}_loss.png"

    plt.savefig(plot_loss_path)
