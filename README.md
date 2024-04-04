# Lung Cancer Detection Model

## Overview
This repository contains a convolutional neural network (CNN) model designed for lung cancer detection from image data. The model is trained on a dataset of lung images, utilizing PyTorch for model development and training. This document outlines how to set up, train, and evaluate the model, specifically tailored for execution within Kaggle's environment.

## Features
- Utilization of PyTorch for building and training a CNN model.
- Implementation of data augmentation to improve model robustness.
- Hyperparameter tuning through random search to optimize model performance.
- Evaluation metrics including accuracy, precision, recall, and confusion matrix analysis.

## Dataset
The model is trained on the "The IQ-OTHNCCD Lung Cancer Dataset" available on Kaggle. This dataset comprises images classified into multiple categories indicative of lung cancer presence and severity.

## Prerequisites
- Kaggle Account
- Basic knowledge of PyTorch and deep learning
- Familiarity with the Kaggle Notebook environment

## Setup Instructions
1. **Fork or clone this repository** to get a copy of the model code and utility scripts.
2. **Upload the dataset** to your Kaggle account. Ensure the dataset path in the code matches your Kaggle dataset path.
3. **Create a new Kaggle Notebook**: Import the model code and dataset to a Kaggle Notebook.
4. **Install Dependencies**: Ensure all required libraries mentioned in the import section are installed in your Kaggle Notebook environment.

## Running the Model
To train and evaluate the model within your Kaggle Notebook:
1. **Load the Dataset**: Adjust the dataset path in the code to point to your uploaded Kaggle dataset.
2. **Configure Hyperparameters**: Set your desired hyperparameters or use the provided defaults for initial testing.
3. **Train the Model**: Execute the training script. The model will train on the specified dataset and output training, validation, and test accuracies.
4. **Evaluate the Model**: After training, the model's performance can be evaluated on unseen data using the provided evaluation metrics.

## Hyperparameter Tuning
Utilize the `random_search` function provided to explore different hyperparameter combinations. This function trains multiple models with randomly selected hyperparameters and saves the model with the best validation accuracy.

## Using the Trained Model
The best-performing model is saved automatically. You can load this model for inference as follows:
```python
model = CNN_for_LungCancer(dropout_rate=best_dropout, fc_units=best_fc_units)
model.load_state_dict(torch.load('path_to_best_model.pt'))
model.eval()
```
Replace `path_to_best_model.pt` with the actual path to your saved model file.
