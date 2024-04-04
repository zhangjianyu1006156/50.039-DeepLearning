# Lung Cancer Detection Model

## Overview
This repository contains a convolutional neural network (CNN) model designed for lung cancer detection from image data. The model was trained on a human lung CT scan image dataset, using PyTorch for model development and training. This document outlines how to set up, train, and evaluate this model, which is customized for execution in the Kaggle environment.

## Features
- Utilization of PyTorch for building and training a CNN model.
- Implementation of data augmentation to improve model robustness.
- Hyperparameter tuning through random search to optimize model performance.
- Evaluation metrics including accuracy and loss.

## Dataset
The model is trained on the "The IQ-OTHNCCD Lung Cancer Dataset" available on Kaggle. This dataset comprises images classified into multiple categories indicative of lung cancer presence and severity.

## Prerequisites
- Kaggle Account
- Basic knowledge of PyTorch and deep learning
- Familiarity with the Kaggle Notebook environment

## Setup instructions
1. **Clone this repository** to obtain a copy of the model code and utility scripts.
2. **Create a new Kaggle Notebook**: Import the model code and data set into the Kaggle Notebook.
3. **Install Dependencies**: Make sure all required libraries mentioned in the import section are installed in your Kaggle Notebook environment.

## Running the Model
To train and evaluate the model within your Kaggle Notebook:
1. **Load the Dataset**: Adjust the dataset path in the code to point to your uploaded Kaggle dataset.
2. **Configure Hyperparameters**: Set your desired hyperparameters or use the provided defaults for initial testing.
3. **Train the Model**: Execute the training script. The model will train on the specified dataset and output training, validation, and test accuracies.
4. **Evaluate the Model**: After training, the model's performance can be evaluated on unseen data using the provided evaluation metrics.

## How to import the dataset
To import the dataset into the Kaggle environment:
1. Click the **File**
2. Search the **dataset name**
3. Click the **Add button**
4. At the right side of the page, you can **copy file path** to use the dataset

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

## Where to find and download the best model weight file
1. Go the **Code**
2. Click **Your work**
3. Find **your notebook**
4. Click the notebook
5. Click the **Output**

## Best model weight file
Google Drive Link: https://drive.google.com/file/d/1fgh0FikO4hpj261qTr95D0yERRaXxySq/view?usp=sharing
