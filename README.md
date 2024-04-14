# Lung Cancer Detection Model

## Overview

This repository contains a custom built Multi-layer perceptron (MLP), Convolutional Neural Network (CNN) and  Convolutional Neural Network + Long Short-Term Memory Network (CNN + LSTM), compared with multiple state-of-the-art models. These models were trained on a human lung CT scan image dataset, designed for lung cancer detection, using PyTorch for model development and training.

## Dataset

The model is trained on the **The IQ-OTHNCCD Lung Cancer Dataset** available on Kaggle. This dataset comprises images classified into multiple categories indicative of lung cancer presence and severity.

Dataset Link (Public): [Kaggle Link](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
Dataset Link (Public):   [OneDrive Link](https://sutdapac-my.sharepoint.com/:f:/g/personal/yashpiyush_mehta_mymail_sutd_edu_sg/Eubsf2h4n1JDi9A80J_QAf8BdbJkKGoBaO9NFcKMvkXirQ?e=ieNRp5)

## Features

- Utilization of PyTorch for building and training custom MLP, CNN and CNN + LSTM models
- Comparing against various state-of-the-art models like ResNet-152, VGG-19, DenseNet-161, MobileNetV3-Large, Wide ResNet-101-2 and GoogleNet.
- Implementation of data augmentation to improve model robustness.
- Helper classes for Loss function, Optimizer and Learning Rate Scheduler for easier experimentation & modularity.
- Easy saving weights from trained model and loading weights for  inferencing.
- Hyperparameter tuning through random search to optimize model performance.
- Evaluation metrics including accuracy and loss over epoch, classification report. 
- Visual plots like Accuracy over epoch, Loss over epoch, confusion matrix.


## Setup instructions

1.  **Clone this repository:** `git clone` the repository or download it as zip file and extract it. 
2. **Install Dependencies:** `pip install -r requirements.txt` This will install all libraries and related dependencies needed for the project. Please install the pytorch version related to your cuda (GPU - this will help grant faster execution times).

## Setup instructions



## Running the Model

To train and evaluate the model using the python files:

 - **Head over to train folder**: Select the specific model you are looking to run. It will be defined as "{model}_train.py"
 - **View the train function**: The train function within a file are named as "def train_{model}" with certain parameters. Please read through the code to understand exactly what the parameters are changing.

	Some common ones include:
	 - ***model_name:*** This is the name of the model you are running. In some cases, this will be used to set a folder, before placing a model weight, plots etc. Additionally, please look at "Inference.py" and make sure the model_name will be compatible with the one defined there.
	 -  ***filename:*** This will set the filename again for a model weight, plot, log file etc.
	 - ***criterion_name:*** This will set the loss function. Please choose out of "Cross Entropy Loss", "Cross Entropy Loss Weighted" and "Multi Margin Loss".
	 - ***scheduler_name:*** This will set the Learning Rate Scheduler. Please choose out of "Step LR", "Exponential LR", "Cosine Annealing LR" and "Reduce LR on Plateau".
	 -  ***optimizer_name:*** This will set the Optimizer. Please choose out of "Adam", "SGD", "AdamW", "Adagrad" and "RMSProp".

Apart from these, the parameters would most likely be hyperparameters or other parameters.

Upon successfully executing the function, it will take care of the whole process, from loading data to outputting the metrics.

Alternatively, you could access the Jupyter Notebook's and run the notebook

## Access weights of model

To access weights of model that has already been trained, which you can simply load for inferencing, please use the below link.

Model weights (Public): [OneDrive Link](https://sutdapac-my.sharepoint.com/:f:/g/personal/yashpiyush_mehta_mymail_sutd_edu_sg/Eubsf2h4n1JDi9A80J_QAf8BdbJkKGoBaO9NFcKMvkXirQ?e=d1b4C7)

## Conducting Inferencing

Download weights from the aforementioned OneDrive Link and place them in "train/weights" folder. 
You can load the model for inferencing by going to "Train/Inference.py". 
Therein, simply define your model, then call the model you want to load (with the appropriate parameters).
There is pre-exisisting commented code for all models implemented in this repository to help you gain pace.

If you would like to conduct inferencing in the .ipynb files, please refer to the simple, 3-step code block below:
```python
model = CNN(parameters = parameters) #Understand model parameters & define appropriately
model.load_state_dict(torch.load('path_to_best_model.pt')) # Load the model from path
model.eval() # Set model in evaluation mode - very important!
```
Replace `path_to_best_model.pt` with the actual path to your saved model file.

## Hyperparameter Tuning

Utilize the `random_search` function in the .ipynb files and define your different hyperparameter combinations to explore randomly. This function trains multiple models with randomly selected hyperparameters and saves the model with the best validation accuracy.

Alternatively, you can define your own set of "experiments" and call them under the respective model you are training for in the "train" folder.

For example: If I want to do hyperparameter tuning of CNN Model, I will head over to "Train/CNN_Train.py" and in the main method, input my desired function.

We recommend utilizing the  `random_search` function with various parameters to consider edge-cases and unique combinations and then "experiment" individually.

PS: Grid search hyperparameter tuning has not been implemented as its resource-intensive