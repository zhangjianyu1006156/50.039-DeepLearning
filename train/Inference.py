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
from TrainHelper import data_prep, train_and_evaluate, test, gen_confusion_matrix, view_classification_report, save_log
from plots import plot_accuracy, plot_loss

import torchvision.models as models
from models.CNN import CNN
from models.MLP import MLP
from models.CNN_LSTM import CNN_LSTM

# from models.LSTMModel import LSTM

def inference(model_name, filename, criterion_name, saved_model_weight, input_features = None, dropout = None, hidden_units = None, fc_units = None, lstm_units = None, num_layers = None, output_features = 3): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtaining train_dataset, validation_dataset and test_dataset
    train_dataset, validation_dataset, test_dataset = data_prep("../data")

    # Creating data loaders for train, validation and test
    train_loader = DataLoader(train_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = True)
    validation_loader = DataLoader(validation_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = constants.CNN_BATCH_SIZE, shuffle = False)

    # Define model you want to inference
    if model_name == "CNN Model":
        inference_model = CNN(dropout_rate = dropout, fc_units = fc_units)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model.to(device)
    
    elif model_name == "MLP Model":
        inference_model = MLP(input_features = 224*224*3, dropout_rate = dropout, hidden_units = hidden_units, output_features = output_features)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model.to(device)
    
    elif model_name == "CNN LSTM Model":
        inference_model = CNN_LSTM(dropout_rate = dropout, fc_units = fc_units, lstm_units = lstm_units, num_layers = num_layers)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model.to(device)

    elif model_name == "ResNet_Model":
        inference_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        num_features = inference_model.fc.in_features
        inference_model.fc = nn.Linear(num_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)
    
    elif model_name == "VGG_Model":
        inference_model = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT')
        inference_model.classifier[6] = nn.Linear(inference_model.classifier[6].in_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)

    elif model_name == "DenseNet_Model":
        inference_model = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT')
        inference_model.classifier[6] = nn.Linear(inference_model.classifier[6].in_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)

    elif model_name == "DenseNet_Model":
        inference_model = models.densenet161(weights='DenseNet161_Weights.DEFAULT')
        inference_model.classifier = nn.Linear(inference_model.classifier.in_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)
    
    elif model_name == "MobileNet_Model":
        inference_model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
        inference_model.classifier[3] = nn.Linear(inference_model.classifier[3].in_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)

    elif model_name == "Wide_ResNet_Model":
        inference_model = models.wide_resnet101_2(weights='Wide_ResNet101_2_Weights.DEFAULT')
        num_features = inference_model.fc.in_features
        inference_model.fc = nn.Linear(num_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)

    elif model_name == "Alexnet_Model":
        inference_model = models.alexnet(weights='AlexNet_Weights.DEFAULT')
        inference_model.classifier[6] = nn.Linear(inference_model.classifier[6].in_features, 3)
        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)

    elif model_name == "GoogleNet_Model":
        inference_model = models.inception_v3(weights = 'Inception_V3_Weights.DEFAULT')
        num_classes = 3
        inference_model.fc = nn.Linear(inference_model.fc.in_features, num_classes)
        
        if hasattr(inference_model, 'AuxLogits'):  # Check and modify the auxiliary classifier
            inference_model.AuxLogits.fc = nn.Linear(inference_model.AuxLogits.fc.in_features, num_classes)

        inference_model.load_state_dict(torch.load(saved_model_weight))
        inference_model = inference_model.to(device)

    
    # Defining loss function
    loss_helper = LossHelper(criterion_name = criterion_name, device = device)
    criterion = loss_helper.set_loss_function()
    
    test_loss, test_accuracy, all_test_labels, all_test_predictions = test(model = inference_model, test_loader = test_loader, criterion = criterion, device = device)

    # print(f"Training accuracies: {train_accuracies}")
    # print(f"Testing accuracies: {test_accuracies}")
    # print(f"Validation accuracies: {validation_accuracies}")
    # print(f"Training loses: {train_losses}")
    # print(f"Validation loses: {validation_losses}")
    # print(f"Testing loses: {test_losses}")

    # Display classification report
    view_classification_report(all_test_labels, all_test_predictions)

    # Generate confusion matrix
    gen_confusion_matrix(filename, all_test_labels, all_test_predictions)


if __name__ == '__main__':
    
    ## All CNN Experiments

    # inference(model_name="CNN Model", filename="CNN_Baseline", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_Baseline.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_HighLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_HighLR.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_LowLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_LowLR.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_HighDropout", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_HighDropout.pt", dropout=0.7, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_LowDropout", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_LowDropout.pt", dropout=0.3, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_HighFCUnits", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_HighFCUnits.pt", dropout=0.5, fc_units=128, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_LowFCUnits", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_LowFCUnits.pt", dropout=0.5, fc_units=32, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_EarlyStopping", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_EarlyStopping.pt", dropout=0.5, fc_units=64, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_SGD_Optimizer", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_SGD_Optimizer.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_CosineAnnealing", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_CosineAnnealing.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_WeightedCE", criterion_name="Cross Entropy Loss Weighted", saved_model_weight="./weights/CNN Model/CNN_WeightedCE.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_RMSProp_Optimizer", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_RMSProp_Optimizer.pt", dropout=0.5, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_ComplexConfig", criterion_name="Cross Entropy Loss Weighted", saved_model_weight="./weights/CNN Model/CNN_ComplexConfig.pt", dropout=0.6, fc_units=96, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_LowDropHighLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_LowDropHighLR.pt", dropout=0.2, fc_units=64, isEarlyStopping=False)
    # inference(model_name="CNN Model", filename="CNN_HighDropLowLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_HighDropLowLR.pt", dropout=0.7, fc_units=64, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_Optimized_Dropout", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_Optimized_Dropout.pt", dropout=0.3, fc_units=64, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_High_Capacity", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_High_Capacity.pt", dropout=0.5, fc_units=128, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_Optimized_SGD", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_Optimized_SGD.pt", dropout=0.3, fc_units=64, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_Advanced_Regularization", criterion_name="Cross Entropy Loss Weighted", saved_model_weight="./weights/CNN Model/CNN_Advanced_Regularization.pt", dropout=0.3, fc_units=128, isEarlyStopping=True)
    # inference(model_name="CNN Model", filename="CNN_Combo_Best_Practices", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/CNN Model/CNN_Combo_Best_Practices.pt", dropout=0.3, fc_units=128, isEarlyStopping=True)

    ## ALL MLP Experiments


    # inference(model_name="MLP Model", filename="MLP_Baseline", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_Baseline.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_HighLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_HighLR.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_LowLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_LowLR.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_HighDropout", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_HighDropout.pt", dropout=0.5, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_ExtraLayer", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_ExtraLayer.pt", dropout=0.25, hidden_units=[512, 256, 128], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_EarlyStopping", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_EarlyStopping.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=True)

    ## ALL CNN-LSTM Experiments

    # inference(model_name="MLP Model", filename="MLP_Baseline", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_Baseline.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_HighLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_HighLR.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_LowLR", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_LowLR.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_HighDropout", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_HighDropout.pt", dropout=0.5, hidden_units=[512, 256], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_ExtraLayer", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_ExtraLayer.pt", dropout=0.25, hidden_units=[512, 256, 128], isEarlyStopping=False)
    # inference(model_name="MLP Model", filename="MLP_EarlyStopping", criterion_name="Cross Entropy Loss", saved_model_weight="./weights/MLP Model/MLP_EarlyStopping.pt", dropout=0.25, hidden_units=[512, 256], isEarlyStopping=True)


    ## Inferencing via MobileNet
    #inference(model_name = "MobileNet_Model", filename = "MobileNet", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/MobileNet_Model/MobileNet.pt")

    ## Inferencing via VGG-19
    #inference(model_name = "VGG_Model", filename = "VGG_NoES", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/VGG_Model/VGG_NoES.pt")

    ## Inferencing via ResNet
    inference(model_name = "ResNet_Model", filename = "ResNet", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/ResNet_Model/ResNet.pt")

    ## Inferencing via Wide-ResNet
    # inference(model_name = "Wide_ResNet_Model", filename = "Wide_ResNet", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/Wide_ResNet_Model/Wide_ResNet.pt")

    ## Inferencing via DenseNet
    #inference(model_name = "Densenet_Model", filename = "Densenet", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/Densenet_Model/Densenet_NoES.pt")

    ## Inferencing via AlexNet
    #inference(model_name = "Alexnet_Model", filename = "Alexnet", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/Alexnet_Model/Alexnet.pt")

    ## Inferencing via GoogleNet
    #inference(model_name = "GoogleNet_Model", filename = "GoogleNet", criterion_name = "Cross Entropy Loss Weighted", saved_model_weight = "./weights/GoogleNet_Model/GoogleNet.pt")


