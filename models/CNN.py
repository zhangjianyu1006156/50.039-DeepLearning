import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, dropout_rate, fc_units):
        
        # Initialize base class, nn.Module
        super(CNN, self).__init__()
        
        # First convolutional layer with 3 input channels (RGB images), 32 output channels, 3x3 kernel size, and 1 pixel padding
        # Note: Width & Height don't change as padding is 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        
        # Batch normalization 
        self.bn1 = nn.BatchNorm2d(32)
        
        # ReLu Activation
        self.act1 = nn.ReLU()
        
        # Max pooling with 2*2 kernel size and stride of 2
        # Note: The Width & Height will be halved here, as stride is 2, with 2*2 kernel size (skipping over alternate columns)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layer applied after the first pooling layer
        self.dropout1 = nn.Dropout(dropout_rate)  
        
        # Second convolutional layer with 32 input channels (output from the previous layer), 64 output channels, 3x3 kernel size, and 1 pixel padding
        # Note: Width & Height don't change as padding is 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(64)  
        
        # ReLu Activation
        self.act2 = nn.ReLU()
        
        # Dropout layer applied after the second pooling layer
        self.dropout2 = nn.Dropout(dropout_rate)   
        
        # Commented - 3rd Convolutional Layer
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(128)
        #self.act3 = nn.ReLU()
        #self.dropout3 = nn.Dropout(dropout_rate)  
        #self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=fc_units)
        
        # First fully connected of neural network, with fc_units number of output features/neurons in the fully connected layer
        self.fc1 = nn.Linear(in_features = 64 * 56 * 56, out_features = fc_units)
        
        # Second fully connected layer with 3 output features for classification (benign, malignant, and normal)
        self.fc2 = nn.Linear(fc_units, 3)
        
        # Final dropout layer with specified dropout rate
        self.dropout4 = nn.Dropout(dropout_rate)


    def forward(self, x):
        
        # Applies the first convolutional layer, then ReLU activation function, then batch normalization and max pooling
        x = self.pool(self.bn1(self.act1(self.conv1(x))))
        
        # Applies dropout to output of the first pooling layer
        x = self.dropout1(x)
        
        # Applies the second convolutional layer, then ReLU activation function, then batch normalization and max pooling
        x = self.pool(self.bn2(self.act2(self.conv2(x))))
        
        # Applies dropout to output of the second pooling layer
        x = self.dropout2(x)
        
        #x = self.pool(self.bn3(self.act3(self.conv3(x))))
        #x = self.dropout3(x)
        
        # Prepare data for input into fully-connected layers by flattening output of the last pooling layer into a 1-dimensional tensor
        x = torch.flatten(x, 1)
        
        # Applies dropout to the output of the first fully connected layer 
        x = self.dropout4(self.fc1(x))
        
        # Computes the final output of the neural network by passing the output of the first fully connected layer through the second fully connected layer
        # This represents class scores for each input sample
        x = self.fc2(x)

        return x