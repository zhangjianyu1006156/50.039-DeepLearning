import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    
    def __init__(self, dropout_rate, fc_units, lstm_units, num_layers):
        
        super(CNN_LSTM, self).__init__()
        
        
        ## CNN Part
        
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
        
        
        
        ## LSTM Part
        
        
        # Obtain inputs of flattened size of 64 * 56 (dimensions after the CNN layers), with tensors in a batch first manner
        self.lstm = nn.LSTM(input_size=64 * 56, hidden_size=lstm_units, num_layers=num_layers, batch_first=True)
        
        # Apply dropout to output of LSTM layer for regularization
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # First Fully Connected Layer mapping output of the LSTM layer to a lower-dimensional space with fc_units neurons
        self.fc1 = nn.Linear(lstm_units, fc_units)  
        
        # Second Fully Connected Layer mapping output of the previous fully connected layer to the final output space with 3 classes
        self.fc2 = nn.Linear(fc_units, 3)  
        
    
    def forward(self, x):
        
        
        ## CNN Part
        
        
        # Applies the first convolutional layer, then ReLU activation function, then batch normalization and max pooling
        x = self.pool(self.bn1(self.act1(self.conv1(x))))
        
        # Applies dropout to output of the first pooling layer
        x = self.dropout1(x)
        
        # Applies the second convolutional layer, then ReLU activation function, then batch normalization and max pooling
        x = self.pool(self.bn2(self.act2(self.conv2(x))))
        
        # Applies dropout to output of the second pooling layer
        x = self.dropout2(x)
        
        # At the end of CNN, the dimensions would be:
        # (batch_size, channels, height, width)
        
        
        ## Prepare for LSTM
        
        
        # Swap the second dimension (channels) with third dimension (height) so it becomes (batch_size, height, channels, width)
        # This is because the sequence length of LSTM needs to be the height of the images 
        x = x.permute(0, 2, 1, 3).contiguous()  
        
        # Load x.size() as such 
        batch_size, seq_len, channels, height = x.size()
        
        # Reshape the last input_size dimensions
        x = x.view(batch_size, seq_len, -1)  
        
        
        ## LSTM Part
        
        
        # Pass input tensor, x to the model. 
        # Returns output of LSTM layer at each time step, along with a tuple of hidden state and cell state of LSTM at last time step
        x, (hn, cn) = self.lstm(x)
        
        # Applying dropout to this selected hidden state of the LSTM at the last time step for each sample in the batch, as we are only interested in final output
        x = self.dropout3(x[:, -1, :])  
        
        # Utilizing the 2 Fully Connected Layer
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x