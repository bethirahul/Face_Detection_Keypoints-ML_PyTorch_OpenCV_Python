## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one 2x2 pool layer, output Tensor for one image becomes (32, 110, 110)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 square conv kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        # after another pool layer, this becomes (64, 54, 54)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 square conv kernel
        self.conv3 = nn.Conv2d(64, 128, 3)
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (128, 52, 52)
        # after another pool layer, this becomes (128, 26, 26)
        
        # third conv layer: 128 inputs, 256 outputs, 3x3 square conv kernel
        self.conv4 = nn.Conv2d(128, 256, 2)
        ## output size = (W-F)/S +1 = (26-2)/1 +1 = 25
        # the output tensor will have dimensions: (256, 25, 25)
        # after another pool layer, this becomes (256, 12, 12); 12.5 is floored
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        
        # 128 outputs * the 24*24 filtered/pooled map size
        #### self.fc1 = nn.Linear(512*4*4, 1000)
        self.fc1 = nn.Linear(256*12*12, 2000)
        
        # dropout with p=0.2
        #### self.fc1_drop = nn.Dropout(p=0.2)
        
        # finally, create 136 output channels (for the 136 keypoints)
        #### self.fc2 = nn.Linear(1000, 136)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        
        # dropout with p=0.4
        #### self.fc2_drop_1 = nn.Dropout(p=0.4)
        
        # dropout with p=0.6
        #### self.fc2_drop_2 = nn.Dropout(p=0.6)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        #### x = self.pool1(F.relu(self.conv1(x)))
        #### x = self.pool2(F.relu(self.conv2(x)))
        #### x = self.pool3(F.relu(self.conv3(x)))
        #### x = self.pool4(F.relu(self.conv4(x)))
        #### x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop6(self.pool(F.relu(self.conv1(x))))
        x = self.drop5(self.pool(F.relu(self.conv2(x))))
        x = self.drop4(self.pool(F.relu(self.conv3(x))))
        x = self.drop2(self.pool(F.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.drop4(F.relu(self.fc1(x)))
        x = self.drop1(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        #### x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        #### x = F.relu(self.fc1(x))
        #### x = self.fc1_drop(x)
        #### x = self.fc2(x)
        #### x = self.fc2_drop_1(x)
        #### x = self.fc2_drop_2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
