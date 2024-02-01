import torch
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
from torch import flatten
import torch.nn as nn

class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		x = flatten(x, 1)
		feature = x.copy()
		x = self.fc1(x)
		x = self.relu3(x)
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output, feature

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.classification_loss = nn.NLLLoss()  # NLLLoss car LogSoftmax est utilisé

    def forward(self, mnist_outputs, mnist_labels, mnist_features, usps_features,lmbda):
        # Perte de classification pour MNIST
        class_loss = self.classification_loss(mnist_outputs, mnist_labels)

        # Perte basée sur la distance pour les features
        feature_loss = torch.norm(mnist_features - usps_features, p=2)**2

        # Combinaison des pertes
        return class_loss + lmbda * feature_loss
