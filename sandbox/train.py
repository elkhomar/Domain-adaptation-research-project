from utils import load_mnist, load_usps
from model import FeatureExtractorCNN, TargetDomainMLP, Trainer, Evaluator
import torch.nn as nn
import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# path to USPS dataset
usps_data_path = '../Digit-Five/usps_28x28.pkl'
train_usps, test_usps = load_usps(usps_data_path, batch_size=64)
train_mnist, test_mnist = load_mnist(batch_size=64)

feature_extractor = FeatureExtractorCNN()
target_model = TargetDomainMLP()
criterion = nn.CrossEntropyLoss()

trainer = Trainer(feature_extractor=feature_extractor, target_model=target_model, criterion=criterion)
evaluator = Evaluator(feature_extractor=feature_extractor, target_model=target_model)

num_epochs = 100
weight_discrepancy_loss = 0
loss_function_str = 'coral'

trainer.train(source_loader=train_mnist, target_loader=train_usps, num_epochs=num_epochs,
              weight_discrepancy=weight_discrepancy_loss, test_loader=test_usps, dataset='USPS', loss_function_str=loss_function_str)

evaluator.evaluate(test_loader=test_usps, dataset='USPS', device='cpu')
evaluator.evaluate(test_loader=test_mnist, dataset='MNIST', device='cpu')

