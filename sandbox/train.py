from utils import load_mnist, load_usps
from model import FeatureExtractorCNN, Trainer, Evaluator
import torch.nn as nn
import torch
from Losses import calculate_mean_distance
# path to USPS dataset
usps_data_path = '/home/sebastien/projets/Domain-adaptation-research-project/data/usps_28x28.pkl'
train_usps, test_usps = load_usps(usps_data_path, batch_size=64)
train_mnist, test_mnist = load_mnist(batch_size=64)

mean_distance = calculate_mean_distance(train_mnist, train_usps)

model = FeatureExtractorCNN()
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model=model, criterion=criterion)
evaluator = Evaluator(model=model)

num_epochs = 50
weight_wasserstein = 0.7
loss_function_str = 'MMD'

trainer.train(source_loader=train_mnist, target_loader=train_usps, num_epochs=num_epochs, weight_discrepancy=weight_wasserstein, loss_function_str=loss_function_str,mean_distance=mean_distance)

evaluator.evaluate(test_loader=test_usps, device='cpu')
