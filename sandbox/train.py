from utils import load_mnist, load_usps
from model import FeatureExtractorCNN, Trainer, Evaluator
import torch.nn as nn

# path to USPS dataset
usps_data_path = '../Digit-Five/usps_28x28.pkl'
train_usps, test_usps = load_usps(usps_data_path, batch_size=64)
train_mnist, test_mnist = load_mnist(batch_size=64)

model = FeatureExtractorCNN()
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model=model, criterion=criterion)
evaluator = Evaluator(model=model)

num_epochs = 50
weight_wasserstein = 0.25
loss_function_str = 'wasserstein'

trainer.train(source_loader=train_mnist, target_loader=train_usps, num_epochs=num_epochs, weight_discrepancy=weight_wasserstein, loss_function_str=loss_function_str)

evaluator.evaluate(test_loader=test_usps, device='cpu')