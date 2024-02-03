import torch.nn as nn
import torch.nn.functional as F
from ot import solve_sample
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW

class FeatureExtractorCNN(nn.Module):
    """
    A convolutional neural network model intended for feature extraction, structured for processing
    1-channel image inputs. The model consists of three convolutional layers, each followed by batch
    normalization, ReLU activation, and max-pooling. Dropout is applied after each max-pooling step and
    before the final fully connected layers to reduce overfitting.

    Attributes:
    - conv1, conv2, conv3 (nn.Conv2d): Convolutional layers.
    - bn1, bn2, bn3 (nn.BatchNorm2d): Batch normalization layers corresponding to each convolutional layer.
    - fc1, fc2 (nn.Linear): Fully connected layers for classification.
    - dropout (nn.Dropout): Dropout layer for regularization.

    Methods:
    - forward(x): Defines the forward pass of the model.
    """
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(3 * 3 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10) 
        self.dropout = nn.Dropout(p=0.2) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x) 
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 3 * 3 * 128) 
        x = self.dropout(x)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x

class Trainer:
    def __init__(self, model, criterion, optimizer_lr=2e-5, scheduler_step_size=10, scheduler_gamma=0.1):
        """
        Initializes the Trainer class with the model, loss criterion, and optimizer settings.

        Parameters:
        - model (torch.nn.Module): The model to be trained.
        - criterion (torch.nn.modules.loss): The loss function used for the classification task.
        - optimizer_lr (float, optional): Learning rate for the optimizer. Default is 2e-5.
        - scheduler_step_size (int, optional): Step size for the learning rate scheduler. Default is 10.
        - scheduler_gamma (float, optional): Gamma for the learning rate scheduler. Default is 0.1.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = AdamW(model.parameters(), lr=optimizer_lr)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def train(self, source_loader, target_loader, num_epochs, weight_discrepancy, loss_function_str=None):
        """
        Trains the model using both source and target data loaders with the option to incorporate a
        domain adaptation loss, specifically the Wasserstein distance, as part of the training process.

        Parameters:
        - source_loader (DataLoader): DataLoader for the source domain data.
        - target_loader (DataLoader): DataLoader for the target domain data.
        - num_epochs (int): The number of epochs to train the model.
        - weight_discrepancy (float): Weight of the discrepancy distance loss in the total loss calculation.
        - loss_function_str (str, optional): A string to specify the type of loss function for domain adaptation. 
          If 'wasserstein', the Wasserstein distance is used as part of the loss.
        """
        self.model.train()
        for epoch in range(num_epochs):
            for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
                self.optimizer.zero_grad()

                source_features = self.model(source_data).detach()
                target_features = self.model(target_data).detach()

                source_features_flat = source_features.view(source_features.size(0), -1).cpu().numpy()
                target_features_flat = target_features.view(target_features.size(0), -1).cpu().numpy()

                classification_loss = self.criterion(self.model(source_data), source_labels)
                discrepancy_loss = torch.tensor(0.0, requires_grad=True)

                if loss_function_str == 'wasserstein':
                    result = solve_sample(source_features_flat, target_features_flat, metric='sqeuclidean', reg=0.1, method='sinkhorn', max_iter=2000)
                    if hasattr(result, 'value'): 
                        discrepancy_loss_value = result.value
                        discrepancy_loss = torch.tensor(discrepancy_loss_value, dtype=torch.float32, requires_grad=True).to(source_features.device)

                total_loss = classification_loss + weight_discrepancy * discrepancy_loss
                total_loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs}, LR: {self.scheduler.get_last_lr()[0]} completed.")


class Evaluator:
    def __init__(self, model):
        """
        Initializes the Evaluator class with the model.

        Parameters:
        - model (torch.nn.Module): The trained model to be evaluated.
        """
        self.model = model

    def evaluate(self, test_loader, device='cpu'):
        """
        Evaluates the model's performance on a test dataset.

        Parameters:
        - test_loader (DataLoader): DataLoader for the test data.
        - device (str, optional): The device to run the evaluation on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.model.eval()
        self.model.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                if data.shape[1] != 1:
                    raise ValueError(f"Source data should have 1 channel, got {data.shape[1]}")

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(labels, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy}%')
