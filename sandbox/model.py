import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import AdamW
from losses import wasserstein, coral

class FeatureExtractorCNN(nn.Module):
    """
    Attributes:
        - conv1, conv2, conv3 (nn.Conv2d): Convolutional layers designed to capture spatial hierarchies 
          of features from input images.
        - bn1, bn2, bn3 (nn.BatchNorm2d): Batch normalization layers applied after each convolutional 
          layer to stabilize learning and normalize the feature maps.
        - fc1, fc2 (nn.Linear): Fully connected layers that act as classifiers based on the high-level 
          features extracted by the convolutional and pooling layers.
        - dropout (nn.Dropout): Dropout layers applied after pooling and before fully connected layers 
          to prevent overfitting by randomly zeroing some of the elements of the input tensor.

    Methods:
        - forward(x, to_g=False): Defines the forward pass of the model. The input `x` is processed 
          through convolutional, batch normalization, activation, and pooling layers, with dropout 
          applied at specified points. If `to_g` is True, the method returns the flattened feature 
          vector after the last pooling layer, bypassing the fully connected layers. This is particularly 
          useful for tasks that require raw features instead of classification outputs, enabling the model 
          to serve dual purposes - feature extraction and classification.
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
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, to_g=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        if to_g:  # If the data is for the target domain, bypass the rest of CNN
            return x.view(-1, 3 * 3 * 128)
        x = x.view(-1, 3 * 3 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TargetDomainMLP(nn.Module):
    """
    Attributes:
    - fc1 (nn.Linear): First fully connected layer that maps input feature vectors to a 512-dimensional
        hidden layer, introducing a layer of abstraction.
    - fc2 (nn.Linear): Second fully connected layer that maps the 512-dimensional hidden representation
        to the final output size of 10, suitable for classification tasks with 10 classes.
    - dropout (nn.Dropout): Dropout layer applied after the first fully connected layer to prevent
        overfitting by randomly setting a fraction of input units to 0 at each update during training.

    Methods:
    - forward(x): Propagates the input through the network. It applies a ReLU activation after the first
        fully connected layer, then dropout, and finally outputs through the second fully connected layer.
        This process is designed to efficiently map the high-dimensional input features to a set of outputs
        that can be used for classification.
    """

    def __init__(self):
        super(TargetDomainMLP, self).__init__()
        self.fc1 = nn.Linear(3 * 3 * 128, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class Trainer:
    """
    Initializes the Trainer class with the model, loss criterion, and optimizer settings.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - criterion (torch.nn.modules.loss): The loss function used for the classification task.
    - optimizer_lr (float, optional): Learning rate for the optimizer. Default is 2e-5.
    - scheduler_step_size (int, optional): Step size for the learning rate scheduler. Default is 10.
    - scheduler_gamma (float, optional): Gamma for the learning rate scheduler. Default is 0.1.
    """
    def __init__(self, feature_extractor, target_model, criterion, optimizer_lr=4e-5, scheduler_step_size=11, scheduler_gamma=0.2):
        self.feature_extractor = feature_extractor
        self.target_model = target_model
        self.criterion = criterion
        self.optimizer = AdamW(list(feature_extractor.parameters()) + list(target_model.parameters()), lr=optimizer_lr)
        # self.scheduler = StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    def train(self, source_loader, target_loader, num_epochs, weight_discrepancy, test_loader, dataset, loss_function_str=None, device='cpu'):
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
        self.feature_extractor.to(device)
        self.target_model.to(device)
        self.feature_extractor.train()
        self.target_model.train()

        for epoch in range(num_epochs):
            total_loss_accumulated = 0.0
            total_val_loss_accumulated = 0.0
            total_samples = 0
            total_val_samples = 0

            for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
                source_data, source_labels = source_data.to(device), source_labels.to(device)
                target_data = target_data.to(device)

                self.optimizer.zero_grad()

                source_features = self.feature_extractor(source_data, to_g=True)
                target_features = self.feature_extractor(target_data, to_g=True)

                source_predictions = self.target_model(source_features)
                classification_loss = self.criterion(source_predictions, source_labels)

                discrepancy_loss = torch.tensor(0.0, requires_grad=True).to(device)

                if loss_function_str == 'wasserstein':
                    discrepancy_loss = wasserstein(source_features, target_features)

                if loss_function_str == 'coral':
                    discrepancy_loss = coral(source_features, target_features)

                total_loss = classification_loss + weight_discrepancy * discrepancy_loss
                total_loss.backward()
                self.optimizer.step()

                total_loss_accumulated += total_loss.item() * source_data.size(0)
                total_samples += source_data.size(0)

            average_loss = total_loss_accumulated / total_samples
            # self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")
            
            self.feature_extractor.eval()
            self.target_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)

                    features = self.feature_extractor(data, to_g=True)
                    outputs = self.target_model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    labels = torch.argmax(labels, dim=1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    val_loss = self.criterion(outputs, labels)
                    total_val_loss_accumulated += val_loss.item() * data.size(0)
                    total_val_samples =+ data.size(0)

            average_val_loss = total_val_loss_accumulated / total_val_samples
            accuracy = 100 * correct / total
            self.scheduler.step(average_val_loss)
            print(f'Accuracy / val loss on the {dataset} test images: {accuracy:.2f}% / {average_val_loss:.4f}')


class Evaluator:
    def __init__(self, feature_extractor, target_model):
        """
        Initializes the Evaluator class with the model.

        Parameters:
        - model (torch.nn.Module): The trained model to be evaluated.
        """
        self.feature_extractor = feature_extractor
        self.target_model = target_model

    def evaluate(self, test_loader, dataset, device='cpu'):
        """
        Evaluates the model's performance on a test dataset.

        Parameters:
        - test_loader (DataLoader): DataLoader for the test data.
        - device (str, optional): The device to run the evaluation on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.feature_extractor.to(device)
        self.target_model.to(device)
        self.feature_extractor.eval()
        self.target_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                features = self.feature_extractor(data, to_g=True)
                outputs = self.target_model(features)
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(labels, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on the {dataset} test images: {accuracy:.4f}%')