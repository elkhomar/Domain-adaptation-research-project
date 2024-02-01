import matplotlib
matplotlib.use("Agg")
import model
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=False,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=False,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
TRAIN_SPLIT = 0.70
VAL_SPLIT = 1 - TRAIN_SPLIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mnist_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
usps_loader = DataLoader(usps_dataset, batch_size=32, shuffle=True)

model = LeNet(numChannels=1, classes=10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
custom_loss = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
lmbda = 0.1
for epoch in range(num_epochs):
    for (mnist_data, mnist_labels), usps_data in zip(mnist_loader, usps_loader):
        mnist_data, mnist_labels = mnist_data.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), mnist_labels.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        usps_data = usps_data.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Forward pass
        mnist_outputs, mnist_features = model(mnist_data)
        usps_features = model(usps_data)[1]

        # Calcul de la perte
        loss = custom_loss(mnist_outputs, mnist_labels, mnist_features, usps_features,lmbda)

        # Backward et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Époque [{epoch + 1}/{num_epochs}], Perte: {loss.item()}')

print("Entraînement terminé")

# Sauvegarde du modèle après l'entraînement
model_path = 'lenet_mnist_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé à {model_path}")


# Fonction pour évaluer le modèle sur l'ensemble de données USPS
def evaluate_model(model, test_loader):
    model.eval()  # Mettre le modèle en mode évaluation
    correct = 0
    total = 0

    with torch.no_grad():  # Pas de calcul de gradient nécessaire
        for data in test_loader:
            images, labels = data
            images, labels = images.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Évaluation du modèle sur les données USPS
usps_accuracy = evaluate_model(loaded_model, usps_loader)
print(f'Précision sur l\'ensemble de données USPS: {usps_accuracy}%')
