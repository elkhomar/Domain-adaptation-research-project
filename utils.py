import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import gzip
import torch
from torchvision import datasets, transforms
from torch.nn.functional import one_hot

def one_hot_collate(batch, num_classes=10):

    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    labels = one_hot(labels, num_classes=num_classes)
    return images, labels

def load_mnist(download=True, path='./data', batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    train_dataset = datasets.MNIST(root=path, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=path, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=one_hot_collate)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=one_hot_collate)

    return train_loader, test_loader


def load_usps(data_dir, batch_size=64, one_hot=True, flatten=False):
    with gzip.open(data_dir, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    train_images = data[0][0]
    train_labels = data[0][1]
    test_images = data[1][0]
    test_labels = data[1][1]
    
    if not flatten:
        # Reshape images to add a channel dimension [num_samples, 1, 28, 28]
        train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
        test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    else:
        # Flatten the images if required
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    
    # Convert numpy arrays to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    test_images = torch.tensor(test_images, dtype=torch.float32)
    
    if one_hot:
        train_labels = group_id_2_label(train_labels, 10)
        test_labels = group_id_2_label(test_labels, 10)
    else:
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)
        
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, test_loader

def group_id_2_label(labels, num_classes):
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)



