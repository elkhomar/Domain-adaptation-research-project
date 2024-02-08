import pickle
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
    labels = labels.float()
    return images, labels

def load_mnist(download=True, path='./data', batch_size=64):
    """
    Loads the MNIST dataset and returns DataLoader objects for the train and test sets.

    Parameters:
    - download (bool, optional): If True, downloads the dataset from the internet if it's not available at `path`. Default is True.
    - path (str, optional): The path where the MNIST dataset is stored or will be downloaded. Default is './data'.
    - batch_size (int, optional): The size of each batch returned by the DataLoader. Default is 64.

    Returns:
    - tuple: A tuple containing two DataLoader instances:
        - train_loader (DataLoader): DataLoader for the training set.
        - test_loader (DataLoader): DataLoader for the test set.
    """
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
    """
    Loads the USPS dataset from a pickle file, processes it, and returns DataLoader objects for the train and test sets.

    Parameters:
    - data_dir (str): The directory path where the USPS dataset pickle file is stored.
    - batch_size (int, optional): The size of each batch returned by the DataLoader. Default is 64.
    - one_hot (bool, optional): If True, converts labels into one-hot encoded format. Default is True.
    - flatten (bool, optional): If True, flattens the images into 1D tensors. Otherwise, images are reshaped to 2D tensors (1, 28, 28). Default is False.

    Returns:
    - tuple: A tuple containing two DataLoader instances:
        - train_loader (DataLoader): DataLoader for the training set.
        - test_loader (DataLoader): DataLoader for the test set.
    """
        
    with gzip.open(data_dir, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    train_images = data[0][0]
    train_labels = data[0][1]
    test_images = data[1][0]
    test_labels = data[1][1]
    
    if not flatten:
        train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
        test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    else:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    
    train_images = torch.tensor(train_images, dtype=torch.float32)
    test_images = torch.tensor(test_images, dtype=torch.float32)
    
    if one_hot:
        train_labels = group_id_2_label(train_labels, 10)
        test_labels = group_id_2_label(test_labels, 10)
    else:
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)
        
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return train_loader, test_loader

def group_id_2_label(labels, num_classes):
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)


