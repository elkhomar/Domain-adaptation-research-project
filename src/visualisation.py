import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os

def visualize_image(image, label):
    """
    Visualizes a single image and its label.

    Parameters:
    - image (Tensor): The image to visualize.
    - label (Tensor): The label of the image.
    """
    plt.imshow(image.to("cpu").squeeze(), cmap='gray')
    plt.title(f'Label: {label.to("cpu").item()}')
    plt.show()



def visualize_tsne(source_embedding, target_embedding, source_labels, target_labels, filename=None, root_dir=None):
    """
    Visualizes the t-SNE of the source and target embeddings coloring by class.

    Parameters:
    - source_features (Tensor): The source features.
    - target_features (Tensor): The target features.
    - source_labels (Tensor): The source labels.
    - target_labels (Tensor): The target labels.
    """
    # Concatenate the source and target features and labels
    features = torch.cat([source_embedding, target_embedding], dim=0).to("cpu").detach().numpy()
    labels = torch.argmax(torch.cat([source_labels, target_labels], dim=0).to("cpu").detach(), 1).numpy()
    domain = np.concatenate((np.zeros(source_embedding.shape[0]), np.ones(target_embedding.shape[0])))

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    transformed = tsne.fit_transform(features)

    # Create a DataFrame for visualization
    data = {
        'x': transformed[:, 0],
        'y': transformed[:, 1],
        'label': labels,
        'domain': domain
    }
    df = pd.DataFrame(data)

    # Visualize Tsne labels
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', hue='label', palette="deep", data=df)
    plt.title('t-SNE_label')

    if filename and root_dir:
        directory = root_dir + "/visualisations/label_tsne/"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + filename + "_label.png")
    else:
        plt.show()

    # Visualize Tsne domain
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', hue='domain', data=df)
    plt.title('t-SNE_domain')

    if filename and root_dir:
        directory = root_dir + "/visualisations/domain_tsne/"
        os.makedirs(directory, exist_ok=True)
        plt.savefig(directory + filename + "_domain.png")
    else:
        plt.show()


if __name__ == "__main__":
    test_inputs = torch.load("inputs_vis")
    visualize_tsne(test_inputs[0], test_inputs[1])