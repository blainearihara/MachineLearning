import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

# Ensure the local 'data' directory exists
data_path = r'C:\Users\blain\Documents\MSOL\Python ML Examples\Sandbox\ML_Sandbox'
os.chdir(data_path)
os.makedirs(data_path, exist_ok=True)

def get_dataloader(dataset_type='mnist', batch_size=2**7, plot_flag=True):
    # LOAD DATASETS AND DATALOADER
    transform = transforms.Compose([transforms.ToTensor()]) #Normalize to [0, 1]
    # add transforms.Normalize((0.1307,), (0.3081,)) to standardize to zero mean and unit variance
    if dataset_type == 'mnist':
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    elif dataset_type == 'cifar10':
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Access the training and test data
    train_data = train_dataset.data #tensor containing the training images
    train_labels = train_dataset.targets #tensor containing the training labels
    test_data = test_dataset.data #tensor containing the test images
    test_labels = test_dataset.targets #tensor containing the test labels

    if plot_flag:
        # Visualize the first 100 images
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        images_subset = images[:100]
        grid = vutils.make_grid(images_subset, nrow=10, normalize=True, padding=2)
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.tight_layout()  # Reduce padding between the figure and subplots
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # Adjust borders
        plt.show()

        #%%
        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        num_classes = len(classes)
        samples_per_class = 7
        fig = plt.figure(figsize=(8, 7))
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(np.int16(train_labels) == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(train_data[idx])
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.tight_layout()  # Reduce padding between the figure and subplots
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce space between images
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # Adjust borders
        plt.show()

    return train_dataset, train_loader