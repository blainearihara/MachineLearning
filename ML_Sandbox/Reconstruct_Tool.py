import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def one_hot(labels, class_size):
    # Create one hot label matrix of size (N, C)
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def plot_reconstruction(dataset_type, train_dataset, model_type, model, num_images, sorted=True):
    device = next(model.parameters()).device
    nrow = 10
    '''
    train_data.data unpacks as a uint8, train_loader unpacks as float32. The models are instantiated with data from DataLoader as float32, so the data needs to be typecasted as float32 before running through the model. But imshow expects data RGB values to be uint8, so after the data is reconstructed, it needs to be typecasted to a uint8

    train_data.data also unpacks as non normalized values [0,255], train_loader unpacks as normalized [0,1.0]. The models are trained from DataLoader so that it is trained on values that are normalized [0,1.0]. To reconstruct image, the input image needs to be normalized...
    '''
    if dataset_type == 'mnist':
        num_classes = 10
        # divide by 255.0 to normalize the images grayscale value, because transformed MNIST dataset is returned as normalized data
        train_data = train_dataset.data.to(device).to(torch.float32)  / 255.0
        train_data = train_data.unsqueeze(1)
        train_labels = train_dataset.targets.to(device).to(torch.int64)
    elif dataset_type == 'cifar10':
        num_classes = 10
        # divide by 255.0 to normalize the images RGB value
        train_data = torch.tensor(train_dataset.data, dtype=torch.float32, device=device)   / 255.0
        train_data = train_data.permute(0,3,1,2) #[N, C, W, H]
        train_labels = torch.tensor(train_dataset.targets, dtype=torch.int64, device=device)
        
    if sorted:
        unique_labels = np.unique(train_labels.cpu())
        ind_labels = np.array([unique_labels[i%len(unique_labels)] for i in range(num_images)])
        samples, labels = [], []
        for n in range(num_images):
            where_labels = torch.where(train_labels==ind_labels[n%len(unique_labels)])[0].cpu()
            mask = np.random.choice(where_labels)
            train_data_samples = train_data[mask]
            samples.append(train_data_samples)
            train_labels_samples = train_labels[mask]
            labels.append(train_labels_samples)
        samples = torch.stack(samples, dim=0)
        labels = torch.stack(labels, dim=0)
    else:
        shuf = np.random.permutation(train_data.shape[0])
        samples = train_data[shuf[:num_images]]
        labels = train_labels[shuf[:num_images]]
    width = nrow if (num_images // nrow > 0) else num_images % nrow
    height = (num_images // nrow + 1) * 2 #times 2 for original and reconstruction image
    # Visualization of original and reconstructed images at the end of each epoch
    with torch.no_grad():
        if model_type  == 'AE':
            reconstructed = model(samples)
        elif model_type == 'AE_contrastive':
            reconstructed = model(samples)
        if model_type  == 'AE_conv':
            reconstructed = model(samples)
        elif model_type == 'VAE':
            reconstructed, _, _ = model(samples)
        elif model_type == "VAE_4NNs":
            reconstructed, _, _ = model(samples)
        elif model_type == 'VAE_contrastive':
            reconstructed, _, _ = model(samples)
        elif model_type == 'VAE_conv':
            reconstructed, _, _ = model(samples)
        elif model_type == 'CVAE':
            one_hots = one_hot(labels, num_classes).to(device)
            reconstructed, _, _ = model(samples, one_hots)
        elif model_type == 'CVAE_contrastive':
            one_hots = one_hot(labels, num_classes).to(device)
            reconstructed, _, _ = model(samples, one_hots)

        # fig, axes = plt.subplots(height, width, figsize=(width*2, height*2))

        if dataset_type == 'mnist':
            samples = samples.squeeze().cpu()
            reconstructed = reconstructed.squeeze().cpu()
            cmap_arg = 'gray'
        elif dataset_type == 'cifar10':
            samples = samples.permute(0,2,3,1).cpu()
            reconstructed = reconstructed.permute(0,2,3,1).cpu()
            cmap_arg = None
        fig = plt.figure(figsize=(width, height))
        for i in range(num_images):
            row,col = divmod(i,nrow)
            samp_idx = row*nrow*2 + col + 1
            recon_idx = row*nrow*2 + nrow + col + 1
            # Original images
            plt.subplot(height, width, samp_idx)
            plt.imshow(samples[i], cmap=cmap_arg)
            plt.axis('off')
            plt.subplot(height, width, recon_idx)
            plt.imshow(reconstructed[i], cmap=cmap_arg)
            plt.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.0)
        plt.show()