#%% Import necessary libraries
import torch
import torch.optim as optim
from torchsummary import summary
import os
import numpy as np
import importlib
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def one_hot(labels, class_size):
    # Create one hot label matrix of size (N, C)
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def get_model(train_dataset, train_loader, dataset_type, model_type, num_epochs=10, latent_size=64, num_latents=1000):

    # Set model architecture
    C, W, H = train_dataset[0][0].shape
    input_dims = [C, W, H]
    hidden_dim = 128
    num_classes = len(np.unique(train_dataset.targets))
    # AE: autoencoder
    # AE_contrastive: AE with contrastive learning
    # VAE: variational autoencoder
    # VAE_4NNs: variational autoencoder that uses parameterized NNs in the decoder to distribution of output images from which to sample an output
    # VAE_contrastive: VAE with contrastive learning
    # CVAE: conditional variational autoencoder
    # CVAE_contrastive: conditional variational autoencoder with contrastive learning

    if model_type == 'AE':
        module_name = f'models.{dataset_type}.AE'
        AE = importlib.import_module(module_name)
        AE.hello_AE()
        # Initialize autoencoder, loss, and optimizer
        model = AE.Autoencoder(input_dims, hidden_dim, latent_size).to(device)
        loss_function = AE.loss_function
        summary(model, input_size=(C, W, H))
    elif model_type == 'AE_contrastive':
        module_name = f'models.{dataset_type}.AE_contrastive'
        AE_contrastive = importlib.import_module(module_name)
        AE_contrastive.hello_AE_contrastive()
        # Initialize variational autoencoder, loss, and optimizer
        model = AE_contrastive.AE_contrastive(input_dims, hidden_dim, latent_size).to(device)
        reconstruction_loss = AE_contrastive.reconstruction_loss
        supervised_contrastive_loss = AE_contrastive.supervised_contrastive_loss
        summary(model, input_size=(C, W, H))
    elif model_type == 'AE_conv':
        module_name = f'models.{dataset_type}.AE_conv'
        AE_conv = importlib.import_module(module_name)
        AE_conv.hello_AE_conv()
        # Initialize autoencoder, loss, and optimizer
        model = AE_conv.AE_conv(input_dims, latent_size).to(device)
        loss_function = AE_conv.loss_function
        summary(model, input_size=(C, W, H))
    elif model_type == 'VAE':
        module_name = f'models.{dataset_type}.VAE'
        VAE = importlib.import_module(module_name)
        VAE.hello_VAE()
        # Initialize variational autoencoder, loss, and optimizer
        model = VAE.VAE(input_dims, hidden_dim, latent_size).to(device)
        loss_function = VAE.loss_function
        summary(model, input_size=(C, W, H))
    elif model_type == 'VAE_4NNs':
        module_name = f'models.{dataset_type}.VAE_4NNs'
        VAE_4NNs = importlib.import_module(module_name)
        VAE_4NNs.hello_VAE_4NNs()
        # Initialize variational autoencoder, loss, and optimizer
        model = VAE_4NNs.VAE_4NNs(input_dims, hidden_dim, latent_size).to(device)
        loss_function = VAE_4NNs.loss_function
        summary(model, input_size=(C, W, H))
    elif model_type == 'VAE_conv':
        module_name = f'models.{dataset_type}.VAE_conv'
        VAE_conv = importlib.import_module(module_name)
        VAE_conv.hello_VAE_conv()
        # Initialize variational autoencoder, loss, and optimizer
        model = VAE_conv.VAE_conv(input_dims, latent_size).to(device)
        loss_function = VAE_conv.loss_function
        summary(model, input_size=(C, W, H))
    elif model_type == 'CVAE':
        module_name = f'models.{dataset_type}.CVAE'
        CVAE = importlib.import_module(module_name)
        CVAE.hello_CVAE()
        # Initialize conditional variational autoencoder, loss, and optimizer
        model = CVAE.CVAE(input_dims, hidden_dim, latent_size, num_classes).to(device)
        CVAEwrap = CVAE.wrap_CVAE(model, num_classes=num_classes)
        loss_function = CVAE.loss_function
        # summary(CVAEwrap, input_size=(C, W, H))
    elif model_type == 'VAE_contrastive':
        module_name = f'models.{dataset_type}.VAE_contrastive'
        VAE_contrastive = importlib.import_module(module_name)
        VAE_contrastive.hello_VAE_contrastive()
        # Initialize variational autoencoder, loss, and optimizer
        model = VAE_contrastive.VAE_contrastive(input_dims, hidden_dim, latent_size).to(device)
        reparametrize = VAE_contrastive.reparametrize
        reconstruction_loss = VAE_contrastive.reconstruction_loss
        kl_divergence_loss = VAE_contrastive.kl_divergence_loss
        supervised_contrastive_loss = VAE_contrastive.supervised_contrastive_loss
        summary(model, input_size=(C, W, H))
    elif model_type == 'CVAE_contrastive':
        module_name = f'models.{dataset_type}.CVAE_contrastive'
        CVAE_contrastive = importlib.import_module(module_name)
        CVAE_contrastive.hello_CVAE_contrastive()
        # Initialize variational autoencoder, loss, and optimizer
        model = CVAE_contrastive.CVAE_contrastive(input_dims, hidden_dim, latent_size, num_classes).to(device)
        reparametrize = CVAE_contrastive.reparametrize
        reconstruction_loss = CVAE_contrastive.reconstruction_loss
        kl_divergence_loss = CVAE_contrastive.kl_divergence_loss
        supervised_contrastive_loss = CVAE_contrastive.supervised_contrastive_loss
        # summary(model, input_size=(C, W, H))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            # Forward pass
            if model_type == 'AE':
                x_hat = model(images)
                loss = loss_function(x_hat, images)
            elif model_type == 'AE_contrastive':
                rcon_weight, c_weight = 1, 0.05 #.01 worked well
                x_hat = model(images)
                latent = model.encoder(images)
                recon_loss = reconstruction_loss(x_hat, images)
                cont_loss = supervised_contrastive_loss(latent, labels)
                loss = rcon_weight*recon_loss + c_weight*cont_loss
            elif model_type == 'AE_conv':
                x_hat = model(images)
                loss = loss_function(x_hat, images)
            elif model_type == 'VAE':
                x_hat, mu, logvar = model(images)
                loss = loss_function(x_hat, images, mu, logvar, beta=1)
            elif model_type == 'VAE_4NNs':
                x_hat, mu, logvar = model(images)
                loss = loss_function(x_hat, images, mu, logvar, beta=1)
            elif model_type == 'VAE_conv':
                x_hat, mu, logvar = model(images)
                loss = loss_function(x_hat, images, mu, logvar, beta=1)
            elif model_type == 'CVAE':
                one_hot_encodings = one_hot(labels, num_classes).to(device)
                x_hat, mu, logvar = model(images, one_hot_encodings)
                loss = loss_function(x_hat, images, mu, logvar, beta=1)
            elif model_type == 'VAE_contrastive':
                rcon_weight, kl_weight, c_weight = 1, 1, 25 #20 worked well
                batch_size = images.shape[0]
                x_hat, mu, logvar = model(images)
                z = reparametrize(mu, logvar)
                recon_loss = reconstruction_loss(x_hat, images)
                kl_loss = kl_divergence_loss(mu, logvar)
                cont_loss = supervised_contrastive_loss(z, labels)
                loss = (rcon_weight*recon_loss + kl_weight*kl_loss)/batch_size + c_weight*cont_loss
            elif model_type == 'CVAE_contrastive':
                rcon_weight, kl_weight, c_weight = 1, 1, 25 #20 worked well
                one_hot_encodings = one_hot(labels, num_classes).to(device)
                batch_size = images.shape[0]
                x_hat, mu, logvar = model(images, one_hot_encodings)
                z = reparametrize(mu, logvar)
                recon_loss = reconstruction_loss(x_hat, images)
                kl_loss = kl_divergence_loss(mu, logvar)
                cont_loss = supervised_contrastive_loss(z, labels)
                loss = (rcon_weight*recon_loss + kl_weight*kl_loss)/batch_size + c_weight*cont_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Visualization of original and reconstructed images at the end of each epoch
        # with torch.no_grad():
        #     sample = images[:5]  # Take 5 examples
        #     if model_type == 'AE':
        #         reconstructed = model(sample)
        #     elif model_type == 'AE_contrastive':
        #         reconstructed = model(sample)
        #     elif model_type == 'AE_conv':
        #         reconstructed = model(sample)
        #     elif model_type == 'VAE':
        #         reconstructed, _, _ = model(sample)
        #     elif model_type == 'VAE_4NNs':
        #         reconstructed, _, _ = model(sample)
        #     elif model_type == 'VAE_contrastive':
        #         reconstructed, _, _ = model(sample)
        #     elif model_type == 'VAE_conv':
        #         reconstructed, _, _ = model(sample)
        #     elif model_type == 'CVAE':
        #         one_hots = one_hot(labels[:5], num_classes).to(device)
        #         reconstructed, _, _ = model(sample, one_hots)
        #     elif model_type == 'CVAE_contrastive':
        #         one_hots = one_hot(labels[:5], num_classes).to(device)
        #         reconstructed, _, _ = model(sample, one_hots)

        #     if dataset_type == 'mnist':
        #         sample = sample.squeeze().cpu()
        #         reconstruction = reconstruction.squeeze().cpu()
        #         cmap_arg = 'gray'
        #     elif dataset_type == 'cifar10':
        #         sample = sample.permute(0,2,3,1).cpu()
        #         reconstructed = reconstructed.permute(0,2,3,1).cpu()
        #         cmap_arg = None
        #     fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        #     for i in range(5):
        #         # Original images
        #         axes[0, i].imshow(sample[i], cmap=cmap_arg)
        #         axes[0, i].axis('off')
        #         # Reconstructed images
        #         imshow_data = reconstructed[i]
        #         axes[1, i].imshow(imshow_data, cmap=cmap_arg)
        #         axes[1, i].axis('off')
        #     plt.tight_layout()
        #     fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.0)    
        #     plt.show()

    #%% Plot the VAE and CVAE sampled images
    # tools.latent_space_interp(model, latent_size=latent_size)
    # if model_type=='VAE' or model_type=='VAE_contrastive' or model_type=='CVAE' or model_type=='CVAE_contrastive':
    #     conditional = True if model_type == 'CVAE' or model_type == 'CVAE_contrastive' else False 
    #     tools.sample_VAE_image(model, num_images=50, latent_size=latent_size, conditional=conditional)
    #%% Extract latent representations and their corresponding labels
    latent_representations = []
    labels = []
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # No need to calculate gradients during extraction
        for images, label_batch in train_loader:
            images = images.to(device)
            if model_type == "AE":
                encoded = model.encoder(images)
            elif model_type == "AE_contrastive":
                encoded = model.encoder(images)
            elif model_type == "AE_conv":
                encoded = model.encoder(images)
            elif model_type == "VAE":
                encoded,_,_ = model.encoder(images)
            elif model_type == "VAE_4NNs":
                encoded,_,_ = model.encoder(images)
            elif model_type == "VAE_contrastive":
                encoded,_,_ = model.encoder(images)
            elif model_type == "VAE_conv":
                encoded,_,_ = model.encoder(images)
            elif model_type == "CVAE":
                one_hots = one_hot(label_batch, num_classes).to(device)
                encoded,_,_ = model.encoder(images, one_hots)
            elif model_type == "CVAE_contrastive":
                one_hots = one_hot(label_batch, num_classes).to(device)
                encoded,_,_ = model.encoder(images, one_hots)
            latent_representations.append(encoded.cpu().numpy())
            labels.extend(label_batch.numpy())

    #%% Plot the tSNE plots
    latent_representations = np.concatenate(latent_representations, axis=0)
    labels = np.array(labels)
    print('data:', latent_representations.shape)
    print('labels:', labels.shape)
    # Randomly sample 1000 point

    indices = np.random.choice(latent_representations.shape[0], num_latents, replace=False)
    latent_representations_sample = latent_representations[indices]
    labels_sample = labels[indices]
    # tools.plot_tsne(latent_representations_sample, labels_sample, plt_dim=2)

    return model, latent_representations_sample, labels_sample