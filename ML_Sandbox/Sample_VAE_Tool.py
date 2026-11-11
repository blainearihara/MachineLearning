import torch
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images(dataset_type, images):
    images = images.permute(0,2,3,1)
    sqrtn = int((images.shape[0])**(.5))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    plt.tight_layout()  # Reduce padding between the figure and subplots
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # Adjust borders
    cmap_arg = 'gray' if dataset_type=='mnist' else None
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img, cmap=cmap_arg)
    return 

def latent_space_interp(dataset_type, model, num_images=12, latent_size=15, conditional=False):
    device = next(model.parameters()).device
    num_classes = 10  #hard code 10 for MNIST
    latent_interp = latent_size
    z0 = torch.randn(num_images, latent_size, device=device)
    z1 = torch.randn(num_images, latent_size, device=device)
    if conditional:
        c = torch.zeros(num_images, num_classes, device=device)
        for i in torch.arange(num_images):
            c[i, i%num_classes] = 1
        z0 = torch.cat((z0, torch.flipud(c)), dim=1)
        z1 = torch.cat((z1, c), dim=1)
        latent_interp += num_classes

    w = torch.linspace(0, 1, num_images, device=device).view(num_images, 1, 1)
    z = (w * z0 + (1 - w) * z1).transpose(0, 1).reshape(num_images*num_images, latent_interp)
    x = model.decoder(z)
    show_images(dataset_type, x.data.cpu())

def sample_VAE_image(dataset_type, model, num_images=5, latent_size=64, conditional=False):
    device = next(model.parameters()).device
    num_classes = 10  #hard code 10 for MNIST
    z = torch.randn(num_images, latent_size, device=device)
    if conditional:
        c = torch.zeros(num_images, num_classes, device=device)
        for i in torch.arange(num_images):
            c[i, i%num_classes] = 1
        z = torch.cat((z, c), dim=1)
    x = model.decoder(z)
    images = x.data.cpu()

    img_dim = int(images.shape[-1]) 
    img_scale = 1.5
    width = num_classes if (num_images // num_classes > 0) else num_images % num_classes
    height = num_images // num_classes + 1
    fig = plt.figure(figsize=(width*img_scale, height*img_scale))
    gs = gridspec.GridSpec(height, width)
    gs.update(wspace=0.05, hspace=0.05)
    cmap_arg = 'gray' if dataset_type=='mnist' else None
    for i, img in enumerate(images):
        h,w = divmod(i,num_classes)
        ax = plt.subplot(gs[h,w])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.permute(1,2,0), cmap=cmap_arg)
    plt.tight_layout()  # Reduce padding between the figure and subplots
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.0)  # Adjust borders
