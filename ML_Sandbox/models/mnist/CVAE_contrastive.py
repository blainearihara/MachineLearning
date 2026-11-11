import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F

def hello_CVAE_contrastive():
    print("Hello from CVAE_contrastive.py!")

class Encoder(nn.Module):
    def __init__(self, input_dims, hidden_dim, latent_size, num_classes):
        super(Encoder, self).__init__()
        self.C, self.W, self.H = input_dims
        self.input_size = self.C*self.W*self.H
        self.num_classes = num_classes
        self.mu_layer = None
        self.logvar_layer = None
        # flattened image [128,784] combined with one-hot vectors [128,10] before passing into the encoder
        # [128,784]+[128,10]=[128,794]->
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size + num_classes, hidden_dim), #[128,794]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU() #[128,400]->[128,400]
            )
        
        # posterior mu layer
        self.mu_layer = nn.Linear(hidden_dim, latent_size)  #[128,400]->[128,15]
        
        # posterior logvar layer
        self.logvar_layer = nn.Linear(hidden_dim, latent_size)  #[128,400]->[128,15]

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None

        # c[128,10]
        # x[128,1,28,28]->
        x = x.view(-1, self.input_size)  #[128,1,28,28]->[128,784]
        c = c.view(-1, self.num_classes) #[128,10]->[128,10]
        x = torch.cat((x, c), dim=1)  #[128,784]+[128,10]->[128,794]
        
        # Pass image through hidden dimensions
        h = self.encoder(x)  #[128,794]->[128,400]
        # Convert encoder output to posterior mu and posterior log-variance
        mu = self.mu_layer(h)  #[128,400]->[128,15]
        logvar = self.logvar_layer(h)  #[128,400]->[128,15]

        # (2) Reparametrize to compute  the latent vector z
        z = reparametrize(mu, logvar) #[128,15]->[128,15]
        z = torch.cat((z, c), dim=1)  #[128,15]+[128,10]->[128,25]
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dim, latent_size, num_classes):
        super(Decoder, self).__init__()
        # Latent space [128,15] combined with one-hot vectors [128,10] before passing into decoder
        # [128,15]+[128,10]=[128,25]->
        self.C, self.W, self.H = input_dims
        self.input_size = self.C*self.W*self.H
        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, hidden_dim), #[128,25]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, self.input_size), #[128,400]->[128,784]
            nn.Sigmoid(), #[128,784]->[128,784]
            nn.Unflatten(1, (self.C, self.W, self.H)) #->[128,1,28,28]         
            )

    def forward(self, z):
        # (3) Pass concatenation of z and one hot vectors through the decoder to resconstruct x  
        x_hat = self.decoder(z) #[128,25]->[128,784]
        return x_hat

class CVAE_contrastive(nn.Module):
    def __init__(self, input_dims, hidden_dim=400, latent_size=15, num_classes=10):
        super(CVAE_contrastive, self).__init__()
        self.encoder = Encoder(input_dims, hidden_dim, latent_size, num_classes)
        self.decoder = Decoder(input_dims, hidden_dim, latent_size, num_classes)

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c) #[128,15]->[128,1,28,28]
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    
    # init epsilon as a normal distribution
    eps = torch.randn_like(logvar)
    
    # convert logvar back to standard deviation
    sigma = torch.exp(0.5 * logvar)
    
    # scale eps by mu and sigma (std dev)
    z = sigma * eps + mu

    return z


def reconstruction_loss(x_hat, x):
    """
    Computes the reconstruction loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    
    Returns:
    - loss: Tensor containing the scalar loss for the reconstruction loss
    """
    # reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
    # reconstruction_loss = F.binary_cross_entropy(x_hat,x,reduction="mean")
    reconstruction_loss = F.binary_cross_entropy(x_hat,x,reduction="sum")
    return reconstruction_loss

def kl_divergence_loss(mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    divergence_loss = -1/2 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return divergence_loss

def supervised_contrastive_loss(z, labels, temp=0.5):
    """
    Supervised contrastive loss for a batch of latent vectors.

    Inputs:
    - z: Tensor, shape (batch_size, latent_dim), latent representations.
    - labels: Tensor, shape (batch_size,), ground truth labels.
    - temp: Float, scaling factor for the similarity scores.

    Returns:
    - loss: Tensor containing the supervised contrastive loss
    """
    # normalize latent vectors
    device = z.device
    z = F.normalize(z, dim=1)
    batch_size = z.size(0)
    
    # compute pairwise cosine similarities
    similarity_matrix = torch.matmul(z, z.T) / temp
    
    # create a mask for positive pairs (examples of the same class)
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T).float().to(device) #positive_mask.shape: (batch_size, batch_size)
    
    # exclude diagonal elements (self-similarity)
    mask = torch.eye(batch_size).bool().to(device)
    positive_mask = positive_mask.masked_fill(mask, 0) #set all diagonal elements to 0
    
    # compute positive and negative similiarity components
    exp_sim = torch.exp(similarity_matrix)
    numerator = exp_sim * positive_mask
    denominator = exp_sim.sum(dim=1, keepdim=True) - exp_sim.diagonal().unsqueeze(1)
    
    # compute supervised contrastive loss
    eps = 1e-10
    contrastive_loss = -torch.log((numerator.sum(dim=1) + eps) / (denominator + eps))
    return contrastive_loss.mean()