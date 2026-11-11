# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def hello_AE_conv():
    print("Hello from AE_conv.py!")

def calc_dims(C_in, H_in, W_in, K_height, K_width, S_height, S_width, P_height, P_width):
             C_out = C_in
             H_out = int(1+(H_in-K_height+2*P_height)/S_height)
             W_out = int(1+(W_in-K_width+2*P_width)/S_width)
             return C_out, H_out, W_out

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, Encoder_Arg, Decoder_Arg):
        super(Encoder, self).__init__()
        input_dims = Encoder_Arg['input_dims']
        C1,H1,W1 = input_dims
        Enc_Chan_out = Encoder_Arg['Chan_out']
        Enc_Kern_Conv = Encoder_Arg['Kern_Conv']
        Enc_Stride_Conv = Encoder_Arg['Stride_Conv']
        # Enc_Stride_Pool = Encoder_Arg['Stride_Pool']
        Enc_Pad = Encoder_Arg['Pad']
        latent_size = Encoder_Arg['latent_size']

        # init encoder using the defined architecture
        # input image [128,1,28,28]->
        C2,H2,W2 = calc_dims(Enc_Chan_out[0], H1, W1, Enc_Kern_Conv[0], Enc_Kern_Conv[1], Enc_Stride_Conv[0], Enc_Stride_Conv[1], Enc_Pad[0], Enc_Pad[1])
        C3,H3,W3 = calc_dims(Enc_Chan_out[1], H2, W2, Enc_Kern_Conv[0], Enc_Kern_Conv[1], Enc_Stride_Conv[0], Enc_Stride_Conv[1], Enc_Pad[0], Enc_Pad[1])
        self.encoder = nn.Sequential(
            nn.Conv2d(C1, Enc_Chan_out[0], kernel_size=(Enc_Kern_Conv[0],Enc_Kern_Conv[1]), stride=(Enc_Stride_Conv[0],Enc_Stride_Conv[1]), padding=(Enc_Pad[0],Enc_Pad[1]), bias=True), #[128,784]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Conv2d(Enc_Chan_out[0], Enc_Chan_out[1], kernel_size=(Enc_Kern_Conv[0],Enc_Kern_Conv[1]), stride=(Enc_Stride_Conv[0],Enc_Stride_Conv[1]), padding=(Enc_Pad[0],Enc_Pad[1]), bias=True), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Flatten()
            )
        
        # alternate way to dynamically compute the output dimension
        # with torch.no_grad():
        #      dummy_input = torch.zeros(1, *input_dims)
        #      dummy_output = self.encoder(dummy_input)
        #      flattened_size = dummy_output.shape[1]

        # init mu layer
        self.fc = nn.Linear(C3*H3*W3, latent_size)  #[128,400]->[128,15]

    def forward(self, x):
        h = self.encoder(x) 
        z = self.fc(h)
        return z

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, Encoder_Arg, Decoder_Arg):
        super(Decoder, self).__init__()
        input_dims = Encoder_Arg['input_dims']
        C1,H1,W1 = input_dims
        Enc_Chan_out = Encoder_Arg['Chan_out']
        Enc_Kern_Conv = Encoder_Arg['Kern_Conv']
        Enc_Stride_Conv = Encoder_Arg['Stride_Conv']
        # Enc_Stride_Pool = Encoder_Arg['Stride_Pool']
        Enc_Pad = Encoder_Arg['Pad']
        latent_size = Encoder_Arg['latent_size']
        C2,H2,W2 = calc_dims(Enc_Chan_out[0], H1, W1, Enc_Kern_Conv[0], Enc_Kern_Conv[1], Enc_Stride_Conv[0], Enc_Stride_Conv[1], Enc_Pad[0], Enc_Pad[1])
        C3,H3,W3 = calc_dims(Enc_Chan_out[1], H2, W2, Enc_Kern_Conv[0], Enc_Kern_Conv[1], Enc_Stride_Conv[0], Enc_Stride_Conv[1], Enc_Pad[0], Enc_Pad[1])

        Dec_Chan_out = Decoder_Arg['Chan_out']
        Dec_Kern_Conv = Decoder_Arg['Kern_Conv']
        Dec_Stride_Conv = Decoder_Arg['Stride_Conv']
        # Dec_Stride_Pool = Decoder_Arg['Stride_Pool']
        Dec_Pad = Decoder_Arg['Pad']

        self.C, self.W, self.H = input_dims
        self.input_size = self.C*self.W*self.H

        self.fc = nn.Sequential(
             nn.Linear(latent_size, C3*H3*W3),
             nn.ReLU(),
             nn.Unflatten(1, (C3, H3, W3))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(Dec_Chan_out[0], Dec_Chan_out[1], kernel_size=(Dec_Kern_Conv[0], Dec_Kern_Conv[1]), stride=(Dec_Stride_Conv[0], Dec_Stride_Conv[1]), padding=(Dec_Pad[0], Dec_Pad[1])),
            nn.ReLU(), #[128,400]->[128,400]
            nn.ConvTranspose2d(Dec_Chan_out[1], C1, kernel_size=(Dec_Kern_Conv[0], Dec_Kern_Conv[1]), stride=(Dec_Stride_Conv[0], Dec_Stride_Conv[1]), padding=(Dec_Pad[0], Dec_Pad[1])),
            nn.Sigmoid(), #->[128,784]  
            )

    def forward(self, z):
        h = self.fc(z)
        x_hat = self.decoder(h)
        return x_hat

# Combine Encoder and Decoder into Autoencoder
class AE_conv(nn.Module):
    def __init__(self, input_dims, latent_size=32):
        super(AE_conv, self).__init__()
        Encoder_Arg = {
            'input_dims': input_dims,
            'Chan_out': [32, 64],
            'Kern_Conv': [4, 4],
            'Stride_Conv': [2, 2],
            'Stride_Pool': [1, 1],
            'Pad': [1, 1],
            'latent_size': latent_size
        }
        Decoder_Arg = {
            'Chan_out': [64, 32],
            'Kern_Conv': [4, 4],
            'Stride_Conv': [2, 2],
            'Stride_Pool': [1, 1],
            'Pad': [1, 1]
        }
        self.encoder = Encoder(Encoder_Arg, Decoder_Arg)
        self.decoder = Decoder(Encoder_Arg, Decoder_Arg)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
def loss_function(x_hat, x):
    loss_function = nn.MSELoss()
    loss = loss_function(x_hat, x)
    return loss

# Confirm model import to drive
print("Successfully imported AE")

"""
**Vanilla Autoencoder:**
  - The vanilla autoencoder is the baseline version of autoencoder that consists of the basic encoder and decoder components that respectively transform the higher-dimensional input images into lower-dimensional representations. Starting with the MNIST dataset, which are 28 by 28 greyscale (single channel dimension) images, the encoder flattens the input into a 784 dimensional vector and then runs the vector through two dimensional reducing linear layers with rectified linear unit (ReLU) activation functions to transform the image into a 64 dimension latent space vector. The decoder reverses the process by taking in the 64 dimensional latent space vectors and dimensionally increases the image back to its 784 dimensional vector before being reshaped back to its 28 by 28 pixel image.
  - The mean squared error (MSE) reconstruction loss is used to train the model such that the averaged squared errors between the reconstructed image and the original image is minimized.
  $$MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y_i})^2$$
  - t-Distributed Stochastic Neighbor Embedding (tSNE) is a dimensional reduction technique that attempts to preserve higher dimensional structure in a lower dimension typically for visualization. The tSNE plot below is a 2-dimensional visualization of the 64-dimensional latent vectors of the 10 classes of the MNIST dataset color-coded by class. It can seen that the encoder is transforming examples of the same class in roughly similar locations in latent space, while placing examples from different classes in different locations in latent space. This plot gives the intution of the latent encodings of different classes that the GMM is attempting to decompose into multiple gaussian distributions.
"""