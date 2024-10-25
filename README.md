# Adversarial Auto-encoder (AAE)

An implementation of an Adversarial Auto-encoder for butterfly image generation using PyTorch. This model combines Variational Autoencoders (VAE) with Generative Adversarial Networks (GAN) to create a powerful generative model.

## Architecture Overview

The model consists of three main components:
- **Encoder**: Converts input images (128x128x3) into latent vectors
- **Decoder**: Reconstructs images from latent vectors 
- **Discriminator**: Distinguishes between real and generated latent vectors

### Key Features
- Latent space dimension: 100
- Image size: 128x128 RGB
- Adversarial training with GAN-based latent regularization
- Progressive upsampling in decoder using nearest neighbor interpolation
- Batch normalization and layer normalization for stable training

## Requirements
- torch
- torchvision
- PIL
- numpy
- pandas
- matplotlib
- tqdm
- torchsummary

## Model Architecture Details

### Encoder
- 5 convolutional layers with increasing channel dimensions
- LeakyReLU activation and batch normalization
- Outputs mean and log variance of latent distribution

### Decoder
- Initial projection followed by 5 upsampling blocks
- Progressive upsampling from 5x5 to 128x128
- Layer normalization and LeakyReLU activation
- Tanh activation for final output

### Discriminator
- Simple fully connected network
- 64 hidden units
- Sigmoid activation for binary classification

## Training

The model is trained using:
- Adam optimizer with learning rates:
  - Encoder/Decoder: 1e-4
  - Discriminator: 1e-5
- Batch size: 128
- Number of epochs: 500
- MSE reconstruction loss
- Adversarial loss for latent space regularization

## Results

The training process saves:
- Generated images every 200 iterations
- Model checkpoints every 25 epochs
- Loss curves for reconstruction, generator, and discriminator losses


## Usage

1. Clone the repository
2. Install required dependencies
3. Run the Jupyter notebook
4. Model weights and generated images will be saved in the `AAE/zdim = 100/` directory
