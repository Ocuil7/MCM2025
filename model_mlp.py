import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initializes the MLP Encoder.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): List specifying the sizes of the hidden layers.
            latent_dim (int): Dimension of the latent space.
        """
        super(MLPEncoder, self).__init__()

        # Create a list of layers
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Add the output layer for the latent space
        layers.append(nn.Linear(prev_dim, latent_dim))

        # Combine layers into a sequential model
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        """
        return self.encoder(x)
    
