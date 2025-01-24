import torch
import torch.nn as nn
from data_preparation import train_dataloader, test_dataloader, val_dataloader

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initializes the MLP Encoder.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): List specifying the sizes of the hidden layers.
            latent_dim (int): Dimension of the latent space.
        """
        super(MLP, self).__init__()

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
    



class MLPModel(nn.Module):
    def __init__(self):
        """
        Initializes the MLP Model.

        Args:
            x_encoder (MLP): Encoder for the main input features.
            add_info_encoder (MLP): Encoder for the additional information.
        """
        super(MLPModel, self).__init__()
        self.x_encoder = MLP(input_dim=4, hidden_dims=[8,8,4], latent_dim=4)
        self.add_info_encoder = MLP(input_dim=73, hidden_dims=[16, 8, 8], latent_dim=4)
        
        self.decoder = MLP(input_dim=8, hidden_dims=[8, 4, 4], latent_dim=4)

    def forward(self, x, add_info):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            add_info (torch.Tensor): Additional information tensor of shape (batch_size, add_info_dim).

        Returns:
            torch.Tensor: Combined latent representation of shape (batch_size, latent_dim).
        """
        x_latent = self.x_encoder(x)
        add_info_latent = self.add_info_encoder(nn.functional.normalize(add_info))
        
        # Combine the two latent representations
        combined_latent = torch.cat((x_latent, add_info_latent), dim=1)
        
        # Decode the combined latent representation
        decoded = self.decoder(combined_latent)
        
        decoded = torch.relu(decoded)
        
        return decoded
    
# Train Loop
# Initialize the model

model = MLPModel()
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop
for epoch in range(50):
    best_loss = float('inf')
    
    for batch in train_dataloader:
        x = batch["x"]
        y = batch["y"]
        add_info = batch["additional_info"]

        # Forward pass
        outputs = model(x, add_info)
        
        # Compute loss
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Look at the validaiton loss every 10 epochs and save the model if it improves
    if epoch % 10 == 0:
        with torch.no_grad():
            val_loss = 0
            for batch in val_dataloader:
                x = batch["x"]
                y = batch["y"]
                add_info = batch["additional_info"]

                # Forward pass
                outputs = model(x, add_info)
                
                # Compute loss
                loss = criterion(outputs, y)
                val_loss += loss.item()

            avg_val_loss = (val_loss / len(val_dataloader)) * 100
            print(f'Epoch [{epoch+1}/100], Validation Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}')

    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Check the model's performance on the test set
with torch.no_grad():
    model.load_state_dict(torch.load('best_model.pth', weights_only=False))
    test_loss = 0
    for batch in test_dataloader:
        x = batch["x"]
        y = batch["y"]
        add_info = batch["additional_info"]

        # Forward pass
        outputs = model(x, add_info)
        
        print(f"Predictions: {outputs}, Actual: {y}")
        
        # Compute loss
        loss = criterion(outputs, y)
        test_loss += loss.item()

    avg_test_loss = (test_loss / len(test_dataloader)) * 100
    print(f'Test Loss: {avg_test_loss:.4f}')