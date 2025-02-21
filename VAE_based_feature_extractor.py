import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float32)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean vector of the latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of the latent space
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):

        x = x.to(torch.float32)
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # Use sigmoid to get output in range [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the VAE loss function
def vae_loss(reconstructed_x, x, mu, logvar):
    # Reconstruction loss (binary cross entropy)
    x = x.to(torch.float32)

    recon_loss = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

input_dim = 20  # Dimension of your binary attribute vector
hidden_dim = 50  # Number of hidden units
latent_dim = 100  # Dimension of the latent space
batch_size = 32  # Batch size
learning_rate = 1e-3
num_epochs = 1000


# Initialize the VAE model
vae = VAE(input_dim, hidden_dim, latent_dim)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

PATH_train='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_4/CycleGAN_SD/dataset_train_case1.npz'

train_data = np.load(PATH_train)

train_attribute_1=train_data['training_attribute_1']
train_attribute_2=train_data['training_attribute_2']
train_attribute_3=train_data['training_attribute_3']
train_attribute_4=train_data['training_attribute_4']
train_attribute_5=train_data['training_attribute_5']
train_attribute_6=train_data['training_attribute_6']
train_attribute_7=train_data['training_attribute_7']
train_attribute_8=train_data['training_attribute_8']
train_attribute_9=train_data['training_attribute_9']
train_attribute_10=train_data['training_attribute_10']
train_attribute_11=train_data['training_attribute_11']
train_attribute_12=train_data['training_attribute_12']
train_attribute_13=train_data['training_attribute_13']
train_attribute_14=train_data['training_attribute_14']
train_attribute_15=train_data['training_attribute_15']

train_attribute=np.concatenate([train_attribute_1, 
                        train_attribute_2,
                        train_attribute_3,
                        train_attribute_4,
                        train_attribute_5,
                        train_attribute_6,
                        train_attribute_7,
                        train_attribute_8,
                        train_attribute_9,
                        train_attribute_10,
                        train_attribute_11,
                        train_attribute_12,
                        train_attribute_13,
                        train_attribute_14,
                        train_attribute_15                       
                        ], axis=0)


data_loader = DataLoader(train_attribute, batch_size=batch_size, shuffle=True)

Q = 0
# Training loop

if Q != 0:

    for epoch in range(num_epochs):
    
        vae.train()
        total_loss = 0
        for batch in data_loader:
            # Get the batch data
            batch_data = batch
 
            reconstructed, mu, logvar = vae(batch_data)

            # Compute the loss
            loss = vae_loss(reconstructed, batch_data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the loss at the end of each epoch
        print(f'Epoch {epoch}, Loss: {total_loss/len(data_loader.dataset)}')

    vae.eval()
    with torch.no_grad():

        x_eval=train_attribute_1[0]
    
        x_eval = torch.tensor(x_eval)  # First convert to tensor
    
        x_eval = x_eval.to(torch.float32)

        mu, _ = vae.encode(x_eval)

        print("Latent feature space representation:", mu)

    torch.save(vae, '/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_4/CycleGAN_SD/vae.h5')


