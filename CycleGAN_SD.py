import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from VAE_based_feature_extractor import VAE

class Discriminator(nn.Module):

    def __init__(self,sample_shape, attribute_size):

        super(Discriminator, self).__init__()
               
        # Flatten and concatenate handled in forward

        self.fc1 = nn.Linear(sample_shape + attribute_size, 36)
        self.norm1 = nn.LayerNorm(36)
        self.fc2 = nn.Linear(36, 18)
        self.norm2 = nn.LayerNorm(18)
        self.output_layer = nn.Linear(18, 1)
    
    def forward(self, sample_input, attribute):

        concatenated=torch.cat((sample_input, attribute),dim=1)

        # Forward pass through the network

        d1 = self.fc1(concatenated)
        d1 = F.leaky_relu(d1, 0.2)
        d1 = self.norm1(d1)

        d2 = self.fc2(d1)
        d2 = F.leaky_relu(d2, 0.2)
        d2 = self.norm2(d2)

        validity = self.output_layer(d2)

        return validity

class Generator(nn.Module):

    def __init__(self,sample_shape, semantic_feature_shape):
        
        super(Generator, self).__init__()

        # Flatten and concatenate handled in forward

        self.fc1 = nn.Linear(sample_shape + semantic_feature_shape, 128)
        self.norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)

        self.fc3 = nn.Linear(64, 52)
        self.norm3 = nn.BatchNorm1d(52)

    def forward(self,sample_input,semantic_feature_1,semantic_feature_2):

        semantic_distance=semantic_feature_1-semantic_feature_2

        concatenated=torch.cat((sample_input, semantic_distance),dim=1)

        # Forward pass through the network
        
        d1 = self.fc1(concatenated)
        d1 = F.leaky_relu(d1, 0.2)
        d1 = self.norm1(d1)

        d2 = self.fc2(d1)
        d2 = F.leaky_relu(d2, 0.2)
        d2 = self.norm2(d2)

        d3 = self.fc3(d2)
        d3 = F.leaky_relu(d3, 0.2)
        d3 = self.norm3(d3)

        return d3
    

class Regressor(nn.Module):

    def __init__(self,sample_shape):

        super(Regressor, self).__init__()

        self.fc1 = nn.Linear(sample_shape,40)

        self.fc2 = nn.Linear(40, 30)

        self.fc3 = nn.Linear(30, 20)
    
    def forward(self,sample_input):

        d1 = self.fc1(sample_input)
        d1 = F.leaky_relu(d1, 0.2)

        d2 = self.fc2(d1)
        d2 = F.leaky_relu(d2, 0.2)

        d3 = self.fc3(d2)
        d3 = torch.sigmoid(d3)

        hidden_output = d2
        predicted_attribute = d3

        return predicted_attribute, hidden_output

class CycleGAN_SD(nn.Module):
    
    def __init__(self, sample_shape, attribute_size, semantic_feature_shape):
        
        super(CycleGAN_SD, self).__init__()

        self.G = Generator(sample_shape, semantic_feature_shape)  # G: source to target 
        self.F = Generator(sample_shape, semantic_feature_shape)  # F: target to source 

        self.D1 = Discriminator(sample_shape, attribute_size) # D1: for source
        self.D2 = Discriminator(sample_shape, attribute_size) # D2: for target

        self.R = Regressor(sample_shape)
    
    def forward(self,source_sample, source_attribute, target_sample, target_attribute,source_semantic_feature,target_semantic_feature):

        fake_target_sample = self.G(source_sample,target_semantic_feature,source_semantic_feature)
        fake_source_sample = self.F(target_sample,source_semantic_feature,target_semantic_feature)

        recon_source_sample = self.F(fake_target_sample,source_semantic_feature,target_semantic_feature)
        recon_target_sample = self.G(fake_source_sample,target_semantic_feature,source_semantic_feature)

        real_source_sample_validity = self.D1(source_sample,source_attribute)
        fake_source_sample_validity = self.D1(fake_source_sample,source_attribute)

        real_target_sample_validity = self.D2(target_sample,target_attribute)
        fake_target_sample_validity = self.D2(fake_target_sample,target_attribute)

        fake_target_predicted_attribute, _ = self.R(fake_target_sample)
        fake_source_predicted_attribute, _ = self.R(fake_source_sample)

        real_target_predicted_attribute, _ = self.R(target_sample)
        real_source_predicted_attribute, _ = self.R(source_sample)
 
        return fake_target_sample, fake_source_sample, recon_source_sample, recon_target_sample, fake_target_predicted_attribute, fake_source_predicted_attribute,real_target_predicted_attribute,real_source_predicted_attribute,real_source_sample_validity,fake_source_sample_validity,real_target_sample_validity,fake_target_sample_validity


def wasserstein_loss_discriminator(fake_validity, real_validity):
    return fake_validity.mean()-real_validity.mean()

def wasserstein_loss_generator(D, fake,attribute):
    return -D(fake,attribute).mean()

criterion_cycle = nn.L1Loss()

def cycle_loss(real, reconstructed):
    return criterion_cycle(reconstructed, real)

def regressor_loss(predicted_atrribute, true_attribute):

    loss = nn.functional.binary_cross_entropy(predicted_atrribute, true_attribute, reduction='mean')

    return loss

def compute_gradient_penalty(D, real_samples, fake_samples, attribute):

    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates,attribute)

    fake = torch.autograd.Variable(torch.ones(d_interpolates.size(), device=real_samples.device), requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty



sample_shape = 52 
attribute_size = 20  # Number of hidden units
semantic_feature_shape = 100  # Dimension of the latent space
batch_size = 32  # Batch size
learning_rate = 1e-3
num_epochs = 1000

cycle_lambda = 10
ac_lambda = 10

model=CycleGAN_SD(sample_shape,attribute_size,semantic_feature_shape)

optimizer_GF = optim.Adam(list(model.G.parameters()) + list(model.F.parameters()), lr=0.0002, betas=(0.5, 0.999))

optimizer_D1 = optim.Adam(model.D1.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D2 = optim.Adam(model.D2.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_R = optim.Adam(model.R.parameters(), lr=0.0002, betas=(0.5, 0.999))

PATH_train='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_4/dataset_train_case1.npz'
train_data = np.load(PATH_train)

train_sample_1=train_data['training_samples_1']
train_sample_2=train_data['training_samples_2']
train_sample_3=train_data['training_samples_3']
train_sample_4=train_data['training_samples_4']
train_sample_5=train_data['training_samples_5']
train_sample_6=train_data['training_samples_6']
train_sample_7=train_data['training_samples_7']
train_sample_8=train_data['training_samples_8']
train_sample_9=train_data['training_samples_9']
train_sample_10=train_data['training_samples_10']
train_sample_11=train_data['training_samples_11']
train_sample_12=train_data['training_samples_12']
train_sample_13=train_data['training_samples_13']
train_sample_14=train_data['training_samples_14']
train_sample_15=train_data['training_samples_15']

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

source_train_sample=np.concatenate([train_sample_1, 
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1,
                        train_sample_1                                             
                        ], axis=0)

source_train_attribute=np.concatenate([train_attribute_1, 
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1,
                        train_attribute_1                                           
                        ], axis=0)





target_train_sample=np.concatenate([#train_sample_1, 
                        train_sample_2,
                        train_sample_3,
                        train_sample_4,
                        train_sample_5,
                        train_sample_6,
                        train_sample_7,
                        train_sample_8,
                        train_sample_9,
                        train_sample_10,
                        train_sample_11,
                        train_sample_12,
                        train_sample_13,
                        train_sample_14,
                        train_sample_15                       
                        ], axis=0)

target_train_attribute=np.concatenate([#train_attribute_1, 
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

VAE=torch.load('/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD_4/vae.h5')

num_batches=int(source_train_sample.shape[0]/batch_size)

torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):

    for batch_i in range(num_batches):

        start_i =batch_i * batch_size
        end_i=(batch_i + 1) * batch_size
        
        source_train_x=source_train_sample[start_i:end_i]
        source_train_y=source_train_attribute[start_i:end_i]

        target_train_x=target_train_sample[start_i:end_i]
        target_train_y=target_train_attribute[start_i:end_i]

        source_train_x=torch.tensor(source_train_x, dtype=torch.float32)
        source_train_y=torch.tensor(source_train_y, dtype=torch.float32)

        target_train_x=torch.tensor(target_train_x, dtype=torch.float32)
        target_train_y=torch.tensor(target_train_y, dtype=torch.float32)

        source_semantic_feature,_=VAE.encode(source_train_y)
        target_semantic_feature,_=VAE.encode(target_train_y)
        
        # training Generators
        optimizer_GF.zero_grad()

        fake_target_sample_1, fake_source_sample_1, recon_source_sample_1, recon_target_sample_1,fake_target_predicted_attribute_1, fake_source_predicted_attribute_1, real_target_predicted_attribute_1,real_source_predicted_attribute_1,real_source_sample_validity_1,fake_source_sample_validity_1,real_target_sample_validity_1,fake_target_sample_validity_1= model(source_train_x,source_train_y,target_train_x,target_train_y,source_semantic_feature,target_semantic_feature)
        
        G_adv_loss = wasserstein_loss_generator(model.D2, fake_target_sample_1, target_train_y)
        F_adv_loss = wasserstein_loss_generator(model.D1, fake_source_sample_1, source_train_y)

        GF_cycle_loss = cycle_loss(source_train_x, recon_source_sample_1)
        FG_cycle_loss = cycle_loss(target_train_x, recon_target_sample_1)
        
        G_ac_loss = regressor_loss(fake_target_predicted_attribute_1,target_train_y)
        F_ac_loss = regressor_loss(fake_source_predicted_attribute_1,source_train_y)

        GF_total_loss = G_adv_loss + F_adv_loss + cycle_lambda *(GF_cycle_loss + FG_cycle_loss) + ac_lambda*(G_ac_loss + F_ac_loss)

        GF_total_loss.backward(retain_graph=True)
        optimizer_GF.step()
        
        # training discriminator_1
       
        optimizer_D1.zero_grad()
        
        fake_target_sample_2, fake_source_sample_2, recon_source_sample_2, recon_target_sample_2,fake_target_predicted_attribute_2, fake_source_predicted_attribute_2, real_target_predicted_attribute_2,real_source_predicted_attribute_2,real_source_sample_validity_2,fake_source_sample_validity_2,real_target_sample_validity_2,fake_target_sample_validity_2= model(source_train_x,source_train_y,target_train_x,target_train_y,source_semantic_feature,target_semantic_feature)

        total_wasserstein_loss_D1 = wasserstein_loss_discriminator(fake_source_sample_validity_2, real_source_sample_validity_2)
        gradient_penalty_D1 = compute_gradient_penalty(model.D1, source_train_x, fake_source_sample_2, source_train_y)

        D1_total_loss = total_wasserstein_loss_D1 + 10*gradient_penalty_D1
        D1_total_loss.backward(retain_graph=True)
        optimizer_D1.step()
        
        # training discriminator_2
        optimizer_D2.zero_grad()

        fake_target_sample_3, fake_source_sample_3, recon_source_sample_3, recon_target_sample_3,fake_target_predicted_attribute_3, fake_source_predicted_attribute_3, real_target_predicted_attribute_3,real_source_predicted_attribute_3,real_source_sample_validity_3,fake_source_sample_validity_3,real_target_sample_validity_3,fake_target_sample_validity_3= model(source_train_x,source_train_y,target_train_x,target_train_y,source_semantic_feature,target_semantic_feature)

        total_wasserstein_loss_D2 = wasserstein_loss_discriminator(fake_source_sample_validity_3, real_source_sample_validity_3)
        gradient_penalty_D2 = compute_gradient_penalty(model.D2, target_train_x, fake_target_sample_3, target_train_y)

        D2_total_loss = total_wasserstein_loss_D2 + 10*gradient_penalty_D2
        D2_total_loss.backward(retain_graph=True)
        optimizer_D2.step()
        
        #training regressor
        optimizer_R.zero_grad()

        fake_target_sample_4, fake_source_sample_4, recon_source_sample_4, recon_target_sample_4,fake_target_predicted_attribute_4, fake_source_predicted_attribute_4, real_target_predicted_attribute_4,real_source_predicted_attribute_4,real_source_sample_validity_4,fake_source_sample_validity_4,real_target_sample_validity_4,fake_target_sample_validity_4= model(source_train_x,source_train_y,target_train_x,target_train_y,source_semantic_feature,target_semantic_feature)
        
        regressor_loss_source = regressor_loss(real_source_predicted_attribute_4,source_train_y)
        regressor_loss_target = regressor_loss(real_target_predicted_attribute_4,target_train_y)

        regressor_loss_total = regressor_loss_source + regressor_loss_target

        regressor_loss_total.backward()
        optimizer_R.step()

    print(f'Epoch {epoch}, GF_Loss: {GF_total_loss}, D1_loss: {D1_total_loss}, D2_loss: {D2_total_loss}, R_loss: {regressor_loss_total}')



















    






    








        








    


