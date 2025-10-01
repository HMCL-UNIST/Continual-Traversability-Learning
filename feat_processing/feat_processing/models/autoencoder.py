import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import os
from ament_index_python.packages import get_package_share_directory


class AutoEncoder(nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(AutoEncoder, self).__init__()
        '''
        Autoencoder
        Generate world feature conditioned to the state feature.
        '''        
        self.model_save_path = os.path.join(get_package_share_directory('feat_processing'), 'models')
        os.makedirs(self.model_save_path, exist_ok=True)
        self.n_feats = 384
        latent_size = 3
        self.device = device        
        self.encoder_layers = nn.Sequential(
            nn.Linear(self.n_feats, 60),            
            nn.LeakyReLU(),
            nn.Linear(60, 40),                           
            nn.LeakyReLU(),            
            nn.Linear(40, latent_size)
        ).to(device=self.device)
            
        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_size, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 60),
            nn.LeakyReLU(),
            nn.Linear(60, self.n_feats)
        ).to(device=self.device)

              
        

    
    def forward(self, x):
        latent = self.encoder_layers(x)
        x = self.decoder_layers(latent)
        return x, latent

    def recon_loss(self, outputs, inputs):
        return F.mse_loss(outputs, inputs)
    
    def load_weight(self, model_path):
        self.model_filename = os.path.join(self.model_save_path, model_path)
        a = torch.load(self.model_filename) 
        self.load_state_dict(torch.load(self.model_filename))
        print(f"Model loaded from {self.model_filename}")
        
    def save_model(self, epoch_number = 0):
        self.model_filename = os.path.join(self.model_save_path, f'autoencoder_epoch_{epoch_number}.pth')
        torch.save(self.state_dict(), self.model_filename)
        print(f"Model saved to {self.model_filename}")

    def train_autoencoder(self, train_loader, num_epochs, optimizer: optim.Adam, validation_loader=None):
        for epoch_number in range(1, num_epochs + 1):
            self.train()            
            loss, mmd_loss = self._train_epoch(train_loader, optimizer)
            print(f"Epoch {epoch_number}/{num_epochs}, AutoEncoder training Loss: {loss:.4f}, mmd_loss: {mmd_loss:.4f}")            
            if validation_loader is not None:                   
                self._test_epoch(validation_loader)
                
            if epoch_number % 500 == 0:
                self.save_model(epoch_number)                
                
    def _test_epoch(self, validation_loader):
        self.eval()
        running_loss = 0
        with torch.no_grad():
            for batch_ind, batch in enumerate(validation_loader):
                inputs = batch                
                inputs = inputs.reshape([-1,inputs.shape[-1]]).to(self.device)                                  
                outputs, latent = self.forward(inputs)
                loss = self.recon_loss(outputs, inputs)
                running_loss += loss.item()
        print(f"Validation Loss: {running_loss / len(validation_loader):.4f}")
            
    def _train_epoch(self, train_loader, optimizer: optim.Adam):
        self.train()
        running_loss = 0
        running_mmd_loss = 0
        for batch_ind, batch in enumerate(train_loader):            
            optimizer.zero_grad()
            inputs = batch
            inputs = inputs.reshape([-1,inputs.shape[-1]]).to(self.device)                                  
            outputs, latent = self.forward(inputs)
            loss = self.recon_loss(outputs, inputs) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()       
                 
        return running_loss / len(train_loader), running_mmd_loss/len(train_loader)

   