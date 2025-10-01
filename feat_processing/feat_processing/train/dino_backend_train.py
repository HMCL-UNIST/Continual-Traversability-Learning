import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from ament_index_python.packages import get_package_share_directory
from feat_processing.models.autoencoder import AutoEncoder  
from feat_processing.train.datasets import BackendFeatDataset   

# Example usage
if __name__ == "__main__":
    
    folder_names = ["data"]
    dataset = BackendFeatDataset(data_dirs = folder_names, data_load =True)
    
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    autoencoder = AutoEncoder(device=device).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    num_epochs = 2000
    autoencoder.train_autoencoder(train_loader, num_epochs, optimizer)