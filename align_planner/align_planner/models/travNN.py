import torch
from torch import nn
from torch.nn import functional as F
import os
from ament_index_python.packages import get_package_share_directory

class TravNN(nn.Module):
    def __init__(self, travnn_args, epsilon = 1e-6):
        super(TravNN, self).__init__()
          
        world_feat_size = travnn_args['world_feat_size']
        state_feat_size = travnn_args['state_feat_size']
        trav_hidden_units = travnn_args['trav_hidden_units']        
        trav_output_size = travnn_args['trav_output_size']                
        self.device = travnn_args.get('device', torch.device("cuda"))
        
        self.model_save_path = os.path.join(get_package_share_directory('align_planner'), 'models')        
        
        input_size = world_feat_size + state_feat_size
        self.input_size = input_size
        self.output_size = trav_output_size
        self.hidden_units = trav_hidden_units
        self.epsilon = epsilon

        
        layers = [nn.Linear(input_size, trav_hidden_units[0]), nn.LeakyReLU()]
        for i in range(1, len(trav_hidden_units)):
            layers.append(nn.Linear(trav_hidden_units[i-1], trav_hidden_units[i]))
            layers.append(nn.LeakyReLU())
        self.network = nn.Sequential(*layers)
        self.nn_fc_mean = nn.Linear(trav_hidden_units[-1], trav_output_size)
        self.nn_fc_var = nn.Linear(trav_hidden_units[-1], trav_output_size)

    def transform_logits(self, mean_z, var_z, epsilon=1e-5):     
        return mean_z, F.softplus(var_z)+epsilon
    
    
    def forward(self, x):
        x = self.network(x)
        mean_z = self.nn_fc_mean(x)
        var_z = self.nn_fc_var(x)  
        return mean_z, var_z
    
    def loss(self,mean, var,targets):        
        nll = 0.5 * (torch.log(2 * torch.pi * var) + ((targets - mean) ** 2) / var)
        return nll.mean()
    
    def load_weight(self, model_path):
        self.model_filename = os.path.join(self.model_save_path, model_path)
        self.load_state_dict(torch.load(self.model_filename))
        print(f"Model loaded from {self.model_filename}")
        
    def save_model(self, epoch_number = 0):
        self.model_filename = os.path.join(self.model_save_path, f'travnn_epoch_{epoch_number}.pth')
        torch.save(self.state_dict(), self.model_filename)
        print(f"Model saved to {self.model_filename}")
