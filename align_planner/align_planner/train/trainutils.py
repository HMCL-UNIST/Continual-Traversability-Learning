import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import os
from ament_index_python.packages import get_package_share_directory
from torch.utils.data import DataLoader, random_split
from align_planner.train.datasets import GridStateDataset


def get_dataloaders(Tasks_dirs, val_split=0.1, test_split=0.1, batch_size=128):        
    train_loaders = []
    val_loaders = []
    test_loaders = []
    tasks_base_dirs = []

    for idx, data_folder in enumerate(Tasks_dirs):
        base_path = os.path.join(get_package_share_directory('align_planner'), data_folder)
        tasks_base_dirs.append(base_path)

        folder_dirs = [
            os.path.join(base_path, env_folder)
            for env_folder in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, env_folder))
        ]

        dataset = GridStateDataset(
            dataset_paths=folder_dirs,
            data_load=True,
            compute_normalizing_constants=False
        )

        # Split the dataset
        total_len = len(dataset)
        test_len = int(total_len * test_split)
        val_len = int(total_len * val_split)
        train_len = total_len - test_len - val_len

        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders, tasks_base_dirs


class ValidationLoss:
    def __init__(self, num_model, num_task):
        self.num_model = num_model
        self.num_task = num_task
        self.acc_df = None
        self.forgetting_df = None
        self.test_loss_table = None
        self.test_ensemble_acc_table = None
        self.table_dir = os.path.join(get_package_share_directory('align_planner'), 'tables')                        
        os.makedirs(self.table_dir, exist_ok=True)
        self.test_loss = {
            f'task_{train_task_num}': {
                f'test_{test_task_num}': np.zeros((0, num_model))  
                for test_task_num in range(num_task)
            }
            for train_task_num in range(num_task)
        }
        self.ensemble_loss = {
            f'task_{train_task_num}': {
                f'test_{test_task_num}': np.zeros((0, 1))  
                for test_task_num in range(num_task)
            }
            for train_task_num in range(num_task)
        }
        self.recon_loss = {
            f'task_{train_task_num}': np.array([])  
            for train_task_num in range(num_task)
        }

    def update_recon(self, train_task_num, recon_loss):
        
        if f'task_{train_task_num}' in self.recon_loss:
            self.recon_loss[f'task_{train_task_num}'] = np.append(
                self.recon_loss[f'task_{train_task_num}'], recon_loss
            )
        else:
            raise KeyError(f"Training task {train_task_num} not found.")
            
    def update_test_loss(self, train_task_num, test_task_num, val_loss_array):
        
        if f'task_{train_task_num}' in self.test_loss:
            if f'test_{test_task_num}' in self.test_loss[f'task_{train_task_num}']:
                self.test_loss[f'task_{train_task_num}'][f'test_{test_task_num}'] = np.vstack(
                    [self.test_loss[f'task_{train_task_num}'][f'test_{test_task_num}'], val_loss_array]
                )
            else:
                raise KeyError(f"Test task {test_task_num} not found.")
        else:
            raise KeyError(f"Training task {train_task_num} not found.")
        

    def update_ensemble_loss(self, train_task_num, val_task_num, test_loss_array):
       
        if f'task_{train_task_num}' in self.ensemble_loss:
            if f'test_{val_task_num}' in self.ensemble_loss[f'task_{train_task_num}']:
                self.ensemble_loss[f'task_{train_task_num}'][f'test_{val_task_num}'] = np.vstack(
                    [self.ensemble_loss[f'task_{train_task_num}'][f'test_{val_task_num}'], test_loss_array]
                )
            else:
                raise KeyError(f"test task {val_task_num} not found.")
        else:
            raise KeyError(f"Training task {train_task_num} not found.")

    def log_epoch_validation_loss(self, train_task_num, test_losses, ensemble_losses):
        
        for val_task, test_loss_array in test_losses.items():
            val_task_num = int(val_task.split('_')[1])  
            self.update_test_loss(train_task_num, val_task_num, test_loss_array)

        for val_task, test_loss_array in ensemble_losses.items():
            val_task_num = int(val_task.split('_')[1])  
            self.update_ensemble_loss(train_task_num, val_task_num, test_loss_array)
            


    def compute_forgetting(self, acc_df):
        forgetting_df = pd.DataFrame(index=acc_df.index, columns=acc_df.columns, dtype=float)
        for row_idx, row_name in enumerate(acc_df.index):  
            for col_idx in range(row_idx + 1, len(acc_df.columns)):  # future tasks only
                current_task = acc_df.columns[col_idx]
                past_tasks = acc_df.columns[:col_idx]
                current_val = acc_df.loc[row_name, current_task]
                past_vals = acc_df.loc[row_name, past_tasks].dropna()
                if not past_vals.empty and pd.notna(current_val):
                    forgetting = current_val - past_vals.min()
                    forgetting_df.loc[row_name, current_task] = round(forgetting, 4)
  
        return forgetting_df


    def compute_accuracy_and_forgetting(self, test_loss_dict = None):
        if test_loss_dict is None:
            test_loss_dict = self.ensemble_loss
        test_loss_table = {}

        for task_idx, (task_name, val_splits) in enumerate(test_loss_dict.items()):
            task_row = {}
            for test_idx, (val_name, loss_array) in enumerate(val_splits.items()):
                if test_idx > task_idx:
                    continue
                min_loss = loss_array.min()
                task_row[val_name] = min_loss
            test_loss_table[task_name] = task_row
        self.test_ensemble_acc_table = test_loss_table
        self.acc_df = pd.DataFrame(test_loss_table)       
        self.forgetting_df =  self.compute_forgetting(self.acc_df)
        return self.acc_df, self.forgetting_df
        
    


class DistanceMetrics(nn.Module):
    def __init__(self, length_scale=1.0, matern_nu=1.5):
        """
        DistanceMetrics as a PyTorch nn.Module
        :param length_scale: Hyperparameter for Gaussian and Matern kernels
        :param matern_nu: Smoothness parameter for the Matern kernel (nu=0.5, 1.5, 2.5)
        """
        super(DistanceMetrics, self).__init__()
        self.length_scale = nn.Parameter(torch.tensor(length_scale, dtype=torch.float32))
        self.matern_nu = matern_nu
        self.mse = nn.MSELoss()


    @staticmethod
    def trace_weighted_mse(logvar,x, y, eps=1e-6):    
        var = torch.exp(logvar) + eps  # to prevent zero variance
        trace = var.sum(dim=1, keepdim=True)  # shape: (B, 1), per-sample trace
        weights = 1.0 / trace  # inverse total uncertainty → higher weight if more certain


    @staticmethod
    def weighted_euclidean_distance(x, y, variances):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if variances.dim() == 1:
            variances = variances.unsqueeze(0)
        variances = variances.expand_as(x)  
        weights = (torch.tanh(-torch.exp(variances))+1)/ 2    
        return torch.norm(weights * (x - y), p=2, dim=1)
    
    
    @staticmethod
    def euclidean_distance(x, y):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        return torch.norm(x - y, p=2, dim=1)

    @staticmethod
    def manhattan_distance(x, y):
        return torch.norm(x - y, p=1, dim=1)

    @staticmethod
    def chebyshev_distance(x, y):
        return torch.max(torch.abs(x - y), dim=1).values

    @staticmethod
    def cosine_distance(x, y):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
        return 1 - torch.sum(x_norm * y_norm, dim=1)

    def mse_loss(self ,x, y):
        return self.mse(x, y)
    
    def gaussian_kernel(self, x, y):
        squared_dist = torch.sum((x - y) ** 2, dim=1)
        return torch.exp(-squared_dist / (2 * self.length_scale ** 2))

    def matern_kernel(self, x, y):
        euclidean_dist = torch.norm(x - y, p=2, dim=1)
        scaled_dist = (torch.sqrt(torch.tensor(2.0 * self.matern_nu)) * euclidean_dist) / self.length_scale
        if self.matern_nu == 0.5:
            return torch.exp(-scaled_dist)
        elif self.matern_nu == 1.5:
            return (1 + scaled_dist) * torch.exp(-scaled_dist)
        elif self.matern_nu == 2.5:
            return (1 + scaled_dist + (scaled_dist ** 2) / 3) * torch.exp(-scaled_dist)
        else:
            raise ValueError("Only nu=0.5, nu=1.5, or nu=2.5 are supported")
    @staticmethod
    def kl_divergence_normal(mu1, log_var1, mu2, log_var2):
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)
        kl_div = 0.5 * (torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1)
        return kl_div.sum(dim=1)
    
   
    def jensen_shannon_divergence_torch(self,mu1, log_var1, mu2, log_var2):
        m1 = mu1.clone()
        m2 = mu2.clone()
        logvar1 = log_var1.clone()
        logvar2 = log_var2.clone()        
        
        cov1 = torch.diag_embed(logvar1.exp())
        cov2 = torch.diag_embed(logvar2.exp())    
        dist1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m1, covariance_matrix=cov1)
        dist2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m2, covariance_matrix=cov2)
        kl_div = torch.distributions.kl.kl_divergence(dist1, dist2) 
        kl_div2 = torch.distributions.kl.kl_divergence(dist2, dist1) 

        return  0.5 * (kl_div + kl_div2)   
    
    def kl_divergence(self,mu1, log_var1, mu2, log_var2, pairwise = False):
        
        if pairwise:
            m1 = mu1.unsqueeze(1).clone()  # [batch1, 1, dim]
            m2 = mu2.unsqueeze(0).clone()  # [1, batch2, dim]
            logvar1 = log_var1.unsqueeze(1).clone()  # [batch1, 1, dim]
            logvar2 = log_var2.unsqueeze(0).clone()  # [1, batch2, dim]
        else:
            m1 = mu1.clone()
            m2 = mu2.clone()
            logvar1 = log_var1.clone()
            logvar2 = log_var2.clone()        
        
        cov1 = torch.diag_embed(logvar1.exp())
        cov2 = torch.diag_embed(logvar2.exp())    
        dist1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m1, covariance_matrix=cov1)
        dist2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m2, covariance_matrix=cov2)
        kl_div = torch.distributions.kl.kl_divergence(dist1, dist2) 
        
        return kl_div
   
    def explicit_kl_divergence(self,mu1, log_var1, mu2, log_var2):
        
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)

        # Compute KL divergence
        kl_div = 0.5 * (
            torch.sum(log_var2 - log_var1, dim=-1) +
            torch.sum((var1 + (mu1 - mu2).pow(2)) / var2, dim=-1) -
            mu1.size(-1)
        )

        return kl_div


        
    def jensen_shannon_divergence(self,mu1, log_var1, mu2, log_var2, pairwise = False):
   
        # Mixture distribution
        mu_m = 0.5 * (mu1 + mu2)
        log_var_m = torch.log(0.5 * (torch.exp(log_var1) + torch.exp(log_var2)))
        # KL divergences
        
        kl1_exp = self.explicit_kl_divergence(mu1, log_var1, mu_m, log_var_m)                            
        kl2_exp = self.explicit_kl_divergence(mu2, log_var2, mu_m, log_var_m)    

        # Jensen-Shannon Divergence
        js_div = 0.5 * (kl1_exp + kl2_exp)    
        return js_div



