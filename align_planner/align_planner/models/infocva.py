import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.optim as optim
import os
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class InfoVAE(nn.Module):
    def __init__(self,feat_args= dict()):
        #  world_feat_size, state_feat_size, hidden_units, latent_size, z_var=3.0, kernel_type='rbf', device=torch.device("cuda")
        super(InfoVAE, self).__init__()
        '''
        conditional variational autoencoder
        generate world feature conditoined to the state feature. 
        '''      
        self.feat_file_name = feat_args.get('weight', 'infocvae.pth')
        self.model_save_path = os.path.join(get_package_share_directory('align_planner'), 'models')
        os.makedirs(self.model_save_path, exist_ok=True)
        
        self.tensorboard = feat_args.get('tensorboard', False)
        if self.tensorboard:
            log_path = os.path.join(get_package_share_directory('align_planner'), 'log')
            os.makedirs(log_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_name = 'infocvae'
            specific_log_dir = os.path.join(log_path, f"{log_name}_{timestamp}")
            self.writer = SummaryWriter(log_dir=specific_log_dir)
        
        self.model_prefix = feat_args.get('model_prefix', '')
        hidden_units = feat_args.get('hidden_units', [16,32,16])
        latent_size = feat_args.get('latent_size', 5)        
        self.latent_size = latent_size
        self.z_var = feat_args.get('z_var', 1.0)
        self.kernel_type = feat_args.get('kernel_type', 'rbf')       
        self.device = feat_args.get('device', torch.device("cuda"))
        self.world_feat_size = feat_args.get('world_feat_size', 6)
        self.state_feat_size = feat_args.get('state_feat_size', 6)

        
        encoder_layers = [nn.Linear(self.world_feat_size+self.state_feat_size, hidden_units[0]), nn.LeakyReLU()]
        # Create hidden layers
        for i in range(1, len(hidden_units)):
            encoder_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            encoder_layers.append(nn.LeakyReLU())        
        
        # Encoder        
        self.encoder_layers = nn.Sequential(*encoder_layers)        
        self.fc21 = nn.Linear(hidden_units[-1], latent_size)
        self.fc22 = nn.Linear(hidden_units[-1], latent_size)

        # Decoder
        decoder_layers = [nn.Linear(latent_size+self.state_feat_size, hidden_units[-1]), nn.LeakyReLU()]
        # Create hidden layers
        for i in range(1,len(hidden_units)):            
            decoder_layers.append(nn.Linear(hidden_units[len(hidden_units)-i], hidden_units[len(hidden_units)-i-1]))
            decoder_layers.append(nn.LeakyReLU())        
        self.decoder_layers = nn.Sequential(*decoder_layers)        
        
        
        self.fc4 = nn.Linear(hidden_units[0], self.world_feat_size)

    def freeze_all_layers(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        
            
    def encode(self, x, c):  # Q(z|x, c)
        '''
        x: (bs, input_size)
        c: (bs, class_size)
        '''
        
        inputs = torch.cat([x, c], 1)  # (bs, world_feature + state_feature)
        h1 = self.encoder_layers(inputs)
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+state_feature)
        h3 = self.decoder_layers(inputs)
        return self.fc4(h3)
    
    def random_gen(self, c, num_recall = None):            
        if num_recall is None:
            num_recall = 2    
        rand_z = torch.randn(c.size(0)*num_recall, self.latent_size).to(self.device)
        repeated_c = c.repeat(num_recall,1)
        recon = self.decode(rand_z, repeated_c)        
        return recon, repeated_c
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)        
        return self.decode(z, c), mu, logvar, z

    def compute_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        # Expand tensors to match dimensions
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / (sigma+1e-9)))
        return result

    def compute_inv_mult_quad(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd
    
    def get_infovae_loss(self,recon_x, x, mu, logvar, z):        
        batch_size = x.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        kld_weight = 1.0
        reg_weight = 100.0    
        beta = 1.0 
        alpha = 0.0
        
        recons_loss = F.mse_loss(recon_x, x)
        recons_loss = beta * recons_loss
        mmd_loss = self.compute_mmd(z)
        mmd_loss = (alpha + reg_weight - 1.)/bias_corr * mmd_loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kld_loss = (1. - alpha) * kld_weight * kld_loss
        loss = recons_loss + kld_loss + mmd_loss                    
        return loss, recons_loss, kld_loss, mmd_loss   

    
    
        
    def get_uncertainty_aware_infovae_loss(self, recall_log_var, recon_x, x, mu, logvar, z):        
        batch_size = x.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        kld_weight = 1.0
        reg_weight = 100.0    
        beta = 1.0 
        alpha = 0.0
        eps = 1e-7
        
        var = torch.exp(recall_log_var) + eps  # to prevent zero variance
        trace = var.sum(dim=1, keepdim=True)  # shape: (B, 1), per-sample trace
        weights = ((torch.tanh(-trace)+1)/ 2).mean(dim=1)
        recons_loss = torch.norm(weights.unsqueeze(1) * (recon_x - x), p=2, dim=1).mean()                      

        recons_loss = beta * recons_loss
        mmd_loss = self.compute_mmd(z)
        mmd_loss = (alpha + reg_weight - 1.)/bias_corr * mmd_loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kld_loss = (1. - alpha) * kld_weight * kld_loss
        loss = recons_loss + kld_loss + mmd_loss                    
        return loss, recons_loss, kld_loss, mmd_loss   
    
    
    def get_loss(self,recon_x, mu, logvar, z):        
        batch_size = recon_x.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        kld_weight = 1.0
        reg_weight = 100.0    
        alpha = 0.5
        mmd_loss = self.compute_mmd(z)
        mmd_loss = (alpha + reg_weight - 1.)/bias_corr * mmd_loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        kld_loss = (1. - alpha) * kld_weight * kld_loss
        loss = kld_loss + mmd_loss                    
        return loss
    
    def load_weight(self, model_path = None):
        if model_path is None:
            model_path = self.feat_file_name
        self.model_filename = os.path.join(self.model_save_path, model_path)
        if os.path.exists(self.model_filename):
            self.load_state_dict(torch.load(self.model_filename))
            print(f"Model loaded from {self.model_filename}")
        else:
            print(f"Model file not found at {self.model_filename}")
        
        
    def save_model(self, epoch_number = 0):
        model_filename = self.model_prefix + f'generator_epoch_{epoch_number}.pth'
        self.model_filename = os.path.join(self.model_save_path, model_filename)
        torch.save(self.state_dict(), self.model_filename)
        print(f"Model saved to {self.model_filename}")

            
     