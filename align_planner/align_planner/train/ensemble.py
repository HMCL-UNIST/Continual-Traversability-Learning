import torch
import torch.optim as optim
from ament_index_python.packages import get_package_share_directory
from torch import nn
import os
import numpy as np
import datetime
from datetime import datetime_CAPI
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


from align_planner.models.infocva import InfoVAE
from align_planner.train.trainutils import DistanceMetrics
from align_planner.train.imost import ImostDataBuffer

class EnsembleNN(nn.Module):
    def __init__(self, feat_args,  model_class, ensemble_args, optimizer_class, *model_args, **model_kwargs):      
        super(EnsembleNN, self).__init__()         
        self.tau_thres = ensemble_args['tau_thres']
        self.lambda_weight = ensemble_args['lambda_weight']
        self.tensorboard = ensemble_args.get('tensorboard', False)
        if self.tensorboard: 
            self.setup_tensorboard()        
        self.method = ensemble_args.get('method', 'full')        
        self.num_models = ensemble_args['num_models']       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.path.join(get_package_share_directory('align_planner'), 'models')
        self.plot_dir = os.path.join(get_package_share_directory('align_planner'), 'plots')                        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_name = ensemble_args.get('weight',None)        
        self.num_recall = ensemble_args.get('num_recall', 30)        
        self.dist_metric = DistanceMetrics()
        self.feat_args = feat_args        
        self.memory_generator = InfoVAE(self.feat_args).to(self.device)          
        
        self.models = nn.ModuleList([model_class(*model_args, **model_kwargs) for _ in range(self.num_models)])                
        self.input_size = self.models[0].input_size        
        self.output_size = self.models[0].output_size
        self.hidden_units = self.models[0].hidden_units                
        
        self.optimizers = [optimizer_class([
                                {'params': model.parameters(), 'lr': 0.001},                                
                            ]) for model in self.models]        
        self.memory_optimizer = optim.Adam(self.memory_generator.parameters(), lr=0.001)
        
    def setup_tensorboard(self):            
        log_path = os.path.join(get_package_share_directory('align_planner'), 'log')
        os.makedirs(log_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name = 'ensemble'
        specific_log_dir = os.path.join(log_path, f"{log_name}_{timestamp}")
        self.writer = SummaryWriter(log_dir=specific_log_dir)
    
    def generate_teacher_predictions(self, x, is_recall):
        world_input = x[:, :self.feat_args['world_feat_size']]
        state_input = x[:, self.feat_args['world_feat_size']:]                         
        recalled_world_feat = None
        repeated_state_input = None
        
        if is_recall:
            recalled_world_feat, repeated_state_input  = self.memory_generator.random_gen(state_input, self.num_recall)                            
            x = torch.cat([recalled_world_feat, repeated_state_input], dim=1)            
        else:            
            x = torch.cat([world_input, state_input], dim=1)
            
        means_z = []
        variances_z = []
        for model in self.models:
            mean_z, var_z = model(x)            
            means_z.append(mean_z)
            variances_z.append(var_z)
        
        means_z = torch.stack(means_z, dim=1)
        variances_z = torch.stack(variances_z, dim=1)               
        
        return (means_z, variances_z, recalled_world_feat, repeated_state_input)
        
        
    def forward(self, x, is_recall = False):
        
        (means_z, variances_z, recalled_world_feat, repeated_state_input) = self.generate_teacher_predictions(x, is_recall)                                
        means, variances = self.models[0].transform_logits(means_z, variances_z)        
        mean_mu_ens, sigma_sq = self.gaussian_mixture_moments(means, variances)
        ale_ens, epi_ens = self.uncertainty_separation_parametric(means, variances)        
        ensemble_output = torch.stack([mean_mu_ens, ale_ens, epi_ens], dim=-1)                
        return (ensemble_output, recalled_world_feat, repeated_state_input)
            
    def gaussian_mixture_moments(self,mus, sigma_sqs):
        with torch.no_grad():
            mu = torch.mean(mus, dim=1)
            sigma_sq = torch.mean(sigma_sqs + mus**2, dim=1) - mu**2
        return mu, sigma_sq

    def uncertainty_separation_parametric(self,mu, var):        
        epistemic_uncertainty = torch.var(mu, dim=1)
        aleatoric_uncertainty = torch.mean(var, dim=1)
        return aleatoric_uncertainty, epistemic_uncertainty

    def load_weight(self, model_path = None):
        if model_path is None:
            model_path = self.model_name        
        if '.pth' not in model_path:
            model_path = f'{model_path}.pth'
        self.model_filename = os.path.join(self.model_dir, model_path)        
        model_weight = torch.load(self.model_filename)
        self.models.load_state_dict(model_weight['ensemble'])                        
        self.memory_generator.load_state_dict(model_weight['memory_generator']) 
        print(f"Model loaded from {self.model_filename}")

    def load_memory_generator_weight(self, model_path = None):
        if model_path is None:
            model_path = self.model_name        
        if '.pth' not in model_path:
            model_path = f'{model_path}.pth'
        self.model_filename = os.path.join(self.model_dir, model_path)        
        model_weight = torch.load(self.model_filename)
        self.memory_generator.load_state_dict(model_weight['memory_generator']) 
        print(f"Memory generator loaded from {self.model_filename}")
              
        
    def save_model(self, epoch_number = 0, weight_name = None, model_name = None):
        if weight_name is None:
            file_name = f'ensemble_epoch_{epoch_number}.pth'
        else:
            file_name = f'{weight_name}_epoch_{epoch_number}.pth'                
        if model_name is not None:
            file_name = model_name            
        self.model_filename = os.path.join(self.model_dir, file_name)
        self.models.eval()        
        self.memory_generator.eval()
        weights = {'ensemble': self.models.state_dict(),                   
                   'memory_generator': self.memory_generator.state_dict()}          
        torch.save(weights, self.model_filename)        
        print(f"Model saved to {self.model_filename}")
                                

    def log_gradients(self, epoch):
        for name, model in enumerate(self.models):
            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'model_{name}/{param_name}.grad', param.grad, epoch)
                    
        if self.memory_generator is not None:
            for param_name, param in self.memory_generator.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'memory_generator/{param_name}.grad', param.grad, epoch)

    def losses_update(self, losses, idx,
                    total_loss=torch.tensor(0.0),
                    model_loss=torch.tensor(0.0),
                    generator_loss=torch.tensor(0.0),
                    recons_loss=torch.tensor(0.0),
                    kld_loss=torch.tensor(0.0),
                    mmd_loss=torch.tensor(0.0),
                    off_recall_output_loss=torch.tensor(0.0),
                    recall_generator_loss=torch.tensor(0.0),
                    recall_recons_loss=torch.tensor(0.0),
                    recall_kld_loss=torch.tensor(0.0),
                    recall_mmd_loss=torch.tensor(0.0),
                    loss_stds=torch.tensor(0.0)):
        losses['total_loss'][idx] += total_loss.item()
        losses['model_loss'][idx] += model_loss.item()
        losses['generator_loss'][idx] += generator_loss.item()
        losses['recons_loss'][idx] += recons_loss.item()
        losses['kld_loss'][idx] += kld_loss.item()
        losses['mmd_loss'][idx] += mmd_loss.item()
        losses['off_recall_output_loss'][idx] += off_recall_output_loss.item()
        losses['recall_generator_loss'][idx] += recall_generator_loss.item()
        losses['recall_recons_loss'][idx] += recall_recons_loss.item()
        losses['recall_kld_loss'][idx] += recall_kld_loss.item()
        losses['recall_mmd_loss'][idx] += recall_mmd_loss.item()
        losses['loss_stds'] += loss_stds.item()
   
    def losses_divide(self,losses, num_batches):
        for key in losses.keys():
            losses[key] = losses[key] / (num_batches+1e-10)
        
    def log_losses(self,losses, epoch):
        for i in range(self.num_models):                    
            self.writer.add_scalar(f'TotalLoss/model{i}', losses['total_loss'][i], epoch)
            self.writer.add_scalar(f'ModelLoss/model{i}', losses['model_loss'][i], epoch)         
            self.writer.add_scalar(f'OffRecallLoss/model{i}', losses['off_recall_output_loss'][i], epoch)                    
            self.writer.add_scalar(f'Generator_Total/model{i}', losses['generator_loss'][i], epoch)                                                           
            self.writer.add_scalar(f'Generator_recon/model{i}', losses['recons_loss'][i], epoch)                                                           
            self.writer.add_scalar(f'Generator_kld/model{i}', losses['kld_loss'][i], epoch)                                                           
            self.writer.add_scalar(f'Generator_mmd/model{i}', losses['mmd_loss'][i], epoch)                                                                       
            self.writer.add_scalar(f'ValidationLoss/model{i}', losses['val_loss'][i], epoch)
        self.writer.add_scalar(f'Statictics/', losses['loss_stds'], epoch)
        self.log_gradients(epoch)
                    
    def losses_init(self):
        losses = {'total_loss': np.array([0.0]*len(self.models)),
                    'model_loss': np.array([0.0]*len(self.models)),
                    'generator_loss': np.array([0.0]*len(self.models)),                    
                    'recons_loss': np.array([0.0]*len(self.models)),
                    'kld_loss': np.array([0.0]*len(self.models)),
                    'mmd_loss': np.array([0.0]*len(self.models)),
                    'off_recall_output_loss': np.array([0.0]*len(self.models)),
                    'recall_generator_loss': np.array([0.0]*len(self.models)),
                    'recall_recons_loss': np.array([0.0]*len(self.models)),
                    'recall_kld_loss': np.array([0.0]*len(self.models)),
                    'recall_mmd_loss': np.array([0.0]*len(self.models)),
                    'val_loss': np.array([0.0]*len(self.models)),
                    'loss_stds': np.array([0.0])   
                    }
        return losses
    
    def compute_weighted_model_loss(self,conf_weight, model,  world_feat, state_input, targets = None):
        input_to_travnn = torch.cat([world_feat, state_input], dim=1)
        mean_z, var_z = model(input_to_travnn)
        mean, var = model.transform_logits(mean_z, var_z)
        assert targets is not None, "Targets are not provided"
        nll = 0.5 * (torch.log(2 * torch.pi * var) + ((targets - mean) ** 2) / var)
        loss = (conf_weight.unsqueeze(1) * nll).mean()        
        return loss

    def compute_model_loss(self,model, world_feat, state_input, targets = None):
        input_to_travnn = torch.cat([world_feat, state_input], dim=1)
        mean_z, var_z = model(input_to_travnn)
        mean, var = model.transform_logits(mean_z, var_z)
        assert targets is not None, "Targets are not provided"
        model_loss = model.loss(mean, var, targets)            
        return model_loss
           
    def get_ran_feat(self, world_feat, mean_range=(-3.0, 3.0), std_range=(0.1, 1.0)):        
        mean_ = torch.empty(1).uniform_(*mean_range).item()
        std_ = torch.empty(1).uniform_(*std_range).item()        
        epsilon = torch.normal(mean=mean_, std=std_, size=world_feat.shape).to(self.device)
        return world_feat + epsilon
    
                    
    def train_ensemble(self, imostbuffer, train_task_num, num_epochs, cur_task_dataloader, val_loader, full_memory_loaders, test_memory_loaders, model_prefix_name = None, T_i_1_model = None, test_loss_tracker = None):
               
        best_loss = float('inf')
        epochs_since_improvement = 0
        early_stop_patience = 300              
        self.loss_stats = []           

        for epoch in range(num_epochs):                                        
            if train_task_num == 0:
                losses  = self._train_epoch_with_memory_recall(None, T_i_1_model, cur_task_dataloader, full_memory_loaders)                                                                                    
            else:
                losses  = self._train_epoch_with_memory_recall(imostbuffer, T_i_1_model, cur_task_dataloader, full_memory_loaders)                                                                                    

            if epoch > 1:
                self.loss_stats.append(losses)        
            if epoch % 20 == 0:                
                model_name = model_prefix_name if model_prefix_name is not None else f'{model_prefix_name}_{epoch}.pth'                
                validation_loss, val_ensemble_loss = self._validate_epoch(val_loader)   
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss avg: {np.mean(losses['total_loss']  ):.4f},  Val ensemble Loss: {val_ensemble_loss[0]:.4f}, val model mean loss {np.mean(validation_loss):.4f}")
               
                if val_ensemble_loss < best_loss:
                    best_loss = val_ensemble_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1                

            if epochs_since_improvement >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")                    
                model_name = model_prefix_name if model_prefix_name is not None else f'{model_prefix_name}_{epoch}_{current_time}.pth'
                self.save_model(epoch, weight_name=model_name)
                break

        if test_memory_loaders is not None:
            test_loss, test_ensemble_loss = self._test_epoch(test_memory_loaders)   
            test_loss_tracker.log_epoch_validation_loss(train_task_num, test_loss, test_ensemble_loss)
   
    def _train_epoch_with_memory_recall(self, imostbuffer:ImostDataBuffer, T_i_1_model: 'EnsembleNN', cur_task_dataloader, full_memory_loaders):
        """
        Train the ensemble for one epoch.
        """
        model_loss_weight = 1e0        
        losses = self.losses_init()  

        prev_task_idx = -1
        for task_idx, dataloader in enumerate(full_memory_loaders):
            if dataloader.dataset == cur_task_dataloader.dataset:
                prev_task_idx = task_idx -1     

        for batch_idx, (inputs, targets) in enumerate(cur_task_dataloader): 
            self.models.train()
            self.memory_generator.eval()
            T_i_1_model.eval()
            inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)            
            world_input = inputs[:, :self.feat_args['world_feat_size']].clone()
            state_input = inputs[:, self.feat_args['world_feat_size']:].clone()   

            ''' Sample from buffer '''
            if self.method in {'imost'} and imostbuffer is not None:                
                sampled_input, sampled_output = imostbuffer.recall_cluster_data()
                sampled_world_input = sampled_input[:,:self.feat_args['world_feat_size']].clone().float().to(self.device)   
                sampled_state_input = sampled_input[:,self.feat_args['world_feat_size']:].clone().float().to(self.device)                   

            ''' Recall experience '''            
            with torch.no_grad():            
                is_recall = self.method in {'naivgen', 'proposed'}  # full and lwf will use the input data distribution                
                (recall_means_z, recall_variances_z, recall_env_feats, recall_state_feats) = T_i_1_model.generate_teacher_predictions(inputs, is_recall)                                                          
                recall_mean, recall_var = T_i_1_model.models[0].transform_logits(recall_means_z, recall_variances_z)                        
                recall_logvar = torch.log(recall_var)

            ''' compute batch statistics of confidence score'''
            if  self.method in {'adapt'} and prev_task_idx >= 0:
                with torch.no_grad():                                                                      
                    (prev_inputs, pre_targets) = next(iter(full_memory_loaders[prev_task_idx]))
                    prev_world_input = prev_inputs[:,:self.feat_args['world_feat_size']].float().to(self.device)   
                    prev_state_input = prev_inputs[:, self.feat_args['world_feat_size']:].float().to(self.device)   
                    reconstructed_world, _, _, _ = T_i_1_model.memory_generator(prev_world_input, prev_state_input)
                    recon_loss = torch.norm(reconstructed_world-prev_world_input, dim=1)
                    conf_mu = torch.mean(recon_loss)  
                    conf_std = torch.sqrt(torch.mean((recon_loss - conf_mu) ** 2))
                    ran_trav_world_feat = self.get_ran_feat(prev_world_input)                    
                    ran_trav_recon, _, _, _ = T_i_1_model.memory_generator(ran_trav_world_feat, prev_state_input)
                    untrc_recon_loss = torch.norm(ran_trav_recon-ran_trav_world_feat, dim=1)                    
                    confidence = torch.exp(-(((untrc_recon_loss - conf_mu) / (conf_std*2.0)) ** 2) * 0.5)
                    confidence[untrc_recon_loss < conf_mu] = 1.0                    
                    zero_targets = torch.zeros_like(pre_targets).float().to(self.device)  
                    

            for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):                                                                                    
                optimizer.zero_grad()  
                '''  computes the model mse loss for the online update data, D_i  '''         
                model_loss = self.compute_model_loss(model, world_input, state_input, targets)                               
                model_loss = model_loss * model_loss_weight
                loss = model_loss

                ''' adapt method'''
                ran_trav_loss = torch.tensor(0.0).to(device=self.device)
                if self.method in {'adapt'} and prev_task_idx >= 0:
                    ran_trav_loss = self.compute_weighted_model_loss(confidence, model, ran_trav_world_feat, prev_state_input, zero_targets)                               
                    loss +=ran_trav_loss

                ''' full memory'''
                if self.method in {'full'}:     
                    if full_memory_loaders is not None:
                        for mem_dataloader in full_memory_loaders:               
                            same_dataset = cur_task_dataloader.dataset == mem_dataloader.dataset                                         
                            if same_dataset == False:
                                (mem_input, mem_output) = next(iter(mem_dataloader))
                                mem_input, mem_output = mem_input.float().to(self.device), mem_output.float().to(self.device)   
                                val_world_input = mem_input[:, :self.feat_args['world_feat_size']].clone()
                                val_state_input = mem_input[:, self.feat_args['world_feat_size']:].clone()      
                                loss += self.compute_model_loss(model, val_world_input, val_state_input, mem_output)* self.lambda_weight       
                           

                ''' IMOST '''
                if self.method in {'imost'} and imostbuffer is not None:             
                    loss += self.compute_model_loss(model, sampled_world_input, sampled_state_input, sampled_output)* self.lambda_weight    

                ''' recall from memory '''
                off_recall_output_loss = torch.tensor(0.0).to(device=self.device)
                if recall_env_feats is not None:
                    recall_input = torch.cat([recall_env_feats, recall_state_feats], dim=1)
                    mean_z, var_z = model(recall_input)
                    mean, var = model.transform_logits(mean_z, var_z)
                    logvar = torch.log(var)
                    
                    if self.method in {'proposed','naivgen'}:                          
                        off_recall_output_loss = self.dist_metric.jensen_shannon_divergence_torch(mean, logvar, recall_mean[:,idx,:], recall_logvar[:,idx,:]).mean() 
                    elif self.method in {'lwf'}: 
                        off_recall_output_loss = self.dist_metric.kl_divergence(model, mean, logvar, recall_mean[:,idx,:], recall_logvar[:,idx,:]).mean()                              
                    else: 
                        assert True, "The method is not supported for recalling update"
                    off_recall_output_loss = off_recall_output_loss * self.lambda_weight
                    loss += off_recall_output_loss
                     
                loss.backward()
                optimizer.step()                                
                                        
                self.losses_update(losses, idx, total_loss = loss, model_loss = model_loss, off_recall_output_loss = off_recall_output_loss)                                
        self.losses_divide(losses, len(cur_task_dataloader))        
        return losses
      

    def train_generator(self, task_num, cur_task_dataloader, num_epochs, model_prefix_name = None, T_i_1_model = None, test_loss_tracker = None):        

        best_loss = float('inf')
        epochs_since_improvement = 0
        early_stop_patience = 50  # ← you can set this to whatever you like
        self.loss_stats = []   
        for epoch in range(num_epochs):                                                
            losses = self.losses_init()   
            self.models.eval()      
            self.memory_generator.train()                     
            for batch_idx, (inputs, targets) in enumerate(cur_task_dataloader):
                if inputs.shape[0] == 1:
                    continue

                self.memory_optimizer.zero_grad()                
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)                
                world_input = inputs[:, :self.feat_args['world_feat_size']]
                state_input = inputs[:, self.feat_args['world_feat_size']:]                                   
                        
                is_recall = self.method in {'naivgen','proposed'}
                if task_num > 0 and is_recall:
                    assert T_i_1_model is not None, "The T_i_1_model is not provided for the proposed method"
                    with torch.no_grad():
                        (recall_outputs, recalled_world_feat, repeated_state_input) = T_i_1_model(inputs, is_recall)      
                        recall_log_var  = torch.log(recall_outputs[:,:,1]+recall_outputs[:,:,2])   
                        '''
                        Uncertainty aware memory filtering
                        '''
                        def filter_recall_world_feat(recall_log_var, recalled_world_feat, repeated_state_input, var_threshold):                            
                            max_log_var, _ = recall_log_var.max(dim=1)
                            mask = max_log_var < torch.log(var_threshold)
                            return recalled_world_feat[mask,:], repeated_state_input[mask,:]
                          
                        if self.method == 'proposed':
                            var_threshold = torch.tensor(self.tau_thres).to(self.device)
                            filt_world_feat, filt_state_input = filter_recall_world_feat(recall_log_var, recalled_world_feat, repeated_state_input,var_threshold)
                        elif self.method =='naivgen':
                            filt_world_feat = recalled_world_feat
                            filt_state_input = repeated_state_input
                        else: 
                            assert True , "The method is not supported for recalling generator update"

                        merged_world_feat = torch.vstack([world_input, filt_world_feat])
                        merged_state_input = torch.vstack([state_input, filt_state_input])
                else:
                    merged_world_feat = world_input
                    merged_state_input = state_input

                recon_world_feat, mu, logvar, z = self.memory_generator(merged_world_feat ,merged_state_input)   
                loss, recons_loss, kld_loss, mmd_loss=  self.memory_generator.get_infovae_loss(recon_world_feat, merged_world_feat, mu, logvar,z)                        

                loss.backward()
                self.memory_optimizer.step()                 
                self.losses_update(losses,
                                    idx=0,
                                    total_loss=loss,
                                    recons_loss=recons_loss,
                                    kld_loss=kld_loss,
                                    mmd_loss=mmd_loss)                            
            if test_loss_tracker is not None:
                test_loss_tracker.update_recon(task_num, recons_loss.item())
           

            self.losses_divide(losses, len(cur_task_dataloader))

                    
            if epoch % 30 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {losses['total_loss'][0]:.4f}")             
            if epoch > 1:                
                self.loss_stats.append(losses.copy())        

            current_loss = losses['total_loss'][0]
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1                
                if epochs_since_improvement >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    self.save_model(epoch_number=epoch, weight_name=model_prefix_name)   
                    break
                

    def _validate_epoch(self, dataloader):
        self.eval()       
        with torch.no_grad():
            running_loss = np.array([0.0]*len(self.models))              
            ensemble_loss = np.array([0.0])                              
            for inputs, targets in dataloader:                    
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                world_input = inputs[:, :self.feat_args['world_feat_size']]
                state_input = inputs[:, self.feat_args['world_feat_size']:]   
                input_to_travnn = torch.cat([world_input, state_input], dim=1)
                ensemble_output, _, _ = self(inputs, is_recall = False)
                ensemble_loss += self.models[0].loss(ensemble_output[:,:,0], ensemble_output[:,:,1]+ensemble_output[:,:,2], targets).item()
                for model_idx, model in enumerate(self.models):                                                
                    mean_z, var_z = model(input_to_travnn)                    
                    mean, var = model.transform_logits(mean_z, var_z)
                    loss = model.loss(mean, var, targets)
                    running_loss[model_idx] += loss.item()                                        
            running_loss = running_loss / len(dataloader)
            ensemble_loss = ensemble_loss / len(dataloader)        
        return running_loss.copy(), ensemble_loss.copy()
    
    def _test_epoch(self, dataloaders):
        self.eval()
        R_vectors = {}
        ensemble_loss_vectors = {}
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]
        with torch.no_grad():
            for task_id, dataloader in enumerate(dataloaders):
                model_losses, ens_loss = self._validate_epoch(dataloader)
                R_vectors[f'test_{task_id}'] = model_losses
                ensemble_loss_vectors[f'test_{task_id}'] = ens_loss
        return R_vectors, ensemble_loss_vectors