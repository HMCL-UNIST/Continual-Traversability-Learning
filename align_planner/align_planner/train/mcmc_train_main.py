from multiprocessing import Process
import multiprocessing
MAX_PARALLEL = 4
semaphore = multiprocessing.Semaphore(MAX_PARALLEL)
import os
import joblib
import torch
import torch.optim as optim
from ament_index_python.packages import get_package_share_directory
import numpy as np
from datetime import datetime

from align_planner.models.travNN import TravNN
from align_planner.train.trainutils import get_dataloaders, ValidationLoss
from align_planner.train.imost import ImostDataBuffer
from align_planner.train.ensemble import EnsembleNN


class SequentialEnsembleLearner:
    def __init__(self, dataloaders, val_loaders, test_loaders, ensemble_args, feat_args, travnn_args, device):        
        self.imostbuffer = ImostDataBuffer(KNN_hyperparameter=10,lambda_param_for_js_filtering=ensemble_args['imost_lambda'])      
        self.test_loss_tracker = ValidationLoss(num_model = ensemble_args['num_models'], num_task = ensemble_args['num_tasks'])
        self.dataloaders = dataloaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.ensemble_args = ensemble_args        
        self.feat_args = feat_args        
        self.travnn_args = travnn_args
        self.device = device
        self.batch_loss_stats = {}

    def train_loop(self, method = 'full', num_epoch_ensemble = 3,num_epoch_generator = 3):        
        seq_trained_ensemble_model_name= []
        seq_trained_generator_model_name = []
        for idx, cur_task_dataloader in enumerate(self.dataloaders):

            val_loader = self.val_loaders[idx] 
            full_memory_loaders = self.dataloaders[:idx+1] if idx < len(self.dataloaders) else None            
            test_memory_loaders = self.test_loaders[:idx+1] if idx < len(self.test_loaders) else None
            if idx == 0:                    
                T_1 = self.update_ensemble(method, None, None, cur_task_dataloader, val_loader, full_memory_loaders, test_memory_loaders, num_epochs_ensemble=num_epoch_ensemble, task_num = idx)
                seq_trained_ensemble_model_name.append(T_1)
                R_1 = self.update_generator(method, T_1, None, cur_task_dataloader, num_epochs=num_epoch_generator, task_num = idx)
                seq_trained_generator_model_name.append(R_1)
                if method in {'imost'}:
                    self.imostbuffer.store_dataloader(cur_task_dataloader)
                continue            
            T_i_1 = seq_trained_ensemble_model_name[-1]
            R_i_1 = seq_trained_generator_model_name[-1]
            T_i = self.update_ensemble(method, T_i_1, R_i_1, cur_task_dataloader, val_loader, full_memory_loaders, test_memory_loaders, num_epochs_ensemble=num_epoch_ensemble, task_num = idx)            
            seq_trained_ensemble_model_name.append(T_i)
            R_i = self.update_generator(method, T_i_1, R_i_1,cur_task_dataloader, num_epochs=num_epoch_generator, task_num = idx)
            seq_trained_generator_model_name.append(R_i)
            if method in {'imost'}:
                self.imostbuffer.store_dataloader(cur_task_dataloader)
        
        stats_path = self.stats_save()
        return stats_path
        
    def update_generator(self,method, T_i_1, R_i_1, cur_task_dataloader, num_epochs=201, task_num = 0):       
        ensemble_args = self.ensemble_args.copy()              
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")            
        generator_model_file_name = f'{method}_e_{task_num}_generator_{current_time}.pth'
        ensemble_model = EnsembleNN( self.feat_args, TravNN, ensemble_args, optim.Adam, self.travnn_args).to(self.device)     
        T_i_1_model = EnsembleNN(self.feat_args, TravNN, ensemble_args, optim.Adam, self.travnn_args).to(self.device)
        if task_num ==0:
            ensemble_model.load_weight(T_i_1)            
        else:
            T_i_1_model.load_weight(T_i_1)     
            T_i_1_model.load_memory_generator_weight(R_i_1)              
            ensemble_model.load_weight(R_i_1)
        ensemble_model.loss_stats = []
        if method in {'naivgen','proposed'}:
            ensemble_model.train_generator(task_num, cur_task_dataloader, num_epochs=num_epochs, model_prefix_name=f'generator_{task_num}_', T_i_1_model=T_i_1_model, test_loss_tracker = self.test_loss_tracker)        
        ensemble_model.save_model(model_name = generator_model_file_name)
        stats_key = f'generator_{task_num}'
        self.batch_loss_stats[stats_key] = ensemble_model.loss_stats.copy()
        return generator_model_file_name
        
    def update_ensemble(self,method, T_i_1, R_i_1, cur_task_dataloader, val_loader,  full_memory_loaders, test_memory_loaders,  num_epochs_ensemble=60, task_num = 0):
        ensemble_args = self.ensemble_args.copy()  
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  
        ensemble_model_file_name = f'{method}_e_{task_num}_ensemble_{current_time}.pth'
        on_ensemble_model = EnsembleNN(self.feat_args, TravNN, ensemble_args, optim.Adam, self.travnn_args).to(self.device) 
        T_i_1_model = EnsembleNN( self.feat_args, TravNN, ensemble_args, optim.Adam, self.travnn_args).to(self.device)                        
        if task_num > 0:                                   
            T_i_1_model.load_weight(T_i_1)  
            T_i_1_model.load_memory_generator_weight(R_i_1)                    
            on_ensemble_model.load_weight(T_i_1)            
        on_ensemble_model.train_ensemble(self.imostbuffer, task_num, num_epochs_ensemble, cur_task_dataloader, val_loader, full_memory_loaders, test_memory_loaders, model_prefix_name = f'e_online_{task_num}_', T_i_1_model = T_i_1_model, test_loss_tracker=self.test_loss_tracker)                         
        stats_key = f'ensemble_{task_num}'
        self.batch_loss_stats[stats_key] = on_ensemble_model.loss_stats.copy()
        on_ensemble_model.save_model(model_name = ensemble_model_file_name)
        return ensemble_model_file_name

    def stats_load(self, stats_file_path):
        if not stats_file_path.endswith('.npy'):
            stats_file_path = stats_file_path + '.npy'
        stat_dir = os.path.join(get_package_share_directory('align_planner'), 'stats')
        stats_path = os.path.join(stat_dir, stats_file_path)
        self.batch_loss_stats = np.load(stats_path, allow_pickle=True).item()

    def stats_save(self):
        stat_dir = os.path.join(get_package_share_directory('align_planner'), 'stats')
        os.makedirs(stat_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")            
        stats_save_path = os.path.join(stat_dir, f'batch_loss_stats_{current_time}.npy')
        np.save(stats_save_path, self.batch_loss_stats)
        
        print(f"Batch loss statistics saved to {stats_save_path}")
        return stats_save_path
    

    
def run_mcmc_for_method(method, dataloaders, val_loaders, test_loaders,
                    ensemble_args_base, feat_args, travnn_args, device, num_mcmc = 10, num_epoch_ensemble = 2000, num_epoch_generator = 2000):
    for run_idx in range(num_mcmc): 
        print(f"[{method}] Run {run_idx}")
        ensemble_args = ensemble_args_base.copy()
        ensemble_args['method'] = method
        learner = SequentialEnsembleLearner(
            dataloaders, val_loaders, test_loaders,
            ensemble_args, feat_args.copy(), travnn_args.copy(), device
        )
        stats_path = learner.train_loop(method=method,
                                        num_epoch_ensemble=num_epoch_ensemble,
                                        num_epoch_generator=num_epoch_generator)
        print(f"[{method}] Run {run_idx} saved to {stats_path}")

        acc_df, forgetting_df = learner.test_loss_tracker.compute_accuracy_and_forgetting()
        save_dir = os.path.join(get_package_share_directory('align_planner'), f'results_{method}')
        os.makedirs(save_dir, exist_ok=True)    
        results = {
            'acc_df': acc_df,
            'forgetting_df': forgetting_df,
            'ensemble_args': ensemble_args_base,
            'feat_args': feat_args,
            'travnn_args': travnn_args
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'{method}_{timestamp}.pkl')    
        joblib.dump(results, save_path)
        print(f"[Saved] Result summary saved to: {save_path}")


def controlled_run(method, dataloaders, val_loaders, test_loaders,
                   args, feat_args, travnn_args, device, sem):
    with sem:
        run_mcmc_for_method(method, dataloaders, val_loaders, test_loaders,
                            args, feat_args, travnn_args, device)
        
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True) 
    Tasks_dirs = ['park_summer', 'bike_trail', 'sand', 'river', 'forest_summer'] 
    load_dataloaders = False    
    current_date = datetime.now().strftime("%Y%m%d_")                    
    dataloader_dir = os.path.join(get_package_share_directory('align_planner'), 'dataloaders')
    os.makedirs(dataloader_dir, exist_ok=True)
    dataloaders_path = os.path.join(dataloader_dir, f'dataloaders_{current_date}.npy')

    if load_dataloaders and os.path.exists(dataloaders_path):
        print(f"Loading dataloaders from: {dataloaders_path}")
        loaded = torch.load(dataloaders_path)
        dataloaders = loaded['train']
        val_loaders = loaded['val']
        test_loaders = loaded['test']
        tasks_base_dirs = loaded['tasks_base_dirs']
    else:
        print("Generating new dataloaders...")
        dataloaders, val_loaders, test_loaders, tasks_base_dirs = get_dataloaders(Tasks_dirs)            
        torch.save({
            'train': dataloaders,
            'val': val_loaders,
            'test': test_loaders,
            'tasks_base_dirs': tasks_base_dirs
        }, dataloaders_path)
        print(f"Dataloaders saved to: {dataloaders_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                

    feat_args = {
        'world_feat_size': dataloaders[0].dataset.dataset.dim_info['feat_dim'],
        'state_feat_size': dataloaders[0].dataset.dataset.dim_info['state_dim'],
        'deviation_size': dataloaders[0].dataset.dataset.dim_info['deviation_dim'], 
        'hidden_units': [24, 48, 48, 24],
        'latent_size': 6,        
        'z_var': 1.0,   
        'kernel_type': 'rbf',   
        'weight': None,
        'device': device,
        'tensorboard': True        
    }

    travnn_args = {
        'world_feat_size': feat_args['world_feat_size'],
        'state_feat_size': feat_args['state_feat_size'],
        'trav_hidden_units': [16, 32, 32, 16],        
        'trav_output_size': feat_args['deviation_size'],
        'device': device
    }

    ensemble_args = {
        'num_tasks': len(dataloaders),
        'num_models': 5,
        'num_recall': 3,
        'tau_thres': 1.5,
        'lambda_weight': 3e-1,
        'imost_lambda': 5.0,
        'tensorboard': False,   
        'weight': None,
        'method': 'full'
    }

    import time
    multiprocessing.set_start_method("spawn", force=True)

    tau_thres_list = [1.5]
    lambda_weights_list = [0.3]
    imost_lambda_list = [5.0]
    
    methods = ['imost', 'adapt', 'full', 'lwf', 'naivgen', 'proposed']

    active_processes = []

    def launch_and_control(p):
        p.start()
        active_processes.append(p)

        while len(active_processes) >= MAX_PARALLEL:
            for proc in active_processes:
                if not proc.is_alive():
                    proc.join()
                    active_processes.remove(proc)
            time.sleep(0.1) 

    for method in methods:
        print(f"[Start] Running method: {method}")

        if method in ['full']:
            for lambda_ in lambda_weights_list:               
                args_copy = ensemble_args.copy()
                args_copy['method'] = method
                args_copy['lambda_weight'] = lambda_
                p = Process(target=run_mcmc_for_method,
                            args=(method, dataloaders, val_loaders, test_loaders,
                                args_copy, feat_args, travnn_args, device))
                launch_and_control(p)

        elif method in ['imost']:
            for imost_lambda_ in imost_lambda_list:
                for lambda_ in lambda_weights_list:       
                    args_copy = ensemble_args.copy()
                    args_copy['method'] = method
                    args_copy['lambda_weight'] = lambda_                
                    args_copy['imost_lambda'] = imost_lambda_
                    p = Process(target=run_mcmc_for_method,
                                args=(method, dataloaders, val_loaders, test_loaders,
                                    args_copy, feat_args, travnn_args, device))
                    launch_and_control(p)


        elif method in ['adapt']:
            p = Process(target=run_mcmc_for_method,
                        args=(method, dataloaders, val_loaders, test_loaders,
                              ensemble_args.copy(), feat_args, travnn_args, device))
            launch_and_control(p)

        elif method in ['lwf', 'naivgen']:
            for lambda_ in lambda_weights_list:
                args_copy = ensemble_args.copy()
                args_copy['method'] = method
                args_copy['lambda_weight'] = lambda_
                p = Process(target=run_mcmc_for_method,
                            args=(method, dataloaders, val_loaders, test_loaders,
                                  args_copy, feat_args, travnn_args, device))
                launch_and_control(p)

        elif method == 'proposed':
            for tau_ in tau_thres_list:
                for lambda_ in lambda_weights_list:
                    args_copy = ensemble_args.copy()
                    args_copy['method'] = method
                    args_copy['tau_thres'] = tau_
                    args_copy['lambda_weight'] = lambda_
                    p = Process(target=run_mcmc_for_method,
                                args=(method, dataloaders, val_loaders, test_loaders,
                                      args_copy, feat_args, travnn_args, device))
                    launch_and_control(p)

    for p in active_processes:
        p.join()

    print("[Done] All experiments completed.")
