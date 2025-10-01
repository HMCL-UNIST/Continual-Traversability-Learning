import torch
from torch.utils.data import Dataset
import os
from ament_index_python.packages import get_package_share_directory
import random

class GridStateDataset(Dataset):
    def __init__(self, dataset_paths, data_load = False, compute_normalizing_constants = True):        
        self.dim_info = None
        self.data = []     
        self.labels = []      
        self.state_input_data = []     
        self.feat_input_data = []
        self.deviation_output_data = []
        self.debug_t_state = []
        self.debug_t_grid = []
        self.plot_path = os.path.join(get_package_share_directory('align_planner'), 'plot')
        os.makedirs(self.plot_path, exist_ok=True)
        self.normalization_path = os.path.join(get_package_share_directory('align_planner'), 'data')
        os.makedirs(self.normalization_path, exist_ok=True)
        if data_load:
            self.env_file_map = self.get_files_grouped_by_env(dataset_paths)            
            self.load_by_env()        
            
        if compute_normalizing_constants or not os.path.exists(os.path.join(self.normalization_path, 'normalize_constants.pkl')):
            self.compute_batch_statistics()
        else:
            self.load_normalizing_constants()            
        
        if len(self.state_input_data) > 0:  
            self.normalize()
            self.ready_input_data()

    def load_by_env(self):
        for env_name, file_list in self.env_file_map.items():
            pt_count = 0
            pt_count_max = 100
            random.shuffle(file_list)
            for file_path in file_list:                
                if pt_count > pt_count_max:
                    break
                pt_count+=1
                data = torch.load(file_path, map_location="cpu")                  
                print(f"File {file_path} is loaded with {data['state_length']} entries")
                for state_idx, state_data in enumerate(data['state_data']):
                    if state_idx < data['state_length']-1:       
                        self.debug_t_state.append(data['state_data'][state_idx+1].t - data['state_data'][state_idx].t)
                        for grid_idx, grid_data in enumerate(data['grid_data']):        
                            self.debug_t_grid.append(data['grid_data'][grid_idx].t - data['state_data'][state_idx].t)
                            feat_input, state_input, deviation_output = grid_data.features_on_states(data['state_data'][state_idx], data['state_data'][state_idx+1])
                            if feat_input is None:
                                continue
                            if torch.isnan(feat_input).any() or torch.isnan(state_input).any() or torch.isnan(deviation_output).any():
                                print(f"NaN value detected in the data at file: {file_path}, state index: {state_idx}, grid index: {grid_idx}")
                                continue
                            if feat_input is not None:                                                        
                                self.feat_input_data.append(feat_input)
                                self.state_input_data.append(state_input)
                                self.deviation_output_data.append(deviation_output)     
                                self.labels.append(env_name)
                
        self.feat_input_data = torch.stack(self.feat_input_data).cpu()
        self.state_input_data = torch.stack(self.state_input_data).cpu()                
        self.deviation_output_data = torch.stack(self.deviation_output_data).cpu()
        

        
        self.dim_info = {'feat_dim': self.feat_input_data.shape[1], 
                         'state_dim': self.state_input_data.shape[1],
                         'deviation_dim': self.deviation_output_data.shape[1]}  
        print("data loaded with len : ", self.feat_input_data.shape[0])
                
                
    

    def get_files_grouped_by_env(self, folder_names, extension='.pt'):
        env_file_map = {}        
        for folder_path in folder_names:            
            file_list = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(extension):
                        file_list.append(os.path.join(root, file))
            folder_name =  os.path.basename(folder_path)
            env_file_map[folder_name] = file_list
        return env_file_map
        
    def ready_input_data(self):
        self.input_data = torch.cat((self.feat_input_data, self.state_input_data), dim=1)   
        
    def load_normalizing_constants(self):
        self.constants = torch.load(os.path.join(self.normalization_path, 'normalize_constants.pkl'))
        print("Normalizing constants loaded.")
        
     

    def __len__(self):
        return self.feat_input_data.shape[0]

    def __getitem__(self, idx):        
        return self.input_data[idx,:], self.deviation_output_data[idx,:]

    

    def normalize(self, constants = None):
        if constants is None:
            constants  = self.constants 
        self.feat_input_data = (self.feat_input_data - constants['feat_mean']) / (constants['feat_std']+1e-9)
        self.state_input_data = (self.state_input_data - constants['state_mean']) / (constants['state_std']+1e-9)
        self.deviation_output_data = (self.deviation_output_data - constants['deviation_mean']) / (constants['deviation_std']+1e-9)        
        print("Data normalized.")
    
        
    def compute_batch_statistics(self):                
        state_data= self.state_input_data        
        state_batch_data_mean = torch.mean(state_data, dim=0)
        state_batch_data_std = torch.std(state_data, dim=0)
        
        feat_data= self.feat_input_data        
        feat_batch_data_mean = torch.mean(feat_data, dim=0)
        feat_batch_data_std = torch.std(feat_data, dim=0)
                
        deviation_data = self.deviation_output_data
        deviation_batch_data_mean = torch.mean(deviation_data, dim=0)
        deviation_batch_data_std = torch.std(deviation_data, dim=0)
        
        self.constants = {
            'state_mean': state_batch_data_mean,
            'state_std': state_batch_data_std,
            'feat_mean': feat_batch_data_mean,
            'feat_std': feat_batch_data_std,
            'deviation_mean': deviation_batch_data_mean,
            'deviation_std': deviation_batch_data_std
        }
        
        save_path = os.path.join(self.normalization_path, 'normalize_constants.pkl')
        torch.save(self.constants, save_path)        
        
        print("Batch statistics computed, and saved in ", save_path)
            


