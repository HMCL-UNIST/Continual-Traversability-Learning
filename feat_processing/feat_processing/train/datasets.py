import torch
from torch.utils.data import Dataset
import os
import pickle
from ament_index_python.packages import get_package_share_directory


class BackendFeatDataset(Dataset):
    def __init__(self, data_dirs, data_load = False, compute_normalizing_constants = True):        
        self.data = []        
        self.dataset_path = os.path.join(get_package_share_directory('feat_processing'), 'data')
        if data_load:
            self.file_list = self.get_files_from_folders(data_dirs)
            self.load_data()            
            
        self.normalize_constants = None   
        if compute_normalizing_constants or not os.path.exists(os.path.join(self.dataset_path, 'normalize_constants.pkl')):
            self.compute_batch_statistics()
        else:
            self.load_normalizing_constants()

    def get_files_from_folders(self, folder_names, package_name = None, extension='.pt'):
        if package_name is None:
            package_name = 'feat_processing'
        buffer_save_paths = []
        for folder_name in folder_names:
            buffer_save_path = os.path.join(get_package_share_directory(package_name), folder_name)
            buffer_save_paths.append(buffer_save_path)
        
        return self.get_all_files_from_directories(buffer_save_paths, extension)

    def get_all_files_from_directories(self,directories, extension='.pt'):
        file_list = []
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(extension):
                        file_list.append(os.path.join(root, file))
        return file_list
    
    
    def load_normalizing_constants(self):
        with open(os.path.join(self.dataset_path, 'normalize_constants.pkl'), 'rb') as f:
            self.normalize_constants = pickle.load(f)
            self.normalize_constants['mean'] = self.normalize_constants['mean'].cuda()
            self.normalize_constants['std'] = self.normalize_constants['std'].cuda()
            
    def load_data(self):        
        for file_path in self.file_list:
            data = torch.load(file_path)
            print(f"File {file_path} is loaded with {len(data)} entries")
            self.data.extend(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    

    def normalize(self, data, constants = None):
        if constants  is None:
            constants  = self.normalize_constants            
        mean = constants['mean']
        std = constants['std']        
        new_data = (data - mean) / std        
        return new_data

    def save_normalize_constants(self, constants, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(constants, f)

    def compute_batch_statistics(self):
        data = torch.stack(self.data)
        batch_data = data.view(-1, data.shape[-1])
        batch_data_mean = torch.mean(batch_data, dim=0)
        batch_data_std = torch.std(batch_data, dim=0)
        constants = {'mean': batch_data_mean, 'std': batch_data_std}
        save_path = os.path.join(self.dataset_path, 'normalize_constants.pkl')
        self.save_normalize_constants(constants, save_path)
        self.data = [self.normalize(d, constants) for d in self.data]


    
    
def load_all_buffers(dataset_path):
    all_data = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(dataset_path, filename)
            data = torch.load(file_path)
            all_data.extend(data)
    return all_data

