import rclpy
import torch
import os
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
from torch.cuda.amp import autocast
from feat_processing.image_parameters import FeatureExtractorParameter   
from ament_index_python.packages import get_package_share_directory
from feat_processing.models.autoencoder import AutoEncoder  
from feat_processing.train.datasets import BackendFeatDataset   



def resolve_model(name, config=None):
    """Get the model class based on the name of the pretrained model.

    Args:
        name (str): Name of pretrained model
        [fcn_resnet50,lraspp_mobilenet_v3_large,detectron_coco_panoptic_fpn_R_101_3x]

    Returns:
        Dict[str, str]:
    """
    if  name == "DINOv2":
        model = DINOv2Model(config)        
        if config.model_compile:
            rclpy.logging.get_logger('feat_processing').info('DinoV2 Model compile begins')
            model = torch.compile(model)
            rclpy.logging.get_logger('feat_processing').info('DinoV2 Model compile ends')
        return {
            "name": config.model + str(config.patch_size),
            "model": model,
        }
    else:
        raise NotImplementedError


class DINOv2Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg: FeatureExtractorParameter = cfg
        self._model_type = self.cfg.model        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tmp_dir = ['data']
        self.encoder_data_set = BackendFeatDataset(data_dirs = tmp_dir, data_load =False,compute_normalizing_constants=False)
        
        if self.cfg.apply_pca:            
            self.load_pca_constants()
        else:
            self.load_encoder(self.cfg.dino_encoder_weight_name)
            rclpy.logging.get_logger('feat_processing').info('Autoencoder network loaded')
        
        # Initialize DINOv2
        if self._model_type == "vit_small":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.embed_dim = 384
        elif self._model_type == "vit_base":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', model_dir='~/.torch/')
            self.embed_dim = 768
        elif self._model_type == "vit_large":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', model_dir='~/.torch/')
            self.embed_dim = 1024
        elif self._model_type == "vit_huge":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', model_dir='~/.torch/')
            self.embed_dim = 1536
        self.patch_size = self.cfg.patch_size        
        
        self.n_feats = self.embed_dim
        self.model.to(self.device)
        self.model.eval()
        
        self.transform=self._create_transform()

    def load_pca_constants(self):
        base_path = os.path.join(get_package_share_directory('feat_processing'), 'models')
        pca_save_path = os.path.join(base_path, f'pca.pkl')    
        pca_dict = torch.load(pca_save_path)
        self.pca_mean = torch.tensor(pca_dict['mean']).cuda()
        self.pca_components = torch.tensor(pca_dict['components']).cuda()
        print("pca constants loaded")
        rclpy.logging.get_logger('feat_processing').info('pca constants loaded')
        
    def load_encoder(self, weights = None):
        Autoencoder_model = AutoEncoder(device=self.device)            
        if weights is None:                        
            self.encoder_data_set = None         
        else:             
            Autoencoder_model.load_weight(weights)        
        self.mlp = Autoencoder_model.encoder_layers
    
    def to_tensor(self, data):
       
        data = data.astype(np.float32)
        if len(data.shape) == 3:  # transpose image-like data
            data = data.transpose(2, 0, 1)
        elif len(data.shape) == 2:
            data = data.reshape((1,) + data.shape)
        if len(data.shape) == 3 and data.shape[0] == 3:  # normalization of rgb images
            data = data / 255.0
        tens = torch.as_tensor(data, device="cuda")
        return tens
    
    def get_sample_input_for_jit(self):        
        return torch.zeros([3,self.cfg.input_size[0], self.cfg.input_size[1]]).to(self.device)
        
    def _create_transform(self):                
        transform = T.Compose([
        T.Resize(size=(self.cfg.input_size[0], self.cfg.input_size[1]), antialias=True), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform
    
    def forward(self, image):
        image = self.transform(image)
        image=image.unsqueeze(0)
    
        with torch.amp.autocast('cuda'), torch.no_grad():
            feat = self.model.forward_features(image)["x_norm_patchtokens"]    
            if self.encoder_data_set is not None:        
                feat = self.encoder_data_set.normalize(feat)
            if self.cfg.apply_pca:
                feat = feat - self.pca_mean
                feat = feat @ self.pca_components.T
            else:
                feat = self.mlp(feat)
                
        B=feat.shape[0]
        C=feat.shape[2]
        H=int(image.shape[2]/self.patch_size)
        W=int(image.shape[3]/self.patch_size)
        feat=feat.permute(0,2,1)
        feat=feat.reshape(B,C,H,W)
        return feat


    def gen_train_sample(self, image):
        im_size = image.shape[-2:]
        image = self.transform(image)
        image=image.unsqueeze(0)
        with autocast(), torch.no_grad():
            feat = self.model.forward_features(image)["x_norm_patchtokens"]                    
        return feat
           
    