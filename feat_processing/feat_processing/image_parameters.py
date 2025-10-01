from dataclasses import dataclass, field
from simple_parsing.helpers import Serializable


@dataclass
class FeatureExtractorParameter(Serializable):
    name: str = "DINOv2"
    interpolation: str = "bilinear"
    model: str = "vit_small"
    patch_size: int = 14
    dim: int = 1
    dropout: bool = False
    dino_feat_type: str = "feat"
    projection_type: str = "nonlinear"
    input_size: list = field(default_factory=lambda: [168, 308])        
    pcl: bool = False
    model_compile: bool = True            
    apply_pca: bool = False
    dino_encoder_weight_name: str = "autoencoder_epoch_100.pth"



@dataclass
class ImageParameter(Serializable):
    
    semantic_segmentation: bool = False
    feat_image_publish: bool = False
    segmentation_model: str = "detectron_coco_panoptic_fpn_R_101_3x"
    show_label_legend: bool = False
    channels: list = field(default_factory=lambda: ["grass", "road", "tree", "sky"])

    image_topic: str = "/front/zed_node/rgb/image_rect_color"
    publish_topic: str = "feat_image"
    publish_image_topic: str = "feat_image_debug"
    camera_info_topic: str = "/front/zed_node/rgb/camera_info"
    channel_info_topic: str = "channel_info"

    
    feature_extractor: bool = True
    feature_config: FeatureExtractorParameter = field(default_factory=FeatureExtractorParameter)

    feature_topic: str = "semantic_feat"
    feat_image_topic: str = "semantic_feat_im"
    feat_channel_info_topic: str = "feat_channel_info"
    resize: float = 0.4
    
