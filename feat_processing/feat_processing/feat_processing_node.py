import sys
import numpy as np
import rclpy
from rclpy.node import Node
import torch
from cv_bridge import CvBridge
from feat_processing.models.networks import resolve_model
from feat_processing.image_parameters  import ImageParameter
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from elevation_map_msgs.msg import ChannelInfo
from sklearn.decomposition import PCA
import torch.nn.functional as NF
import time
from rclpy.qos import qos_profile_sensor_data

class FeatureProcessingNode(Node):
    def __init__(self, sensor_name):
        super().__init__("feature_processing_node")
        self.param: ImageParameter = ImageParameter()
        print(self.param.dumps_yaml())
        self.semseg_color_map = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
        self.feature_extractor = None
        self.semantic_model = None
        self.cv_bridge = CvBridge()
        self.initialize_model()

        
        self.P = None
        self.header = None
        self.register_sub_pub()
        self.prediction_img = None

        

    def initialize_model(self):
        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)
            self.get_logger().info("initialize_model is done")

    def register_sub_pub(self):
    
        if self.param.camera_info_topic is not None and self.param.resize is not None:
            
            self.create_subscription(
                CameraInfo,
                self.param.camera_info_topic,
                self.image_info_callback,
                10
            )
            
            self.feat_im_info_pub = self.create_publisher(
                CameraInfo,
                self.get_name() + self.param.camera_info_topic + "_resized",
                qos_profile_sensor_data
            )

        if "compressed" in self.param.image_topic:
            self.compressed = True
            self.create_subscription(
                CompressedImage,
                self.param.image_topic,
                self.image_callback,
                10
            )
        else:
            self.compressed = False
            self.create_subscription(
                Image,
                self.param.image_topic,
                self.image_callback,
                10
            )

    # Publishers
        if self.param.semantic_segmentation:
            # TODO: not yet needed
            raise NotImplementedError("Semantic segmentation is not yet implemented.")
            
        if self.param.feature_extractor:
            self.feature_pub = self.create_publisher(
                Image,
                self.get_name() + "/" + self.param.feature_topic,
                qos_profile_sensor_data
            )
            self.feat_im_pub = self.create_publisher(
                Image,
                self.get_name() + "/" + self.param.feat_image_topic,
                qos_profile_sensor_data
            )
            self.feat_channel_info_pub = self.create_publisher(
                ChannelInfo,
                self.get_name() + "/" + self.param.feat_channel_info_topic,
                qos_profile_sensor_data
            )



    def image_info_callback(self, msg):
        """Callback for camera info.

        Args:
            msg (CameraInfo):
        """
     
        msp_p = msg.p
        msp_p[0] = msg.k[0] 
        msp_p[2] = msg.k[2] 
        msp_p[5] = msg.k[4] 
        msp_p[6] = msg.k[5] 
        self.P = np.array(msp_p).reshape(3, 4)
      
        self.height = int(self.param.resize * msg.height)
        self.width = int(self.param.resize * msg.width)
        self.info = msg
        self.info.height = self.height
        self.info.width = self.width
        
        self.P[:2, :3] = self.P[:2, :3] * self.param.resize
        self.info.k = self.P[:3, :3].flatten().tolist()
        self.info.p = self.P.flatten().tolist()


    def convert_image_to_tensor(self, cv_image):
        cv_image = cv_image.astype(np.float32)        
        if len(cv_image.shape) == 3:  # transpose image-like data
            cv_image = cv_image.transpose(2, 0, 1)
        elif len(cv_image.shape) == 2:
            cv_image = cv_image.reshape((1,) + cv_image.shape)        
        cv_image = cv_image / 255.0 # normalization of rgb images
        torch_tensor = torch.as_tensor(cv_image, device = self.device)        
        return torch_tensor
    
    def image_callback(self, rgb_msg):

        if self.P is None:
            return
        start_time = time.time()

        if self.compressed:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb_msg)
          
            torch_tensor = self.convert_image_to_tensor(image)
        else:
            image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")            
            torch_tensor = self.convert_image_to_tensor(image)
        self.header = rgb_msg.header
        self.process_image(torch_tensor)
        

        if self.param.semantic_segmentation:
            raise NotImplementedError("Semantic segmentation is not yet implemented.")            
        if self.param.feature_extractor:
            self.publish_feature()
            if self.param.feat_image_publish:
                self.publish_feature_image(self.features)
            self.publish_channel_info([f"feat_{i}" for i in range(self.features.shape[0])], self.feat_channel_info_pub)

        if self.param.resize is not None:
            self.feat_im_info_pub.publish(self.info)
        
        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        if duration > 0.05:
            self.get_logger().info(f"feature processing took {duration:.6f} seconds")


    def process_image(self, image):
        """Depending on setting generate color, semantic segmentation or feature channels.
        Args:
            image:
            u:
            v:
            points:
        """
        if self.param.semantic_segmentation:
            # TODO: not yet needed
            raise NotImplementedError("Semantic segmentation is not yet implemented.")              
        if self.param.feature_extractor:
            im_size = image.shape[-2:]            
            self.features = self.feature_extractor["model"](image)        
            if self.param.resize is not None:   
                im_size = [int(im_size[0] * self.param.resize), int(im_size[1] * self.param.resize)]    
            self.features = NF.interpolate(self.features, im_size, mode=self.param.feature_config.interpolation, align_corners=False)
            self.features = self.features.squeeze()

    def publish_feature(self):        
        if len(self.features.shape) ==2 : 
            self.features = self.features.unsqueeze(0)
        features = self.features[:,:,:]
        img = features.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)                
 
        feature_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="passthrough")        
        feature_msg.header.frame_id = self.header.frame_id
      
        feature_msg.header.stamp = self.header.stamp
        self.feature_pub.publish(feature_msg)



    def publish_feature_image(self, features):
        data = np.reshape(features.cpu().detach().numpy(), (features.shape[0], -1)).T
        n_components = 3
        pca = PCA(n_components=n_components).fit(data)
        pca_descriptors = pca.transform(data)
        img_pca = pca_descriptors.reshape(features.shape[1], features.shape[2], n_components)
        comp = img_pca  # [:, :, -3:]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        comp_img = (comp_img * 255).astype(np.uint8)        
        
        feat_msg = self.cv_bridge.cv2_to_imgmsg(comp_img, encoding="passthrough")
        feat_msg.header.frame_id = self.header.frame_id
        feat_msg.header.stamp = self.header.stamp
        self.feat_im_pub.publish(feat_msg)

    def publish_channel_info(self, channels, pub):
        """Publish fusion info."""
        info = ChannelInfo()
        info.header = self.header
        info.channels = channels
        pub.publish(info)

def main(args=None):
    rclpy.init(args=args)
    sensor_name = sys.argv[1] if len(sys.argv) > 1 else 'default_sensor'
    node = FeatureProcessingNode(sensor_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()