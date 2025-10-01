import sys
import numpy as np
import rclpy
from rclpy.node import Node
import torch
from cv_bridge import CvBridge
from feat_processing.models.networks import resolve_model
from feat_processing.image_parameters  import ImageParameter

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from ament_index_python.packages import get_package_share_directory
import tf2_ros
import time
import os
import threading

def np_euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class FeatureRecordingNode(Node):
    def __init__(self, sensor_name):
        super().__init__("feature_recording_node")
        self.param: ImageParameter = ImageParameter()
        self.param.feature_config.dino_encoder_weight_name = None
        print(self.param.dumps_yaml())        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        self.feature_extractor = None
        self.semantic_model = None
        self.cv_bridge = CvBridge()
        self.initialize_model()

        self.features = None
        self.buffer = []
        self.prev_odom = None
        self.max_buffer_size = 30  

        self.buffer_save_path = os.path.join(get_package_share_directory('feat_processing'), 'data')
      
        os.makedirs(self.buffer_save_path, exist_ok=True)
        self.get_logger().info(f"Buffer save path set to: {self.buffer_save_path}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.P = None
        self.header = None
        self.register_sub_pub()
        self.prediction_img = None
        




    def initialize_model(self):
        if self.param.feature_extractor:
            self.feature_extractor = resolve_model(self.param.feature_config.name, self.param.feature_config)
            self.get_logger().info("initialize_model is done")

    def register_sub_pub(self):
        """Register publishers and subscribers."""
     
        if self.param.camera_info_topic is not None and self.param.resize is not None:
            
            self.create_subscription(
                CameraInfo,
                self.param.camera_info_topic,
                self.image_info_callback,
                10
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
        ##
        ##
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
        if len(cv_image.shape) == 3: 
            cv_image = cv_image.transpose(2, 0, 1)
        elif len(cv_image.shape) == 2:
            cv_image = cv_image.reshape((1,) + cv_image.shape)        
        cv_image = cv_image / 255.0 
      
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
        
        feat_end_time = time.time()
        feat_duration = feat_end_time - start_time  
        if feat_duration > 0.025:
            self.get_logger().info(f"featu_processing took {feat_duration:.6f} seconds")
            

        if self.param.semantic_segmentation:
            raise NotImplementedError("Semantic segmentation is not yet implemented.")            
        
  
        end_time = time.time()  
        duration = end_time - feat_end_time  
        if duration > 0.005:
            self.get_logger().info(f"record_feature single iter took {duration:.6f} seconds")


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
            self.features = self.feature_extractor["model"].gen_train_sample(image).squeeze()
            self.update_buffer()
            
    def update_buffer(self):
        current_odom = self.get_current_odometry()
        if current_odom is not None:
            if self.prev_odom is None:
                self.buffer.append(self.features)
                self.prev_odom = current_odom

            if self.has_moved(current_odom, self.prev_odom):
                self.buffer.append(self.features)
                self.prev_odom = current_odom
                if len(self.buffer) >= self.max_buffer_size:
                    threading.Thread(target=self.save_buffer, args=(self.buffer.copy(),)).start()
                    self.buffer = []
                

    
    def has_moved(self, current_odom, prev_odom):
        position_diff = torch.norm(current_odom['position'] - prev_odom['position'])
        cur_rpy = np_euler_from_quaternion(current_odom['orientation'])
        prev_rpy = np_euler_from_quaternion(prev_odom['orientation'])
        yaw_diff = abs(cur_rpy[-1] - prev_rpy[-1])        
        return position_diff > 0.1 or yaw_diff > 0.1  # Adjust the threshold as needed


    def get_current_odometry(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            position = torch.tensor([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            orientation = torch.tensor([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            return {'position': position, 'orientation': orientation}
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("TF lookup failed")
            return None

    def save_buffer(self, data):
        # Save the buffer to disk
        timestamp = int(time.time())
        save_path = os.path.join(self.buffer_save_path, f'buffer_{timestamp}.pt')
        torch.save(data, save_path)
        self.get_logger().info(f"Buffer saved to {save_path}")

def main(args=None):
    rclpy.init(args=args)
    sensor_name = sys.argv[1] if len(sys.argv) > 1 else 'default_sensor'
    node = FeatureRecordingNode(sensor_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()