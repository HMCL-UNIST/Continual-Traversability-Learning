
import rclpy
from rclpy.node import Node
import torch
import time
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from align_planner.align_planner_parameters import MppiPlannerConfig
from nav_msgs.msg import Odometry
import threading
from grid_map_msgs.srv import GetGridMap  
import os
from align_planner.utils.ptype import VehicleState, GridData
from align_planner.utils.buffers import GridDataBuffer, VehicleStateBuffer
from align_planner.utils.tractionEKF import EKF
from ament_index_python.packages import get_package_share_directory
import numpy as np
from rcl_interfaces.msg import SetParametersResult  


class DataLoggerNode(Node):
    def __init__(self):
        super().__init__('data_logger_node',start_parameter_services= True, allow_undeclared_parameters = True, automatically_declare_parameters_from_overrides=True)        
        self.recording_enabled = False               
        self.declare_parameter('data_recording_enabled', False)
        self.declare_parameter('enable_random_ctrl_pub', False)
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.params = self.init_param()
        self.grid_record_Hz = 2
        self.state_record_Hz = 10
        buffer_recording_sec = 10
        self.target_vel = 0.0
        self.target_omega = 0.0
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                                
        self.cur_state = VehicleState()
        self.ekf = EKF()
        self.cur_grid = GridData()
        self.grid_buffer = GridDataBuffer(buffer_length=buffer_recording_sec * self.grid_record_Hz)
        self.state_buffer = VehicleStateBuffer(buffer_length=buffer_recording_sec * self.state_record_Hz)       
        
        data_directory = self.get_parameter('save_dir').get_parameter_value().string_value
        self.buffer_save_path = os.path.join(get_package_share_directory('align_planner'), data_directory)
        
        os.makedirs(self.buffer_save_path, exist_ok=True)
         
        
        self.grid_map_client = self.create_client(GetGridMap, 'get_raw_submap')   
        while not self.grid_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the get_grid_map service...')
            
        self.init_state = False
        self.cur_cmd_time = None
        self.cur_odom = None
        self.cur_cmd = None
        self.cur_joy = None
        self.register_sub_pub()                
        
    def init_param(self):
        py_param_names = MppiPlannerConfig.get_param_names()
        py_param_class: MppiPlannerConfig = MppiPlannerConfig()        
        params =  py_param_class.to_dict()   
        ros_param_list = self.get_parameters(py_param_names)
        ros_param_dict = MppiPlannerConfig.create_hierarchical_dict(ros_param_list)        
        MppiPlannerConfig.update_dict_recursive(params, ros_param_dict)
        return params
         

    def update_control_values(self):
        # Generate random smooth control signals
        self.linear_ax = np.random.uniform(-1.0, 1.0)  # Random linear velocity between 0 and 1
        self.angular_alpha_z = np.random.uniform(-1.0, 1.0)  # Random angular velocity between -0.5 and 0.5        
        self.target_vel = np.random.uniform(0.1, 1.0)  # Random target velocity between 0 and 1
        self.target_omega = np.random.uniform(-0.5, 0.5)  # Random target angular velocity between -0.5 and 0.5



    def cmd_timer_callback(self):        
        twist = Twist()        
        
        distance_to_lin_vel = abs(self.linear_velocity - self.target_vel)
        if distance_to_lin_vel > 0.2:
            if self.linear_velocity < self.target_vel:            
                self.linear_velocity = self.linear_velocity + abs(self.linear_ax) / self.control_hz        
            elif self.linear_velocity > self.target_vel:
                self.linear_velocity = self.linear_velocity - abs(self.linear_ax) / self.control_hz        
            
        self.linear_velocity = np.clip(self.linear_velocity, 0.0, 1.0)
        if self.linear_velocity <= 0.0:
            self.target_vel = np.random.uniform(0.3, 1.0)  # Random target velocity between 0 and 1
            self.linear_ax = np.random.uniform(0.5, 1.0)  # Random linear velocity between 0 and 1
            self.linear_velocity = self.linear_velocity + self.linear_ax / self.control_hz        
            self.linear_velocity = np.clip(self.linear_velocity, 0.0, 1.0)
        
        distance_to_ang_vel = abs(self.angular_velocity - self.target_omega)
        if distance_to_ang_vel > 0.2:
            if self.target_omega < self.angular_velocity:
                self.angular_velocity = self.angular_velocity - abs(self.angular_alpha_z) / self.control_hz
            elif self.target_omega > self.angular_velocity:
                self.angular_velocity = self.angular_velocity + abs(self.angular_alpha_z) / self.control_hz
        
        self.angular_velocity = np.clip(self.angular_velocity, -1.0, 1.0)
        twist.linear.x = self.linear_velocity
        twist.linear.y = self.linear_ax
        
        twist.angular.z = self.angular_velocity
        twist.angular.y = self.angular_alpha_z
        
        
        if self.enable_random_ctrl:
            self.contorl_pub.publish(twist)
            twist_odom = Odometry()
            twist_odom.header.stamp = self.get_clock().now().to_msg()
            twist_odom.twist.twist = twist
            self.cmd_odom_pub.publish(twist_odom)
            
    def buffers_clear(self):
        self.state_buffer.clear()
        self.grid_buffer.clear()

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'data_recording_enabled':
                self.recording_enabled = param.value
                self.get_logger().info(f'Recording enabled set to: {self.recording_enabled}')
                self.buffers_clear()
                
                
            if param.name == 'enable_random_ctrl_pub':
                self.enable_random_ctrl = param.value
                self.get_logger().info(f'pub random ctrl to: {self.enable_random_ctrl}')
            self._update_param(param, self.params)        
                
        return SetParametersResult(successful=True)


    def _update_param(self, param, params):
        """General function to update hierarchical parameters dynamically."""
        keys = param.name.split('.')
       
        # Traverse hierarchy to find the correct parameter
        for key in keys[:-1]:
            params = params.get(key, {})

        # Update the parameter value
        if keys[-1] in params:
            params[keys[-1]] = param.value
            self.get_logger().info(f"Updated parameter {param.name} to {param.value}")

    def register_sub_pub(self):        
        self.get_logger().info("register sub and pub is done")
       
        self.joy_subscriber = self.create_subscription(
            Joy,
            '/j100_0408/joy_teleop/joy',
            self.joy_callback, qos_profile=qos_profile_sensor_data
        )
        

        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/dlio/odom_node/odom',
            self.odom_callback,
            qos_profile=qos_profile_sensor_data
        )
        
        self.cmd_subscriber = self.create_subscription(
            Twist,
            '/j100_0408/platform/cmd_vel_unstamped',
            self.cmd_callback,
            qos_profile=qos_profile_sensor_data
        )

        
        
   
        self.grid_record_timer = self.create_timer(1.0 / self.grid_record_Hz, self.grid_record_callback)
        self.state_record_timer = self.create_timer(1.0 / self.state_record_Hz, self.state_record_callback)
        self.ekf_timer = self.create_timer(1.0 / 10, self.ekf_callback)
        
        self.traction_pub = self.create_publisher(Odometry, '/traction_msg',10)
        self.cmd_odom_pub = self.create_publisher(Odometry, '/cmd_odom_twist',10)
        
        
        self.enable_random_ctrl = False
        self.contorl_pub = self.create_publisher(Twist, '/mppi/cmd_vel',10)
        self.control_update_hz = 0.5
        self.control_hz = 10
        self.cmd_timer = self.create_timer(1/self.control_hz, self.cmd_timer_callback)        
        self.update_control_timer = self.create_timer(1/self.control_update_hz, self.update_control_values)
    
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.linear_ax = 0.0
        self.angular_alpha_z = 0.0
    
    def joy_callback(self,msg):
        self.cur_joy = msg
        if self.cur_joy.buttons[5] == 0:                        
            if self.cur_joy.buttons[4] == 0: 
                self.buffers_clear()

    def odom_callback(self, msg):
        self.cur_odom = msg                
        self.cur_state.update_from_odom(odom_msg=msg, is_body_frame = False)   
   

    def cmd_callback(self, msg):
        self.cur_cmd = msg
        self.cur_cmd_time = self.get_clock().now()     
        self.cur_state.update_from_cmd(twist_msg=msg)        
                

    def ekf_callback(self):
        if self.init_state is False:
            self.ekf.reset(self.cur_state)   
            self.ekf.set_ctrl(self.cur_state)        
            self.ekf.predict()
            self.init_state = True
        else:                 
            self.ekf.update_from_vehicleState(self.cur_state)        
        self.ekf.set_ctrl(self.cur_state)
        self.ekf.predict()

    def state_record_callback(self):
        if self.cur_odom is not None and self.cur_cmd_time is not None and self.cur_joy is not None:                 
            odom_time = self.cur_odom.header.stamp.sec + self.cur_odom.header.stamp.nanosec * 1e-9
            cmd_time = self.cur_cmd_time.seconds_nanoseconds()[0] + self.cur_cmd_time.seconds_nanoseconds()[1] * 1e-9
            time_diff = odom_time - cmd_time
            
            self.cur_state.update_from_odom_and_cmd(self.cur_odom, self.cur_cmd, is_body_frame = False)
            self.cur_state.update_traction(self.ekf.state[-2], self.ekf.state[-1])

            traction_msg = self.ekf.get_odom(self.cur_odom.header)
            self.traction_pub.publish(traction_msg)
            
            self.state_buffer.push(self.cur_state.copy())                        
            
            if abs(time_diff) > 1.0 / self.state_record_Hz:                
                self.get_logger().info(f'Time difference between odom and cmd: {time_diff} seconds, violating the recording frequency')        
                  
            

            if self.state_buffer.is_full() and len(self.grid_buffer) > 0 and   self.recording_enabled:                
                data_save_path = os.path.join(self.buffer_save_path, f'grid_state_data_{self.cur_odom.header.stamp.sec}.pt')
                self.get_logger().info(f"Saving data to {data_save_path}")

                self.ctrl_timer_thread = threading.Thread(target=self.save_data, args = (data_save_path,))
                self.ctrl_timer_thread.start()
                
    
    def grid_record_callback(self):     
        self.request_subgrid_map()
        
        
    def request_subgrid_map(self):                
        if self.cur_odom is not None:                          
            request = GetGridMap.Request()        
            request.frame_id = "map"
            request.position_x = self.cur_odom.pose.pose.position.x   # Center x-coordinate of subgrid in the map frame
            request.position_y = self.cur_odom.pose.pose.position.y   # Center y-coordinate of subgrid in the map frame            
            request.length_x =  float(self.params['Map_config']['map_size'])   # Width of the subgrid
            request.length_y =  float(self.params['Map_config']['map_size'])    # Height of the subgrid            
            request.layers = self.params['Map_config']['layers']  # Specify map layers to include
               
            future = self.grid_map_client.call_async(request)            
            future.add_done_callback(self.callback_response)

    def callback_response(self, future):        
        try:            
            response = future.result()            
            self.grid_map_process(response.map)                        
        except Exception as e:
            self.get_logger().error(f"Grid map Service call failed: {e}")
            
    def grid_map_process(self, grid_map_msg):      
        grid_resolution = grid_map_msg.info.resolution
        if abs(self.params['Map_config']['map_res']-grid_resolution) > 1e-3:
            raise AssertionError(f"Grid size is not matched with the map resolution. Please change the param. Expected: {self.params['Map_config']['map_res']}, Got: {grid_resolution}")
            
        grid_start_time = time.time()              
        self.cur_grid.update_from_grid_map(grid_map_msg)        
        self.grid_buffer.push(self.cur_grid)                    
        grid_end_time = time.time()
        time_diff = grid_end_time - grid_start_time
        if abs(time_diff) > 1.0 / self.grid_record_Hz:
            self.get_logger().info(f"Grid map processing time: {time_diff} seconds, violating the recording frequency")
        
    def save_data(self, data_save_path):       
        data_dict = {
            'state_data': self.state_buffer.get_data(),
            'grid_data': self.grid_buffer.get_data(),
            'state_length': len(self.state_buffer.data_list),
            'grid_length': len(self.grid_buffer.data_list)
        }
        torch.save(data_dict,data_save_path)
        self.buffers_clear()        

    
                

def main(args=None):
    rclpy.init(args=args)        
    torch.cuda.empty_cache()    
    node = DataLoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()