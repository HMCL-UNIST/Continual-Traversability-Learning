import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
import queue    
import jax
import jax.numpy as jnp
import cupy as cp
import torch
from align_planner.align_planner_parameters import MppiPlannerConfig
from align_planner.mppi.jax_mppi import JAXMPPI    
from align_planner.utils.ptype import VehicleState, GridData
from align_planner.utils.ros2_utils import get_goal_marker, get_jax_array_msg
import time
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default, qos_profile_action_status_default
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from grid_map_msgs.srv import GetGridMap  
import threading
from collections import deque

class AlignPlannerNode(Node):
    def __init__(self, planner_name):
        super().__init__('align_planner_node',start_parameter_services= True, allow_undeclared_parameters = True, automatically_declare_parameters_from_overrides=True)
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.params = self.init_param()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                                        
        self.initialize_model()        
        self.register_sub_pub()                
    
    def init_param(self):
        py_param_names = MppiPlannerConfig.get_param_names()
        py_param_class: MppiPlannerConfig = MppiPlannerConfig()        
        params =  py_param_class.to_dict()   
        ros_param_list = self.get_parameters(py_param_names)
        ros_param_dict = MppiPlannerConfig.create_hierarchical_dict(ros_param_list)        
        MppiPlannerConfig.update_dict_recursive(params, ros_param_dict)
        return params
         

    def parameter_callback(self, params):
        for param in params:
            self._update_param(param, self.params)        
        self.jaxmppi.update_cost_scales(self.params)
        self.get_logger().info(f"Updated parameters")
        return SetParametersResult(successful=True)
    
    
    def _update_param(self, param, params):
        keys = param.name.split('.')
        for key in keys[:-1]:
            params = params.get(key, {})
        if keys[-1] in params:
            params[keys[-1]] = param.value
            self.get_logger().info(f"Updated parameter {param.name} to {param.value}")
            
    def get_next_cmd(self):
        with self.ctrl_buffer_lock:
            if self.ctrl_buffer:
                return self.ctrl_buffer.popleft()
            else:
                return None
    
    def update_ctrl_buffer(self, ctrl_cmd: jnp.ndarray):
        cmd_list = ctrl_cmd.tolist() 
        with self.ctrl_buffer_lock:
            self.ctrl_buffer.clear()
            self.ctrl_buffer.extend(cmd_list)
        
    def initialize_model(self):              
        self.ctrl_buffer = deque(maxlen=10) 
        self.ctrl_buffer_lock = threading.Lock()
        self.cur_imu = None
        self.cur_odom = None
        self.cur_grid = None
        self.cur_grid_center = None
        self.cur_state = VehicleState()
        self.cur_grid = GridData()  
        self.jax_key = jax.random.PRNGKey(0)       
        self.jaxmppi = JAXMPPI(self.params, self.jax_key)             
        self.goal_state = jnp.array([0.0,0.0])
        self.map_client = self.create_client(GetGridMap, '/get_raw_submap')  
        self.get_logger().info("initialize_model is done")            
        
    def register_sub_pub(self):        
        self.get_logger().info("register sub and pub is done")      
        self.pred_traj_pub = self.create_publisher(Float64MultiArray, 'predicted_paths', 1)  
        self.marker_pub =  self.create_publisher(
            Marker,
            "mppi_goal_marker",
            qos_profile_sensor_data
        )
        
        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            qos_profile_system_default
        )
        
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.params['odom_topic'],
            self.odom_callback,
            qos_profile_sensor_data
        )
        
        self.cmd_pub = self.create_publisher(
            Twist,
            self.params['cmd_topic'],
            qos_profile_action_status_default
        )
        
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the get_grid_map service...')
        self.grid_record_rate = self.params['grid_call_rate']  
        self.map_timer = self.create_timer(1.0/self.grid_record_rate, self.request_subgrid_map)
        self.opt_loop_rate = self.params['opt_loop_rate']      
        self.opt_timer = self.create_timer(1.0 / self.opt_loop_rate, self.opt_loop_callback)        
        self.ctrl_timer_thread = threading.Thread(target=self.init_ctrl_loop)
        self.ctrl_timer_thread.start()


    def init_ctrl_loop(self):
        self.ctrl_loop_rate = self.params['ctrl_loop_rate']     
        while True:
            self.ctrl_loop_callback()
            time.sleep(1.0 / self.ctrl_loop_rate)
        
    def request_subgrid_map(self):
        if self.cur_odom is not None:                
            request = GetGridMap.Request()                    
            request.frame_id = "map"
            request.position_x = self.cur_odom.pose.pose.position.x   
            request.position_y = self.cur_odom.pose.pose.position.y   
            request.length_x =  self.params['Map_config']['map_size'] 
            request.length_y =  self.params['Map_config']['map_size']       
            request.layers = self.params['Map_config']['layers']  
            future = self.map_client.call_async(request)
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
        grid_end_time = time.time()
        time_diff = grid_end_time - grid_start_time
        if abs(time_diff) > 1.0 / self.grid_record_rate:
            self.get_logger().info(f"Grid map processing time: {time_diff} seconds, violating the recording frequency")
        
            
    def publish_twist(self, ctrl):
        if self.params['cmd_pub_enble']:
            twist_msg = Twist()
            twist_msg.linear.x = float(ctrl[0])
            twist_msg.angular.z = float(ctrl[1])
            self.cmd_pub.publish(twist_msg)

    def goal_callback(self, msg):
        self.get_logger().info(f"Received goal position x: {msg.pose.position.x}, y: {msg.pose.position.y}")
        self.goal_state = jnp.array([msg.pose.position.x, msg.pose.position.y])        
        self.jaxmppi.reset()
        
    def odom_callback(self, odom_msg: Odometry):     
        self.cur_odom = odom_msg
        
    def ctrl_loop_callback(self):
        if self.cur_odom is not None:
            cur_odom_position = jnp.array([self.cur_odom.pose.pose.position.x, self.cur_odom.pose.pose.position.y])
            distance = jnp.linalg.norm(self.goal_state[:2] - cur_odom_position)                   
            ctrl = self.get_next_cmd()
            if ctrl is None:
                self.get_logger().info(f"ctrl buffer empty..")
                ctrl = np.array([0.0, 0.0])                                  
            else:
                ctrl = jnp.asarray(ctrl)
            if distance < 0.2:  
                ctrl = np.array([0.0, 0.0])                            
            self.publish_twist(ctrl)
            

    def opt_loop_callback(self):        
        if self.cur_odom is not None:
            start_time = time.time()             
            cur_state_ = VehicleState()
            cur_state_.update_from_odom(self.cur_odom)
            cur_state_jnp = cur_state_.get_mppi_state_jnp()
            (grid_data_jnp, geo_trav_grid_jnp,  grid_is_valid_jnp, grid_center_jnp, grid_map_length_xy_jnp) = self.cur_grid.get_grid_info_jnp()                                                    

            if grid_data_jnp is not None:            
                self.jax_key, subkey = jax.random.split(self.jax_key)
                grid_data_jnp = jnp.nan_to_num(grid_data_jnp, nan=0.0)
                geo_trav_grid_jnp = jnp.nan_to_num(geo_trav_grid_jnp, nan=0.0)
                goal_marker = get_goal_marker(self.goal_state, self.get_clock().now().to_msg())
                self.marker_pub.publish(goal_marker)
                ctrl_cmd, pred_states, total_cost = self.jaxmppi.optimize(self.jax_key, cur_state_jnp, self.goal_state, grid_data_jnp, grid_center_jnp, geo_trav_grid_jnp)
                cost_min_index = jnp.argmin(total_cost)                
                self.update_ctrl_buffer(ctrl_cmd)
                
                if pred_states is not None:
                    best_pred = jnp.expand_dims(pred_states[cost_min_index, :, :2], axis=0)
                    pred_traj_msg = get_jax_array_msg(best_pred)                                 
                    if pred_traj_msg is not None:
                        self.pred_traj_pub.publish(pred_traj_msg)       

            end_time = time.time()
            computation_time = end_time - start_time
            if computation_time > 1.0 / self.opt_loop_rate:                                
                self.get_logger().info(f"Control loop computation time: {computation_time:.6f} seconds..")

def main(args=None):
    rclpy.init(args=args)
    planner_name = sys.argv[1] if len(sys.argv) > 1 else 'default_planner'
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    node = AlignPlannerNode(planner_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()