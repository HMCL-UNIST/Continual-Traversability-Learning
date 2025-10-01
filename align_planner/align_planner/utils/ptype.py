from dataclasses import dataclass, field
import numpy as np
import copy
import torch 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist 
import math
import jax.numpy as jnp
from align_planner.utils.buffers import VehicleStateBuffer

@dataclass
class PythonMsg:
    '''
    Base class for python messages. Intention is that fields cannot be changed accidentally,
    e.g. if you try "state.xk = 10", it will throw an error if no field "xk" exists.
    This helps avoid typos. If you really need to add a field use "object.__setattr__(state,'xk',10)"
    Dataclasses autogenerate a constructor, e.g. a dataclass fields x,y,z can be instantiated
    "pos = Position(x = 1, y = 2, z = 3)" without you having to write the __init__() function, just add the decorator
    Together it is hoped that these provide useful tools and safety measures for storing and moving data around by name,
    rather than by magic indices in an array, e.g. q = [9, 4.5, 8829] vs. q.x = 10, q.y = 9, q.z = 16
    '''

    def __setattr__(self, key, value):
        '''
        Overloads default attribute-setting functionality to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)

    def print(self, depth=0, name=None):
        '''
        default __str__ method is not easy to read, especially for nested classes.
        This is easier to read but much longer

        Will not work with "from_str" method.
        '''
        print_str = ''
        for j in range(depth): print_str += '  '
        if name:
            print_str += name + ' (' + type(self).__name__ + '):\n'
        else:
            print_str += type(self).__name__ + ':\n'
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                print_str += val.print(depth=depth + 1, name=key)
            else:
                for j in range(depth + 1): print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'

        if depth == 0:
            print(print_str)
        else:
            return print_str

    def from_str(self, string_rep):
        '''
        inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)
        '''
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key + '=', val_str_index) + len(key) + 1  # add 1 for the '=' sign
            value_substr = string_rep[val_str_index: string_rep.find(',',
                                                                     val_str_index)]  # (thomasfork) - this should work as long as there are no string entries with commas

            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))

    def copy(self):
        return copy.deepcopy(self)



@dataclass
class Position(PythonMsg):
    x: float = field(default=0)
    y: float = field(default=0)
    z: float = field(default=0)    


@dataclass
class BodyLinearVelocity(PythonMsg):
    vx: float = field(default=0)
    vy: float = field(default=0)
    vz: float = field(default=0)

    
@dataclass
class BodyAngularVelocity(PythonMsg):
    wx: float = field(default=0)
    wy: float = field(default=0)
    wz: float = field(default=0)

@dataclass
class Actuation(PythonMsg):    
    u_vx: float = field(default=0)
    u_wz: float = field(default=0)    
    
@dataclass
class Traction(PythonMsg): 
    linear: float = field(default=0.0)
    angular: float = field(default=0.0)

@dataclass
class Orientation(PythonMsg):
    phi: float = field(default=0)
    theta: float = field(default=0)
    psi: float = field(default=0)
    qx: float = field(default=0)
    qy: float = field(default=0)
    qz: float = field(default=0)
    qw: float = field(default=1)
    
    def __post_init__(self):
        self.update_from_quaternion()
    
    @staticmethod
    def q_to_rot_mat(x,y,z,w):
        """
        Convert a quaternion into a rotation matrix.
        """        
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
        
    def get_inv_rot_mat(self):
        return self.q_to_rot_mat(self.qx, self.qy, self.qz, self.qw).T
        
    @staticmethod
    def wrap_to_pi(angle):
        """
        Wraps an angle in radians to the range [-pi, pi].

        Args:
            angle (float): Angle in radians.

        Returns:
            float: Wrapped angle in radians.
        """        
        while angle <= -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle
    
    def update_from_quaternion(self):
        qx = self.qx 
        qy = self.qy 
        qz = self.qz 
        qw = self.qw 

        
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        self.phi = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            self.theta = math.copysign(math.pi / 2, sinp)
        else:
            self.theta = math.asin(sinp)

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        self.psi = math.atan2(siny_cosp, cosy_cosp)


@dataclass
class VehicleState(PythonMsg):
    '''
    Complete vehicle state (local, global, and input)
    '''
    t: float = field(default=0)  # time in seconds
    p: Position = field(default=Position)  # global position
    r: Orientation = field(default=Orientation)  # global orientation (phi, theta, psi)    
    v: BodyLinearVelocity = field(default=BodyLinearVelocity)  # body linear velocity
    w: BodyAngularVelocity = field(default=BodyAngularVelocity)  # body angular velocity            
    u: Actuation = field(default=Actuation)  # actuation (vx, wz)
    traction: Traction = field(default=Traction)
    
    def update_from_cmd(self, twist_msg: Twist):
        self.u = Actuation(u_vx=twist_msg.linear.x, u_wz=twist_msg.angular.z)   
        
        
    def update_from_odom(self, odom_msg: Odometry, is_body_frame=False):
        self.t = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
        self.p = Position(x=odom_msg.pose.pose.position.x, y=odom_msg.pose.pose.position.y, z=odom_msg.pose.pose.position.z)                
        self.r = Orientation(phi=0, theta=0, psi=0,qx=odom_msg.pose.pose.orientation.x, qy=odom_msg.pose.pose.orientation.y, qz=odom_msg.pose.pose.orientation.z, qw=odom_msg.pose.pose.orientation.w)        
        if is_body_frame is False:
            rot_mat = Orientation.q_to_rot_mat(odom_msg.pose.pose.orientation.x,odom_msg.pose.pose.orientation.y,odom_msg.pose.pose.orientation.z,odom_msg.pose.pose.orientation.w)
            inv_rot_mat_ = np.linalg.inv(rot_mat)        
            global_vel = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y,odom_msg.twist.twist.linear.z]
            local_vel = inv_rot_mat_.dot(global_vel)
        else:
            local_vel = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y,odom_msg.twist.twist.linear.z]
        local_vel[0] = max(0.0, local_vel[0])
        self.v = BodyLinearVelocity(vx=local_vel[0], vy=local_vel[1], vz=local_vel[2])
        self.w = BodyAngularVelocity(wx=odom_msg.twist.twist.angular.x, wy=odom_msg.twist.twist.angular.y, wz=odom_msg.twist.twist.angular.z)        
                
    def update_from_odom_and_cmd(self, odom_msg: Odometry, twist_msg: Twist, is_body_frame=False):
        self.update_from_odom(odom_msg=odom_msg, is_body_frame=is_body_frame)   
        self.update_from_cmd(twist_msg=twist_msg)           
        
    def update_traction(self,linear, angular):
        self.traction = Traction(linear=linear, angular=angular)
            
    def get_mppi_state_tensor(self, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor([self.p.x, self.p.y, self.r.psi, self.v.vx, self.w.wz], dtype=torch.float32).to(device)        
    
    def get_mppi_state_jnp(self):        
        return jnp.array([self.p.x, self.p.y, self.r.psi, self.v.vx, self.w.wz])
        
    def as_tensor(self, device=None):
        '''
        Return the model output as a tensor.
        '''
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor([self.v.vx, self.w.wz, self.u.u_vx, self.u.u_wz], dtype=torch.float32).to(device)

    
    def forward(self,dt):        
        next_state = self.copy()
        next_state.p.x = self.p.x + dt * (self.u.u_vx * math.cos(self.r.psi))
        next_state.p.y =  self.p.y + dt * (self.u.u_vx * math.sin(self.r.psi))
        next_state.r.psi = self.r.psi + dt * self.u.u_wz                      
        next_state.r.psi = Orientation.wrap_to_pi(next_state.r.psi)        
        next_state.u.u_vx = self.u.u_vx
        next_state.u.u_wz = self.u.u_wz
        return next_state
        


@dataclass
class VehicleStateDeviation(PythonMsg):
    '''
    This class records the deviation in vehicle states: position, linear and angular velocities.
    '''
    dt: float = field(default=0.1) # sampling time
    prev_t: float = field(default=0)  # previous time step
    t: float = field(default=0)  # current time step
    del_t: float = field(default=0)  # time difference (t - prev_t)
    dp: Position = field(default=Position)  # deviation in position
    dv: BodyLinearVelocity = field(default=BodyLinearVelocity)  # deviation in body linear velocity
    dw: BodyAngularVelocity = field(default=BodyAngularVelocity)  # deviation in body angular velocity
    dr: Orientation = field(default=Orientation)  # deviation in orientation
    traction: Traction = field(default=Traction)
    
    def __post_init__(self):
        self.del_t = self.t - self.prev_t

    def update_deviation(self, cur_state: VehicleState, next_state: VehicleState):
        '''
        Update the deviation based on the actual state and reference state.
        '''
        del_x = (next_state.p.x - cur_state.p.x)
        del_y = (next_state.p.y - cur_state.p.y)
        delta_t = next_state.t - cur_state.t                 

        self.prev_t = cur_state.t
        self.t = next_state.t
        self.del_t = self.t - self.prev_t

        self.traction.linear = next_state.traction.linear
        self.traction.angular = next_state.traction.angular
     
        model_based_update = cur_state.forward(self.dt)        
        inv_rot_mat = cur_state.r.get_inv_rot_mat()
        
        position_diff = np.array([next_state.p.x - model_based_update.p.x,
                                    next_state.p.y - model_based_update.p.y,
                                    next_state.p.z - model_based_update.p.z
                                ])
        local_position_diff = inv_rot_mat.dot(position_diff)
        
        self.dp = Position(
            x=local_position_diff[0],
            y=local_position_diff[1],
            z=local_position_diff[2],
        )
        self.dr.psi = Orientation.wrap_to_pi(next_state.r.psi - model_based_update.r.psi)
        
        self.dv = BodyLinearVelocity(
            vx=next_state.v.vx - cur_state.v.vx,
            vy=next_state.v.vy - cur_state.v.vy,
            vz=next_state.v.vz - cur_state.v.vz
        )
        
        self.dw = BodyAngularVelocity(
            wx=next_state.w.wx - cur_state.w.wx,
            wy=next_state.w.wy - cur_state.w.wy,
            wz=next_state.w.wz - cur_state.w.wz
        )
        
        
        
        
    def as_tensor(self, device=None):
        '''
        Return the model mismatch  as output
        '''
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor([self.traction.linear,self.traction.angular], dtype=torch.float32).to(device)


@dataclass
class GridData(PythonMsg):
    '''
    Class to store grid map data.
    '''
    t: float = field(default=None)  # time in seconds
    center_pose: Position = field(default=Position)  # center position of the grid         
    resolution: float = 0.25
    map_length_x: float = 100
    map_length_y: float = 100
    layer_names: list = field(default=None)  # names of the layers
    data: torch.tensor = field(default=None) 
    geo_trav_data: torch.tensor = field(default=None) 
    is_valid: torch.tensor = field(default=None) 

    def update_from_grid_map(self, grid_map_msg):
        self.t = grid_map_msg.header.stamp.sec + grid_map_msg.header.stamp.nanosec * 1e-9        
        self.center_pose = Position(x=grid_map_msg.info.pose.position.x, y=grid_map_msg.info.pose.position.y, z=grid_map_msg.info.pose.position.z)
        self.resolution = grid_map_msg.info.resolution
        self.map_length_x = grid_map_msg.info.length_x
        self.map_length_y = grid_map_msg.info.length_y
        self.layer_names = grid_map_msg.layers        
        grid_count_x = int(grid_map_msg.info.length_x / grid_map_msg.info.resolution)
        grid_count_y = int(grid_map_msg.info.length_y / grid_map_msg.info.resolution)        
        map_data_list = []
        for idx, layer_data  in enumerate(grid_map_msg.data):            
            if self.layer_names[idx] == 'is_valid':
                self.is_valid = torch.tensor(layer_data.data, dtype=torch.float32).reshape(grid_count_x, grid_count_y)
                self.is_valid = torch.flip(self.is_valid, dims=[0,1])
                continue
            if self.layer_names[idx] == 'geo_trav':
                self.geo_trav_data = torch.tensor(layer_data.data, dtype=torch.float32).reshape(grid_count_x, grid_count_y)
                self.geo_trav_data = torch.flip(self.geo_trav_data, dims=[0,1])
                

            layer_tensor = torch.tensor(layer_data.data, dtype=torch.float32).reshape(grid_count_x, grid_count_y)
            layer_tensor = torch.flip(layer_tensor, dims=[0,1])
            map_data_list.append(layer_tensor)
        self.data = torch.stack(map_data_list, dim=0)
        
    def get_features_from_grid(self, pose_x, pose_y):        
        row_idx = ((pose_y - self.center_pose.y + self.map_length_y / 2) / self.resolution).to(dtype=torch.long, device=self.data.device)
        row_idx = torch.clip(row_idx, 0, self.data.shape[1]-1)        
        col_idx = ((pose_x - self.center_pose.x + self.map_length_x / 2) / self.resolution).to(dtype=torch.long, device=self.data.device)
        col_idx = torch.clip(col_idx, 0, self.data.shape[2]-1)        
        return self.data[:, row_idx, col_idx].T, self.is_valid[row_idx,col_idx].T
    
    def get_features_on_pose(self,pose_x, pose_y):    
        row_idx = int((pose_y - self.center_pose.y + self.map_length_y / 2) / self.resolution)
        col_idx = int((pose_x - self.center_pose.x + self.map_length_x / 2) / self.resolution)
        if 0 <= row_idx < self.data.shape[1] and 0 <= col_idx < self.data.shape[2]:                                        
            return self.data[:, row_idx, col_idx].T, self.is_valid[row_idx,col_idx].T
        else:
            return None, False    
      
        
    def get_grid_info_tensor(self, device=None):
        '''
        Return the model output as a tensor.
        '''
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            
        if self.data is None:
            return (None, None, None, None)
        grid = self.data.to(device, dtype=torch.float32)
        is_valid = self.is_valid.to(device, dtype=torch.float32)        
        center = torch.tensor([self.center_pose.x, self.center_pose.y], dtype=torch.float32).to(device)
        map_length = torch.tensor(self.map_length_x, dtype=torch.float32).to(device)
        return (grid,is_valid, center, map_length)
    
    def get_grid_info_jnp(self):
        '''
        Return the model output as a jax jnp.
        '''        
        if self.data is None:            
            return (None,None,  None, None, None)        
        numpy_grid = self.data.detach().cpu().numpy()
        grid = jnp.array(numpy_grid)        
        numpy_is_valid = self.is_valid.detach().cpu().numpy()
        is_valid = jnp.array(numpy_is_valid)
        center = jnp.array([self.center_pose.x, self.center_pose.y])
        map_length = jnp.array([self.map_length_x])
        
        numpy_geo_trav_grid = self.geo_trav_data.detach().cpu().numpy()
        geo_trav_grid = jnp.array(numpy_geo_trav_grid)        
        return (grid, geo_trav_grid, is_valid, center, map_length)
    
    
    def compute_wheel_positions(self,base_xypsi, car_length = 0.3 , car_width = 0.4):
        half_length = car_length / 2
        half_width = car_width / 2
        # Define relative positions of the wheels in the car's local frame
        local_wheel_offsets = torch.tensor([
            [half_length,  half_width],  # Front Left
            [half_length, -half_width],  # Front Right
            [-half_length,  half_width],  # Rear Left
            [-half_length, -half_width],  # Rear Right
        ], dtype=base_xypsi.dtype, device=base_xypsi.device)  # Shape: (4, 2)

        # Extract base positions and orientation
        x_base = base_xypsi[:, 0]  # Shape: (batch_size,)
        y_base = base_xypsi[:, 1]  # Shape: (batch_size,)
        psi_base = base_xypsi[:, 2]  # Shape: (batch_size,)

        # Rotation matrix for each batch (based on psi_base)
        cos_psi = torch.cos(psi_base)
        sin_psi = torch.sin(psi_base)
        rotation_matrix = torch.stack([
            torch.stack([cos_psi, -sin_psi], dim=1),
            torch.stack([sin_psi,  cos_psi], dim=1)
        ], dim=1)  # Shape: (batch_size, 2, 2)

        # Rotate local wheel offsets into the global frame
        global_offsets = torch.matmul(rotation_matrix, local_wheel_offsets.T).permute(0, 2, 1)  # Shape: (batch_size, 4, 2)

        # Add global offsets to the base positions
        base_positions = torch.stack([x_base, y_base], dim=1).unsqueeze(1)  # Shape: (batch_size, 1, 2)
        wheel_positions = base_positions + global_offsets  # Shape: (batch_size, 4, 2)

        return wheel_positions
    

    
    def features_on_states(self,state: VehicleState, next_state:VehicleState, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        xypsi = torch.tensor([[state.p.x, state.p.y,state.r.psi]], dtype=torch.float32).to(device)
        wheels_xy = self.compute_wheel_positions(xypsi).reshape(-1,2)                                
        sampled_feature_lr, is_valid_lr = self.get_features_on_pose(wheels_xy[0,0], wheels_xy[0,1])                  
        sampled_feature_rr, is_valid_rr = self.get_features_on_pose(wheels_xy[1,0], wheels_xy[1,1])                  
        sampled_feature_lf, is_valid_lf = self.get_features_on_pose(wheels_xy[2,0], wheels_xy[2,1])                  
        sampled_feature_rf, is_valid_rf = self.get_features_on_pose(wheels_xy[3,0], wheels_xy[3,1])                  
        if is_valid_rf == 1 and is_valid_lf == 1 and is_valid_rr == 1 and is_valid_lr == 1:        
            sampled_feature = torch.stack([sampled_feature_lr, sampled_feature_rr, sampled_feature_lf, sampled_feature_rf], dim=0)                  
            grid_feat, is_grid_valid = self.get_features_from_grid(wheels_xy[:,0], wheels_xy[:,1])        
            grid_feat = grid_feat.reshape(-1)
            assert torch.allclose(sampled_feature.view(-1), grid_feat, atol=1e-5), "Error in sampling features"
        else:
            return None, None, None
        
            
        _, is_valid_next = self.get_features_on_pose(next_state.p.x, next_state.p.y)        
        next_xypsi = torch.tensor([[next_state.p.x, next_state.p.y,next_state.r.psi]], dtype=torch.float32).to(device)
        next_wheels_xy = self.compute_wheel_positions(next_xypsi).reshape(-1,2)                                
        next_sampled_feature_lr, next_is_valid_lr = self.get_features_on_pose(next_wheels_xy[0,0], next_wheels_xy[0,1])                  
        next_sampled_feature_rr, next_is_valid_rr = self.get_features_on_pose(next_wheels_xy[1,0], next_wheels_xy[1,1])                  
        next_sampled_feature_lf, next_is_valid_lf = self.get_features_on_pose(next_wheels_xy[2,0], next_wheels_xy[2,1])                  
        next_sampled_feature_rf, next_is_valid_rf = self.get_features_on_pose(next_wheels_xy[3,0], next_wheels_xy[3,1])                  
        
        
        if next_is_valid_rf == 1 and next_is_valid_lf == 1 and next_is_valid_rr == 1 and next_is_valid_lr == 1:    
            deviation = VehicleStateDeviation()
            deviation.update_deviation(state, next_state)
            return grid_feat,  state.as_tensor(), deviation.as_tensor()            
        else:
            return None, None, None 
   
    
    def features_on_trajectory(self, vehicle_states_buffer: VehicleStateBuffer, device=None):
        '''
        Compute the index of positions for each pose of vehicle states and check if the pose is inside the map boundary.
        Use grid sample library to get the part of torch tensor of self.data for each vehicle state.
        Return this as output tensor.
        '''
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        map_features = []
        vehicle_states_tensors = []
        state_deviations = []
        
        vehicle_states = vehicle_states_buffer.get_data()
        for i in range(len(vehicle_states) - 1):            
            state = vehicle_states[i]
            sampled_feature, is_valid = self.get_features_on_pose(state.p.x, state.p.y)            
            next_state = vehicle_states[i + 1]
            _, is_valid_next = self.get_features_on_pose(next_state.p.x, next_state.p.y)
            
            if is_valid == 1 and is_valid_next == 1:                
                map_features.append(sampled_feature)
                # Compute the state deviation between the current state and the next vehicle state
                deviation = VehicleStateDeviation()
                deviation.update_deviation(state, next_state)
                vehicle_states_tensors.append(state.as_tensor())
                state_deviations.append(deviation.as_tensor())

        if len(map_features) > 0:
            map_features = torch.stack(map_features).to(device)
            vehicle_states_tensors = torch.stack(vehicle_states_tensors).to(device)
            state_deviations = torch.stack(state_deviations).to(device)
            return map_features, vehicle_states_tensors, state_deviations
        else:
            return None, None, None
        


