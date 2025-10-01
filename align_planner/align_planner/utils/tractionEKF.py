import numpy as np
from align_planner.utils.ptype import VehicleState
from nav_msgs.msg import Odometry

class EKF:
    def __init__(self):
        self.dt = 1/10  # time step
        # State vector [x, y, psi, vx, wz, linear_traction, angular_traction]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.control = np.array([0.0, 0.0])
        # State covariance matrix        
        self.P = np.diag([0.01, 0.01, 0.01,0.01,0.01, 1.0, 1.0]) 
        # Process noise covariance
        self.Q = np.diag([0.05, 0.05, 0.05,0.05,0.05, 0.3, 0.3])
        # Measurement noise covariance
        self.R = np.diag([0.05, 0.05, 0.05,0.05,0.05, 10.0, 10.0])
        self.cur_state = VehicleState()

    def reset(self, cur_state: VehicleState):
        self.state[-2] = 1.0
        self.state[-1] = 1.0
        self.set_state(cur_state)            
        self.set_ctrl(cur_state)        
        self.P = np.diag([0.1, 0.1, 0.1,0.1, 0.1, 1.0, 1.0]) 
        
    def wrap_to_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def predict(self, control = None):
        self.state[2] = self.wrap_to_pi(self.state[2])
        if control is None:
            vx_cmd = self.control[0]
            wz_cmd = self.control[1]
        else:
            vx_cmd, wz_cmd = control        
        x, y, psi, vx, wz, linear_slip, angular_slip = self.state
                        
        self.state[0] = x + self.dt * self.control[0] * linear_slip * np.cos(psi)
        self.state[1] = y + self.dt * self.control[0] * linear_slip * np.sin(psi)
        self.state[2] = psi + self.dt * angular_slip * self.control[1]
        self.state[3] =  self.control[0] * linear_slip 
        self.state[4] = self.control[1] * angular_slip
        
        F = np.zeros((7, 7))

        F[0, 0] = 1
        F[0, 2] = -self.dt * vx_cmd * linear_slip * np.sin(psi)
        F[0, 5] = self.dt * vx_cmd * np.cos(psi)

        F[1, 1] = 1
        F[1, 2] = self.dt * vx_cmd * linear_slip * np.cos(psi)
        F[1, 5] = self.dt * vx_cmd * np.sin(psi)

        F[2, 2] = 1
        F[2, 6] = self.dt * wz_cmd

        F[3, 5] = vx_cmd

        F[4, 6] = wz_cmd

        F[5, 5] = 1
        F[6, 6] = 1
        
        
        self.P = F @ self.P @ F.T + self.Q
  
    def unwrap(self, angle, prev_angle):
        return angle + (prev_angle - angle + np.pi) // (2 * np.pi) * 2 * np.pi


    def set_state(self,cur_state:VehicleState):
        x = cur_state.p.x
        y = cur_state.p.y
        psi = cur_state.r.psi        
        vx = cur_state.v.vx        
        wz = cur_state.w.wz        
        self.state = np.array([x, y, psi,vx,wz, self.state[-2], self.state[-1]])
    
    def set_ctrl(self,cur_state:VehicleState):
        self.cur_state = cur_state  
        self.control = np.array([float(cur_state.u.u_vx), float(cur_state.u.u_wz)])
    
    def get_odom(self,header):
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.orientation.x = self.state[2]        
        odom_msg.pose.pose.orientation.y = self.state[4]        
        odom_msg.pose.pose.orientation.z = self.control[1]         ## wz_cmd
        
        odom_msg.twist.twist.linear.x = self.state[-2]  ## linear traction 
        odom_msg.twist.twist.linear.y = self.state[3]  ## vx hat         
        odom_msg.twist.twist.linear.z = self.control[0]  ## vx_cmd         
        
        odom_msg.twist.twist.angular.z = self.state[-1] ## angular traction  
        return odom_msg
    
    def unwrap(self, angle, prev_angle):
        return angle + (prev_angle - angle + np.pi) // (2 * np.pi) * 2 * np.pi

    def angular_distance(self, angle, prev_angle):
        return self.wrap_to_pi(angle - prev_angle)
    
    def update_from_vehicleState(self,cur_state:VehicleState):
        measured_psi = cur_state.r.psi
   
        measured_psi = self.wrap_to_pi(measured_psi)
        ref_angle = self.wrap_to_pi(self.state[2])        
        distance = self.angular_distance(measured_psi, ref_angle)      
        corrected_angle = ref_angle + distance 
        measurement = [cur_state.p.x, cur_state.p.y, corrected_angle,cur_state.v.vx,cur_state.w.wz]
        
        self.update(measurement)
   
        
    def update(self, measurement):
        x, y, psi,vx,wz = self.state[:5]

        z_pred = np.array([x, y, psi,vx,wz])
        z_pred = self.state
        
        H = np.eye(7)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        virtual_measurment = np.array([self.state[-2], self.state[-1]])        
        if abs(self.control[0]) < 0.05:
            virtual_measurment[0] = 1.0        
        if abs(self.control[1]) < 0.05:            
            virtual_measurment[1] = 1.0      
        
        z = np.hstack([measurement,virtual_measurment])
        err_vec = z - z_pred
        residual_update = K @ (err_vec)    
        self.state += residual_update
        self.P = (np.eye(7) - K @ H) @ self.P

        

    def get_state(self):
        return self.state
