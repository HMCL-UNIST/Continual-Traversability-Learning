import jax
import jax.numpy as jnp
from typing import Sequence
import os
import torch
import time
from typing import Tuple
from functools import partial
from flax import linen as nn
from ament_index_python.packages import get_package_share_directory
from flax.core import freeze, unfreeze

 
def convert_pytorch_weights_ensemble(model_path, model_dir, jax_model_cls, input_shape, key, num_models=5):

    if '.pth' not in model_path:
        model_path = f'{model_path}.pth'
    model_filename = os.path.join(model_dir, model_path)

    # Load PyTorch weights
    torch_weights = torch.load(model_filename, map_location="cpu")
    print(f"Model loaded from {model_filename}")

    # Split the state_dict into per-model dictionaries
    model_param_dicts = [{} for _ in range(num_models)]
    for key_name, value in torch_weights['ensemble'].items():
        parts = key_name.split('.', 1)
        model_idx = int(parts[0])
        param_subkey = parts[1] if len(parts) > 1 else ""
        model_param_dicts[model_idx][param_subkey] = value


    # Convert each model
    jax_param_list = []
    for i in range(num_models):
        subkey = jax.random.fold_in(key, i)
        jax_model = jax_model_cls()
        variables = jax_model.init(subkey, jnp.ones(input_shape))
        flax_params = unfreeze(variables["params"])

        torch_model_params = model_param_dicts[i]
        torch_layer_idx = 0
        for name, layer in flax_params.items():
            for param_name, _ in layer.items():
                torch_tensor = list(torch_model_params.values())[torch_layer_idx]
                array = torch_tensor.detach().cpu().numpy()

                if "kernel" in param_name:
                    array = array.T

                flax_params[name][param_name] = jnp.asarray(array)
                torch_layer_idx += 1

        
        jax_param_list.append(freeze(flax_params))

    return jax_param_list

class TravNNJAX(nn.Module):
    input_size: int
    hidden_units: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.hidden_units):
            x = nn.Dense(h)(x)
            x = nn.leaky_relu(x)

        mean_z = nn.Dense(self.output_size)(x)
        var_z = nn.Dense(self.output_size)(x)

        return mean_z, var_z


 
@jax.jit
def normalize(tensor):
        max_val = jnp.max(tensor)
        min_val = jnp.min(tensor)
        return (tensor - min_val) / (max_val - min_val + 1e-11)

class JAXMPPI:
    def __init__(self, params, key: jax.Array):
        self.params = params
        self.data_dir = os.path.join(get_package_share_directory('align_planner'), params["Travnn_args"]["normalizing_constant_dir"])                        
        self.model_dir = os.path.join(get_package_share_directory('align_planner'), params["Travnn_args"]["model_dir"])                        

        self.key = key # jax.random.PRNGKey(0)        
        
        self.dt = params['Dynamics_config']['dt'] 
        self.K =  params["MPPI_config"]["BATCHSIZE"] 
        self.T =  params["MPPI_config"]["TIMESTEPS"]        
        
        ''' 
        Paramaeters for input sampling 
        '''
        self.nx = params['Sampling_config']["state_dim"]
        self.nu = params['Sampling_config']["control_dim"]
        self.temperature = params['Sampling_config']["temperature"]
        self.scaled_dt = params['Sampling_config']["scaled_dt"]
        self.CTRL_NOISE = jnp.diag(jnp.array([params['Sampling_config']["noise_0"], params['Sampling_config']["noise_1"]]))
        self.CTRL_NOISE_INV = jnp.linalg.inv(self.CTRL_NOISE)
        self.CTRL_NOISE_MU = jnp.zeros((self.nu,))
        self.min_vel = params['Sampling_config']['min_vel']
        self.max_vel = params['Sampling_config']['max_vel']                  
        self.max_ax = params['Sampling_config']["max_ax"]
        self.min_ax = params['Sampling_config']["min_ax"]
        self.max_wz = params['Sampling_config']["max_yaw_rate"]
        self.min_wz = -self.max_wz
        self.max_alpha_x = params['Sampling_config']["max_alpha_x"]
        self.min_alpha_x = params['Sampling_config']["min_alpha_x"]

        '''
        Parameters for Dynamics rollout 
        '''
        key_vx, key_wz, key_model = jax.random.split(self.key, 3)
        vx_cmds = jnp.zeros([self.K, self.T])
        wz_cmds = jnp.zeros([self.K, self.T])        
        self.us = jnp.stack([vx_cmds, wz_cmds], axis=-1)        
        self.u = self.us[0,:,:]
        self.states = jnp.zeros([self.T, self.nx])                
        self.car_length = params['Dynamics_config']['car_length']
        self.car_width = params['Dynamics_config']['car_width']
        self.grid_tensor_size = int(params['Map_config']['map_size']/ params['Map_config']['map_res']+1)
        self.grid_dim = params['Travnn_args']['grid_feat_size']
        
        ########## DEBUG
        self.grid_data = jnp.zeros([self.grid_dim, self.grid_tensor_size, self.grid_tensor_size])
        self.geo_trav_grid_data = jnp.zeros([self.grid_tensor_size, self.grid_tensor_size])
        self.grid_center = jnp.zeros(2)
        ########## DEBUG END
        self.grid_map_length = params['Map_config']['map_size']
        self.grid_res = params['Map_config']['map_res']
        self.grid_map_size_pixel = int(self.grid_map_length / self.grid_res)
        
        '''
        Prameters for Cost computation
        '''
        self.mean_slip_cost_scale = params['Cost_config']["mean_slip_cost_scale"]
        self.uncert_costs_scale = params['Cost_config']["uncert_costs_scale"]
        self.terminal_dist_costs_scale = params['Cost_config']["terminal_dist_costs_scale"]
        self.geo_trav_cost_scale = params['Cost_config']["geo_trav_cost_scale"]
        self.ctrl_cost_scale = params['Cost_config']["ctrl_cost_scale"] 
        
        
        '''
        Prediction ready
        '''        
        self.pred_out_dim =  params["Travnn_args"]["trav_output_size"] 
        jax_param_list = convert_pytorch_weights_ensemble(
            model_path=params["Ensemble_args"]["ensemble_weight"],
            model_dir=self.model_dir,
            jax_model_cls=lambda: TravNNJAX(
                input_size=params["Travnn_args"]["input_size"],
                hidden_units=params["Travnn_args"]["trav_hidden_units"],
                output_size=self.pred_out_dim
            ),
            input_shape=(1, params["Travnn_args"]["input_size"]),
            key=jax.random.PRNGKey(0),
            num_models=params["Ensemble_args"]["num_models"]
        )
        params_stacked = jax.tree_util.tree_map(lambda *p: jnp.stack(p), *jax_param_list)

        @partial(jax.vmap, in_axes=(0, None))
        def model_apply_ensemble(params, x):
            ## TODO: replace Hardcoded network params.
            model = TravNNJAX(
                input_size=21,
                hidden_units=[16,32,32,16],
                output_size=2
            )
            return model.apply({"params": params}, x)
        
        self.model_apply = jax.jit(partial(model_apply_ensemble, params_stacked))        
        self.load_normalizing_constants()

        '''
        Compile sampling function 
        '''
        def sample_single(key: jax.Array, U: jnp.ndarray, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            noise = (jax.random.normal(key, (self.T, self.nu)) @ self.CTRL_NOISE) + self.CTRL_NOISE_MU
            perturbed = U + noise
            perturbed = perturbed.at[:, 0].set(jnp.clip(perturbed[:, 0], self.min_ax, self.max_ax))
            perturbed = perturbed.at[:, 1].set(jnp.clip(perturbed[:, 1], self.min_alpha_x, self.max_alpha_x))
            cumsum = jnp.cumsum(perturbed, axis=0) * self.scaled_dt
            controls = jnp.clip(state[-self.nu:] + cumsum, -1.0, 1.0)
            controls = controls.at[:,0].set(jnp.clip(controls[:,0], self.min_vel, self.max_vel))
            controls = controls.at[:,1].set(jnp.clip(controls[:,1], self.min_wz, self.max_wz))
            diffs = jnp.diff(controls - state[-self.nu:], axis=0)
            perturbed = perturbed.at[1:,:].set(diffs / self.scaled_dt)
            new_noise = perturbed - U
            action_cost = self.temperature * (new_noise @ self.CTRL_NOISE_INV)
            perturb_cost = jnp.sum(U * action_cost)    
            return controls, perturb_cost, new_noise, key
        

        def update_control_single(cost_total: jnp.ndarray, U: jnp.ndarray, state: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            weights = jnp.exp((-1.0 / self.temperature) * (cost_total))
            omega = weights / jnp.sum(weights)
            delta_U = jnp.sum(omega[:, None, None] * noise, axis=0)
            updated_U = U + delta_U
            cumsum = jnp.cumsum(U, axis=0) * self.scaled_dt
            controls = jnp.clip(state[-self.nu:] + cumsum, -1.0, 1.0)
            controls = controls.at[:,0].set(jnp.clip(controls[:,0], self.min_vel, self.max_vel))
            controls = controls.at[:,1].set(jnp.clip(controls[:,1], self.min_wz, self.max_wz))

            updated_U = updated_U.at[:,0].set(jnp.clip(updated_U[:,0], self.min_vel, self.max_vel))
            updated_U = updated_U.at[:,1].set(jnp.clip(updated_U[:,1], self.min_wz, self.max_wz))

            return controls, updated_U

        self.sample_controls = jax.jit(jax.vmap(sample_single, in_axes=(0, None, None)))
        self.update_control = jax.jit(update_control_single)

        
        def rollout_fn(states, us, sampled_epsilon,grid_data, grid_center):
            return jax.vmap(self.rollout_us, in_axes=(0, 0, 0, None, None,None))(states, us,sampled_epsilon, grid_data,grid_center, self.model_apply)    
        
        self.rollout_fn = jax.jit(rollout_fn)
      
        def geo_trav_compute_wheel_positions(state):
            x, y, yaw = state[:,0], state[:,1], state[:,2]
            dx = self.car_length / 2.0 
            dy = self.car_width / 2.0  
            # Compute the positions of the four corners
            corners = jnp.array([
                [x + dx * jnp.cos(yaw) - dy * jnp.sin(yaw), y + dx * jnp.sin(yaw) + dy * jnp.cos(yaw)],  # Front-left
                [x + dx * jnp.cos(yaw) + dy * jnp.sin(yaw), y + dx * jnp.sin(yaw) - dy * jnp.cos(yaw)],  # Front-right
                [x - dx * jnp.cos(yaw) - dy * jnp.sin(yaw), y - dx * jnp.sin(yaw) + dy * jnp.cos(yaw)],  # Rear-left
                [x - dx * jnp.cos(yaw) + dy * jnp.sin(yaw), y - dx * jnp.sin(yaw) - dy * jnp.cos(yaw)]   # Rear-right
            ])
            return corners
        
        def get_geo_trav_cost_from_grid(pose_x, pose_y, grid_data, grid_center):
            row_idx = ((pose_y - grid_center[1] + self.grid_map_length / 2) / self.grid_res).astype(jnp.int32)
            row_idx = jnp.clip(row_idx, 0, self.grid_tensor_size - 1)
            col_idx = ((pose_x - grid_center[0] + self.grid_map_length / 2) / self.grid_res).astype(jnp.int32)
            col_idx = jnp.clip(col_idx, 0, self.grid_tensor_size - 1)
            features = grid_data[row_idx.flatten(), col_idx.flatten()].T            
            return features
    
        
        def geo_trav_cost_single(states, slips, geo_grid, grid_center):
                     
            corners = geo_trav_compute_wheel_positions(states[:,:3])        
            geo_costs = get_geo_trav_cost_from_grid(corners[:,0,:],corners[:,1,:],geo_grid, grid_center)
             
            return jnp.sum(geo_costs)
                
        def slip_uncertainty_cost(slips):
            var_clip_min = 5e-2
            var_clip_max = 2e1

            pred_mean = slips[:, :, 0]
            deviation_to_one = jnp.abs(pred_mean - 1.0)
            mean_slip_cost = jnp.sum(deviation_to_one)

            pred_var = slips[:, :, 1]
            pred_var = jnp.clip(pred_var, var_clip_min, var_clip_max)
            slip_var_cost = jnp.sum(pred_var)

            return mean_slip_cost, slip_var_cost
        
        def goal_cost(goal,state):
            wp_vec = goal - state[-1,:2]
            terminal_distance_cost = jnp.linalg.norm(wp_vec)
            return terminal_distance_cost
            
        
        self.goal_cost = jax.jit(jax.vmap(goal_cost,in_axes=(None,0)))
        self.slip_uncertainty_cost = jax.jit(jax.vmap(slip_uncertainty_cost, in_axes=(0)))
        self.geo_trav_cost = jax.jit(jax.vmap(geo_trav_cost_single,in_axes=(0, 0, None, None)))

        print("Compile Done")
        

    
    def load_normalizing_constants(self):
            try:
                constants = torch.load(os.path.join(self.data_dir, 'normalize_constants.pkl'))                            
                self.constants_feat_mean  = jnp.array(constants['feat_mean'].cpu().numpy())
                self.constants_feat_std   = jnp.array(constants['feat_std'].cpu().numpy())
                self.constants_state_mean = jnp.array(constants['state_mean'].cpu().numpy())
                self.constants_state_std  = jnp.array(constants['state_std'].cpu().numpy())
                self.constants_out_mean   = jnp.array(constants['deviation_mean'].cpu().numpy())
                self.constants_out_std    = jnp.array(constants['deviation_std'].cpu().numpy())
            except FileNotFoundError:
                raise AssertionError("Normalizing constants missing")
            print("Normalizing constants loaded.")

        
    def rollout_us(self,state, us, sampled_epsilon, grid_data, grid_center, model_apply):
        
        def compute_wheel_positions(state):
            x, y, yaw = state[0], state[1], state[2]
            dx = self.car_length / 2.0 
            dy = self.car_width / 2.0 
            corners = jnp.array([
                [x + dx * jnp.cos(yaw) - dy * jnp.sin(yaw), y + dx * jnp.sin(yaw) + dy * jnp.cos(yaw)],  # Front-left
                [x + dx * jnp.cos(yaw) + dy * jnp.sin(yaw), y + dx * jnp.sin(yaw) - dy * jnp.cos(yaw)],  # Front-right
                [x - dx * jnp.cos(yaw) - dy * jnp.sin(yaw), y - dx * jnp.sin(yaw) + dy * jnp.cos(yaw)],  # Rear-left
                [x - dx * jnp.cos(yaw) + dy * jnp.sin(yaw), y - dx * jnp.sin(yaw) - dy * jnp.cos(yaw)]   # Rear-right
            ])
            return corners
        
        def get_features_from_grid(pose_x, pose_y, grid_data, grid_center):
            row_idx = ((pose_y - grid_center[1] + self.grid_map_length / 2) / self.grid_res).astype(jnp.int32)
            row_idx = jnp.clip(row_idx, 0, self.grid_tensor_size - 1)
            col_idx = ((pose_x - grid_center[0] + self.grid_map_length / 2) / self.grid_res).astype(jnp.int32)
            col_idx = jnp.clip(col_idx, 0, self.grid_tensor_size - 1)
            features = grid_data[:, row_idx, col_idx].T            
            return features
            
      
        def transform_logits(mean_z, var_z, epsilon=1e-5):
            return mean_z, jnp.log(1. + jnp.exp(var_z)) + epsilon
        
        
        batched_transform_logits = jax.jit(jax.vmap(transform_logits, in_axes=(0, 0)))

        def gaussian_mixture_moments(mus, sigma_sqs):
            """
            mus, sigma_sqs: shape (N, 1, D)
            returns: (1, D)
            """
            mu = jnp.mean(mus, axis=0)  # (1, D)
            sigma_sq = jnp.mean(sigma_sqs + mus**2, axis=0) - mu**2  # (1, D)
            return mu, sigma_sq


        def uncertainty_separation_parametric(mu, var):
            """
            mu: shape (N, 1, D)
            var: shape (N, 1, D)
            Returns: (1, D), (1, D)
            """
            epistemic_uncertainty = jnp.var(mu, axis=0)  # (1, D)
            aleatoric_uncertainty = jnp.mean(var, axis=0)  # (1, D)
            return aleatoric_uncertainty, epistemic_uncertainty

        def step(carry, u):                        
            sampled_epsilon, state, i = carry            
            eps = jax.lax.dynamic_slice(sampled_epsilon, (i, 0), (1, self.pred_out_dim))
            # batch = jax.lax.dynamic_slice(sampled_epsilon, (i * batch_size, 0), (batch_size, X.shape[1]))
            vx_cmd, wz_cmd = u[0], u[1]
            wheels_xy = compute_wheel_positions(state)            
            grid_feat = get_features_from_grid(wheels_xy[:,0], wheels_xy[:,1],grid_data, grid_center)
            grid_feat = grid_feat.reshape(-1)                      
            state_input = jnp.concatenate([state[3:], u])                        
            
            ## Normalize inputs 
            normalized_grid_feat = (grid_feat - self.constants_feat_mean) / self.constants_feat_std
            normalized_state_input = (state_input - self.constants_state_mean) / self.constants_state_std            
            mlp_input = jnp.concatenate([normalized_grid_feat, normalized_state_input])
           

            means_zs, variances_zs = model_apply(jnp.expand_dims(mlp_input, axis=0))      
            means_zs, variances_zs = batched_transform_logits(means_zs, variances_zs)

            mean_mu_ens, _ = gaussian_mixture_moments(means_zs, variances_zs)  # (1, D)
            ale_ens_var, epi_ens_var = uncertainty_separation_parametric(means_zs, variances_zs)  # both (1, D)
            mean_mu_ens = mean_mu_ens.squeeze()
            ale_ens_var = ale_ens_var.squeeze()
            epi_ens_var = epi_ens_var.squeeze() 
            tot_uncert_var = ale_ens_var + epi_ens_var             

            out_disturbance = mean_mu_ens * self.constants_out_std + self.constants_out_mean         


            out_mean_mu_ens = mean_mu_ens * self.constants_out_std + self.constants_out_mean
            out_tot_uncert_var = tot_uncert_var * self.constants_out_std * self.constants_out_std 
            pred_slip_dists = jnp.stack([out_mean_mu_ens, out_tot_uncert_var], axis=-1)        
                          
            slip_lin, slip_ang = out_disturbance[0], out_disturbance[1]
            
            x, y, yaw, vx, wz = state[0], state[1], state[2], state[3], state[4]
            yaw_next = yaw + self.dt * wz_cmd * slip_ang
            vx_next = jnp.clip(vx_cmd * slip_lin, self.min_vel, self.max_vel)
            wz_next = jnp.clip(wz_cmd * slip_ang, self.min_wz, self.max_wz)
        
    
            state = state.at[0].set(x + self.dt * vx_next * jnp.cos(yaw))
            state = state.at[1].set(y + self.dt * vx_next * jnp.sin(yaw))
            state = state.at[2].set(yaw_next)
            state = state.at[3].set(vx_next)
            state = state.at[4].set(wz_next)
            i_next = i + 1 
            return (sampled_epsilon, state, i_next), (state, pred_slip_dists)
    
        (f_sampled_epsilon, final_state, final_i), (states, pred_slip_dists) = jax.lax.scan(step, (sampled_epsilon, state, 0), us)
        return states, pred_slip_dists  
    
    def reset(self):
        self.u = jnp.zeros_like(self.u)
    
    
    def optimize(self,key,state,goal_xy, grid_data, grid_center, geo_trav_grid_data):
        # self.key = key 
        keys = jax.random.split(key, self.K)       
        
        self.u = jnp.roll(self.u, shift=2, axis=0)

        sampled_epsilon = jax.random.normal(key, shape=(self.K, self.T, 2))
        states = jnp.repeat(state[None, :], self.us.shape[0], axis=0)                        

        '''
        Control input sampling
        '''        
        sampled_controls, ctrl_cost, noise, keys = self.sample_controls(keys, self.u, state)                         
        
        '''
        Forward dynamics
        '''
        pred_states, slips_dist = self.rollout_fn(states,sampled_controls, sampled_epsilon, grid_data, grid_center)

        '''
        Cost computation
        '''                
        ctrl_cost = normalize(ctrl_cost)* self.ctrl_cost_scale

        geo_costs = self.geo_trav_cost(pred_states, slips_dist, geo_trav_grid_data, grid_center)
        geo_costs = normalize(geo_costs)*self.geo_trav_cost_scale
        
        goal_costs = self.goal_cost(goal_xy, pred_states)
        goal_costs = normalize(goal_costs) *self.terminal_dist_costs_scale
        
        mean_slip_cost, var_slip_cost = self.slip_uncertainty_cost(slips_dist)
        mean_slip_cost = normalize(mean_slip_cost) * self.mean_slip_cost_scale
        var_slip_cost = normalize(var_slip_cost) * self.uncert_costs_scale
        
        total_cost = mean_slip_cost + var_slip_cost + goal_costs + geo_costs + ctrl_cost

        '''
        Control upate
        '''
        updated_controls, self.u = self.update_control(total_cost, self.u, state, noise)
        updated_controls.block_until_ready()                 
        return updated_controls, pred_states, total_cost 
        

    def update_cost_scales(self,params):
        self.mean_slip_cost_scale = params['Cost_config']["mean_slip_cost_scale"]
        self.uncert_costs_scale = params['Cost_config']["uncert_costs_scale"]
        self.terminal_dist_costs_scale = params['Cost_config']["terminal_dist_costs_scale"]
        self.geo_trav_cost_scale = params['Cost_config']["geo_trav_cost_scale"]
        self.ctrl_cost_scale = params['Cost_config']["ctrl_cost_scale"] 
        
        
