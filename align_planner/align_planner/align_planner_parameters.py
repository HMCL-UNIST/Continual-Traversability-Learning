from dataclasses import dataclass, field
from simple_parsing.helpers import Serializable
from typing import List, Optional

@dataclass
class FeatArgs:
    world_feat_size: int = 16
    state_feat_size: int = 4
    deviation_size: int = 2
    hidden_units: List[int] = field(default_factory=lambda: [24, 48, 24])
    latent_size: int = 6
    z_var: float = 1.0
    kernel_type: str = 'rbf'
    weight: Optional[str] = ''
    device: str = 'cuda' 
    tensorboard: bool = False

@dataclass
class EnsembleArgs:
    num_models: int = 5
    num_recall: int = 20
    output_size: int = 2
    tensorboard: bool = True
    align_loss_enable: bool = True
    off_recall_enable: bool = True
    weight: Optional[str] = None
    model_dir: str = 'models'
    ensemble_weight: str = 'ensemble_epoch_21'


@dataclass
class TravNNArgs:
    world_feat_size: int = 16 
    grid_feat_size: int = 4
    state_feat_size: int = 4
    input_size: int = 20  
    trav_hidden_units: List[int] = field(default_factory=lambda: [16,32,32,16])
    trav_output_size: int = 2
    device: str = 'cuda'  




@dataclass
class CostConfig(Serializable):
    goal_w: float = 1.0
    speed_w: float = 0.1
    roll_w: float = 0.0
    lethal_w: float = 1.0
    stop_w: float = 0.0
    speed_target: float = 2.5
    critical_SA: float = 1.414
    critical_RI: float = 0.8
    car_bb_width: float = 0.3
    car_bb_length: float = 0.4
    critical_vert_acc: float = 4.0
    critical_vert_spd: float = 0.2    
    mean_slip_cost_scale: float = 1.0
    uncert_costs_scale: float = 1.0
    terminal_dist_costs_scale: float = 1.0
    geo_trav_cost_scale: float = 1.0
    ctrl_cost_scale: float = 1.0
    var_clip_min: float = 5e-2
    var_clip_max: float = 2e0
    jit_enable: bool = True

@dataclass
class DynamicsConfig(Serializable):
    wheelbase: float = 0.3
    throttle_to_wheelspeed: float = 3.0
    steering_max: float = 0.42
    dt: float = 0.1
    D: float = 1.0
    B: float = 6.8
    C: float = 1.5
    lf: float = 0.15
    lr: float = 0.15
    Iz: float = 0.1
    LPF_tau: float = 0.2
    res_coeff: float = 0.01
    drag_coeff: float = 0.005
    car_length: float = 0.3
    car_width: float = 0.4
    cg_height: float = 0.225
    type: str = "slip3d"
    jit_enable: bool = True

@dataclass
class SamplingConfig(Serializable):
    state_dim: int = 5
    control_dim: int = 2
    noise_0: float = 1.0
    noise_1: float = 0.5
    scaled_dt: float = 0.1
    temperature: float = 0.02
    max_vel: float = 1.0
    min_vel: float = -1.0
    max_yaw_rate: float = 1.0
    max_ax: float = 1.0
    min_ax: float = -1.0
    max_alpha_x: float = 1.0
    min_alpha_x: float = -1.0
    

@dataclass
class MPPIConfig(Serializable):
    ROLLOUTS: int = 2
    BATCHSIZE: int = 1024
    TIMESTEPS: int = 30
    BINS: int = 1
    u_per_command: int = 1    

@dataclass
class MapConfig(Serializable):
    map_size: float = 10.25
    map_res: float = 0.25    
    elevation_range: float = 4.0        
    layers: List[str] = field(default_factory=lambda: [
        'is_valid', 'elevation', 'normal_x', 'normal_y', 'normal_z', 'feat_0', 'feat_1', 'feat_2'
    ])

@dataclass
class VehicleConfig(Serializable):
    make: str = "track"
    model: str = "flux"
    max_speed: float = 1.0
    max_steer: float = 0.488

@dataclass
class MppiPlannerConfig(Serializable):
    Cost_config: CostConfig = field(default_factory=CostConfig) 
    Dynamics_config: DynamicsConfig = field(default_factory=DynamicsConfig)  
    Sampling_config: SamplingConfig = field(default_factory=SamplingConfig)      
    Feat_args: FeatArgs = field(default_factory=FeatArgs)
    Ensemble_args: EnsembleArgs = field(default_factory=EnsembleArgs)
    Travnn_args: TravNNArgs = field(default_factory=TravNNArgs)
    MPPI_config: MPPIConfig = field(default_factory=MPPIConfig)
    Map_config: MapConfig = field(default_factory=MapConfig)
    wp_radius: float = 0.25
    lookahead: float = 3.0
    track_width: float = 0.25
    generate_costmap_from_path: int = 1
    speed_limit: float = 3.0
    debug: bool = False
    scenario: str = "test"
    map_name: str = "smallgrid"
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    use_speed_ctrl: bool = False
    skip_points: int = 1
    elevation_multiplier: float = 1.0
    use_yaml: bool = False
    waypoint_file: str = "waypoints.yaml"
    ctrl_loop_rate: float = 10.0
    opt_loop_rate: float = 10.0
    grid_call_rate: float = 2.0
    odom_topic: str = '/odom'
    cmd_topic: str = '/cmd_vel'
    cmd_pub_enble: bool = False
    manual_ctrl: bool = False




    @classmethod
    def get_param_names(cls, config_class=None, prefix=''):
        if config_class is None:
            config_class = cls
        param_names = []
        for field_name, field_type in config_class.__annotations__.items():
            if hasattr(field_type, '__annotations__'):
                param_names.extend(cls.get_param_names(field_type, prefix + field_name + '.'))
            else:
                param_names.append(prefix + field_name)
        return param_names



    @classmethod
    def update_dict_recursive(cls, target=None, source=None):
        for key, value in source.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recursively update the target dictionary
                target[key] = cls.update_dict_recursive(target.get(key, {}), value)
            else:
                # Otherwise, just set the value
                target[key] = value
        return target

    @classmethod
    def create_hierarchical_dict(cls, param_list):
        param_dict = {}
        for param in param_list:
            keys = param.name.split('.')
            d = param_dict
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            d[keys[-1]] = param.value
        return param_dict
    