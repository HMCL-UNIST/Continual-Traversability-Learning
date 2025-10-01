import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('align_planner'),
        'config',
        'data_logger_param.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'logger_name',
            default_value='data_logger',
            description='Name of the logger'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
          DeclareLaunchArgument(
            'save_dir',
            default_value='data',
            description='Directory to save data'
        ),
        Node(
            package='align_planner',
            executable='data_logger_node',
            name='data_logger_node',            
            parameters=[
                config,
                {'save_dir': LaunchConfiguration('save_dir')},
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            arguments=[LaunchConfiguration('logger_name')],
            output='screen'
        )
    ])