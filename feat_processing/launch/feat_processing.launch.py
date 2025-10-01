import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('feat_processing'),
        'config',
        'feat_param.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'sensor_name',
            default_value='front_cam',
            description='Name of the sensor'
        ),
        Node(
            package='feat_processing',
            executable='feat_processing_node',
            name='feat_processing_node',
            parameters=[config],
            arguments=[LaunchConfiguration('sensor_name')],
            output='screen'
        )
    ])