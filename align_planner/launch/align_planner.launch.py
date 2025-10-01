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
        'align_planner_param.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'planner_name',
            default_value='mppi',
            description='Name of the planner'
        ),
        Node(
            package='align_planner',
            executable='align_planner_node',
            name='align_planner_node',
            parameters=[config],
            arguments=[LaunchConfiguration('planner_name')],
            output='screen'
        )
    ])