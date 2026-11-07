"""Launch the HumanSense node against an existing camera topic.

    ros2 launch humansense_ros humansense.launch.py image_topic:=/camera/image_raw
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("backend", default_value="auto"),
        DeclareLaunchArgument("publish_annotated", default_value="false"),
        Node(package="humansense_ros", executable="humansense_node", name="humansense",
             output="screen",
             parameters=[{"backend": LaunchConfiguration("backend"),
                          "publish_annotated": ParameterValue(LaunchConfiguration("publish_annotated"),
                                                              value_type=bool)}],
             remappings=[("image", LaunchConfiguration("image_topic"))]),
    ])
