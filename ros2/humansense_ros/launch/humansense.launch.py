"""Launch the HumanSense node against an existing camera topic.

    ros2 launch humansense_ros humansense.launch.py image_topic:=/camera/image_raw
    ros2 launch humansense_ros humansense.launch.py image_topic:=... robot_x:=0.5 robot_y:=0.8
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
        DeclareLaunchArgument("robot_x", default_value="nan"),
        DeclareLaunchArgument("robot_y", default_value="nan"),
        Node(package="humansense_ros", executable="humansense_node", name="humansense",
             output="screen",
             parameters=[{"backend": LaunchConfiguration("backend"),
                          "publish_annotated": ParameterValue(LaunchConfiguration("publish_annotated"),
                                                              value_type=bool),
                          "robot_x": ParameterValue(LaunchConfiguration("robot_x"), value_type=float),
                          "robot_y": ParameterValue(LaunchConfiguration("robot_y"), value_type=float)}],
             remappings=[("image", LaunchConfiguration("image_topic"))]),
    ])
