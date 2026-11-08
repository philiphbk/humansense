"""Self-contained demo: replay a video as a camera and run HumanSense on it.

    ros2 launch humansense_ros demo.launch.py source:=/path/to/video.mp4
    ros2 launch humansense_ros demo.launch.py source:=... robot_x:=0.5 robot_y:=0.8 annotated:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("source"),
        DeclareLaunchArgument("backend", default_value="auto"),
        DeclareLaunchArgument("fps", default_value="15.0"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("annotated", default_value="true"),
        DeclareLaunchArgument("robot_x", default_value="nan"),
        DeclareLaunchArgument("robot_y", default_value="nan"),
        Node(package="humansense_ros", executable="video_publisher", name="video_publisher",
             output="screen",
             parameters=[{"source": LaunchConfiguration("source"),
                          "fps": ParameterValue(LaunchConfiguration("fps"), value_type=float),
                          "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool)}],
             remappings=[("image", "/camera/image_raw")]),
        Node(package="humansense_ros", executable="humansense_node", name="humansense",
             output="screen",
             parameters=[{"backend": LaunchConfiguration("backend"),
                          "publish_annotated": ParameterValue(LaunchConfiguration("annotated"),
                                                              value_type=bool),
                          "robot_x": ParameterValue(LaunchConfiguration("robot_x"), value_type=float),
                          "robot_y": ParameterValue(LaunchConfiguration("robot_y"), value_type=float)}],
             remappings=[("image", "/camera/image_raw")]),
    ])
