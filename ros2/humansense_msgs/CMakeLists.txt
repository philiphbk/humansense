cmake_minimum_required(VERSION 3.8)
project(humansense_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Person.msg"
  "msg/Cue.msg"
  "msg/Violation.msg"
  "msg/Risk.msg"
  "msg/HumanSenseState.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
