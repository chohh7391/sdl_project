cmake_minimum_required(VERSION 3.8)
project(perception_interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Action
set(ACTION_FILES
  action/JointPosture.action
  action/SE3.action
  action/SE3Array.action
  action/Gripper.action
  action/Move.action
  action/QR.action
  action/ToolChanger.action
)

# Service
set(SERVICE_FILES
  srv/get_object_pose.srv
  srv/get_ft_data.srv
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/JointPosture.action"
  "action/SE3.action"
  "action/SE3Array.action"
  "action/Gripper.action"
  "action/Move.action"
  "action/QR.action"
  "action/ToolChanger.action"
  "srv/GetObjectInfo.srv"
  "srv/GetFtData.srv"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces sensor_msgs)

install(
  DIRECTORY action srv
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
