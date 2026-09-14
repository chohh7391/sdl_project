cmake_minimum_required(VERSION 3.5)
project(tamp)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY
  scripts
  launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  src/
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/server/tamp_server.py
  scripts/server/tamp_real_server.py
  scripts/client/tamp_client.py
  scripts/xdl/tamp_xdl_parser.py
  scripts/bridge/cho_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()