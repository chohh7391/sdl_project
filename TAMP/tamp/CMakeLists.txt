cmake_minimum_required(VERSION 3.5)
project(tamp)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY
  scripts
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  src/
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  scripts/server/tamp_server.py
  scripts/client/tamp_client.py
  scripts/xdl/tamp_xdl_parser.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()