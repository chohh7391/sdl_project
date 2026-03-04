cmake_minimum_required(VERSION 3.8)
project(perception_manager)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 의존성 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(perception_interfaces REQUIRED) # 사용자 정의 인터페이스

# 2. 라이브러리 생성 (PerceptionManager 클래스 구현부 컴파일)
add_library(${PROJECT_NAME}
  src/perception_manager.cpp  # 클래스의 멤버 함수들이 구현된 파일
)

# 라이브러리에 의존성 패키지 링크
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  perception_interfaces
)

# 헤더 경로 포함
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 3. 실행 파일 생성 (main 함수가 포함된 파일)
add_executable(perception_manager_node
  src/main.cpp
)

# 실행 파일에 위에서 만든 라이브러리 및 종속성 링크
target_link_libraries(perception_manager_node
  ${PROJECT_NAME}
)

ament_target_dependencies(perception_manager_node
  rclcpp
  # 라이브러리가 이미 종속성을 가지고 있으므로 rclcpp 정도만 명시해도 충분합니다.
)

# 4. 설치 (Install)
install(TARGETS
  ${PROJECT_NAME}
  perception_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# 5. Export 및 패키징
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  perception_interfaces
)

ament_package()