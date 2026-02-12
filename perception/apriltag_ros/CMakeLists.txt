cmake_minimum_required(VERSION 3.8)
project(apriltag_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Werror -Wall -Wextra -Wpedantic)
  add_link_options("-Wl,-z,relro,-z,now,-z,defs")
endif()

option(ASAN "use AddressSanitizer to detect memory issues" OFF)

if(ASAN)
    set(ASAN_FLAGS "\
        -fsanitize=address \
        -fsanitize=bool \
        -fsanitize=bounds \
        -fsanitize=enum \
        -fsanitize=float-cast-overflow \
        -fsanitize=float-divide-by-zero \
        -fsanitize=nonnull-attribute \
        -fsanitize=returns-nonnull-attribute \
        -fsanitize=signed-integer-overflow \
        -fsanitize=undefined \
        -fsanitize=vla-bound \
        -fno-sanitize=alignment \
        -fsanitize=leak \
        -fsanitize=object-size \
    ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(apriltag_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d)
find_package(apriltag 3.2 REQUIRED)

if(cv_bridge_VERSION VERSION_GREATER_EQUAL 3.3.0)
    add_compile_definitions(cv_bridge_HPP)
endif()

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# database of tag functions
# This is a good modular approach. The source file remains the same.
add_library(tags STATIC
    src/tag_functions.cpp
)
target_link_libraries(tags
    apriltag::apriltag
)
set_property(TARGET tags PROPERTY
    POSITION_INDEPENDENT_CODE ON
)

# conversion functions as template specialisation
# I'm assuming you'll move this logic into a header file as discussed, so we
# won't create a separate library for it. Instead, we'll expose the header.
# If you keep it as a source file, the original code is fine.
# For a cleaner approach, a header-only solution is better.
# For now, let's keep the original structure as it's also a valid approach.
add_library(conversion STATIC
    src/conversion.cpp
)
target_link_libraries(conversion
    apriltag::apriltag
    Eigen3::Eigen
    opencv_core
)
ament_target_dependencies(conversion
    geometry_msgs
    tf2
)
set_property(TARGET conversion PROPERTY
    POSITION_INDEPENDENT_CODE ON
)

# pose estimation methods
# This is also a good modular approach. The source file remains the same.
add_library(pose_estimation STATIC
    src/pose_estimation.cpp
)
target_link_libraries(pose_estimation
    apriltag::apriltag
    Eigen3::Eigen
    opencv_calib3d
    conversion
)
ament_target_dependencies(pose_estimation
    geometry_msgs
    tf2
)
set_property(TARGET pose_estimation PROPERTY
    POSITION_INDEPENDENT_CODE ON
)

add_executable(apriltag_node
    src/AprilTagNode.cpp
)
target_link_libraries(apriltag_node
    tags
    conversion
    pose_estimation
    apriltag::apriltag
)
ament_target_dependencies(apriltag_node
    rclcpp
    sensor_msgs
    apriltag_msgs
    tf2_ros
    image_transport
    cv_bridge
)

install(TARGETS
    tags
    conversion
    pose_estimation
    apriltag_node
    EXPORT ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    include/
    DESTINATION include
)

install(DIRECTORY cfg
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  set(ament_cmake_clang_format_CONFIG_FILE "${CMAKE_SOURCE_DIR}/.clang-format")
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()