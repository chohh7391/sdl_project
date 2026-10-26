# Copyright 2026 Hyunho Cho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(sdl_pour_controller)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# The build type comes from the command line (-DCMAKE_BUILD_TYPE=Release), as
# for cho_controller_fr5. Never -Ofast: it folds isfinite() / isnan() to
# constants, and the finite guards here rely on them.
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pinocchio REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cho_controller_common REQUIRED)
find_package(cho_controller_base REQUIRED)
# FR5BaseController. The installed package exports its include directory and
# libcho_controller_fr5.so the classic way (no CMake targets), with everything
# its headers include as exported dependencies.
find_package(cho_controller_fr5 REQUIRED)
find_package(sdl_cell_interfaces REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/pouring_controller.cpp
    src/pour/container_gate.cpp
    src/pour/flow_model.cpp
    src/pour/guards.cpp
    src/pour/level.cpp
    src/pour/lip_path.cpp
    src/pour/material_profile.cpp
    src/pour/pour_planner.cpp
    src/pour/pour_types.cpp
    src/pour/scale_filter.cpp
    src/pour/shaping_law.cpp
    src/servers/pour_action_server.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(${PROJECT_NAME}
    cho_controller_fr5
    cho_controller_base
    cho_controller_common
    controller_interface
    hardware_interface
    pinocchio
    pluginlib
    rclcpp
    rclcpp_action
    rclcpp_lifecycle
    realtime_tools
    geometry_msgs
    sdl_cell_interfaces
)

pluginlib_export_plugin_description_file(
    controller_interface sdl_pour_controller.xml)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(
    DIRECTORY include/
    DESTINATION include
)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)

    # The pour law only. These link src/pour/ directly and pull in no ROS at
    # all, which is the point of that subdirectory: a material's behaviour can
    # be pinned against a synthetic vessel without a controller_manager, a
    # scale, or a robot.
    ament_add_gtest(test_pour
        test/test_pour.cpp
        src/pour/container_gate.cpp
        src/pour/flow_model.cpp
        src/pour/guards.cpp
        src/pour/material_profile.cpp
        src/pour/pour_planner.cpp
        src/pour/pour_types.cpp
        src/pour/scale_filter.cpp
        src/pour/shaping_law.cpp
    )
    target_include_directories(test_pour PRIVATE include test)

    # Where the lip goes, from where the vessel sits in the gripper. Eigen only.
    ament_add_gtest(test_lip_path
        test/test_lip_path.cpp
        src/pour/lip_path.cpp
    )
    target_include_directories(test_lip_path PRIVATE include ${EIGEN3_INCLUDE_DIRS})

    # Standing the held vessel upright before the pour. Eigen only.
    ament_add_gtest(test_level
        test/test_level.cpp
        src/pour/level.cpp
    )
    target_include_directories(test_level PRIVATE include ${EIGEN3_INCLUDE_DIRS})

    # Not a test: prints both laws' results on pour_sim.hpp's vessels, which is
    # where the comparison quoted in pouring_controller.yaml comes from. Built
    # with the tests, left in the build tree, never installed.
    add_executable(pour_ab
        test/pour_ab.cpp
        src/pour/container_gate.cpp
        src/pour/flow_model.cpp
        src/pour/guards.cpp
        src/pour/material_profile.cpp
        src/pour/pour_planner.cpp
        src/pour/pour_types.cpp
        src/pour/scale_filter.cpp
        src/pour/shaping_law.cpp
    )
    target_include_directories(pour_ab PRIVATE include test)
endif()

# Everything the installed headers include, so a package that includes them
# finds it through find_package(sdl_pour_controller) alone.
ament_export_dependencies(
    cho_controller_fr5
    cho_controller_base
    cho_controller_common
    controller_interface
    Eigen3
    geometry_msgs
    hardware_interface
    pinocchio
    pluginlib
    rclcpp
    rclcpp_action
    rclcpp_lifecycle
    realtime_tools
    sdl_cell_interfaces
)
ament_package()
