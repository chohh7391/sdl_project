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

cmake_minimum_required(VERSION 3.8)
project(sdl_cell_bringup)

find_package(ament_cmake REQUIRED)

# The cell's robot registry: the installed core one plus
# config/robot_config_additions.yaml (scripts/make_robot_config.py). It depends
# on the core's files too, so a core rebuild with a changed registry regenerates
# it.
ament_index_has_resource(_core_registry_prefix packages cho_robot_config)
if(NOT _core_registry_prefix)
  message(FATAL_ERROR "cho_robot_config is not installed: source the core workspace first")
endif()
file(GLOB _core_registry_files "${_core_registry_prefix}/share/cho_robot_config/config/*.yaml")
set(_robot_config_dir "${CMAKE_CURRENT_BINARY_DIR}/robot_config")
set(_robot_config_stamp "${CMAKE_CURRENT_BINARY_DIR}/robot_config.stamp")
add_custom_command(
  OUTPUT "${_robot_config_stamp}"
  COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_robot_config.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/robot_config_additions.yaml" "${_robot_config_dir}"
  COMMAND ${CMAKE_COMMAND} -E touch "${_robot_config_stamp}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_robot_config.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/config/robot_config_additions.yaml"
    ${_core_registry_files}
  COMMENT "Generating the cell's robot registry")
add_custom_target(robot_config ALL DEPENDS "${_robot_config_stamp}")
install(DIRECTORY "${_robot_config_dir}/" DESTINATION share/${PROJECT_NAME}/robot_config)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/robot_config.dsv.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/robot_config.sh.in")

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/.flake8")
  ament_add_pytest_test(test_robot_config test/test_robot_config.py TIMEOUT 120)
  ament_add_pytest_test(test_launch_files test/test_launch_files.py TIMEOUT 120)
endif()

ament_package()
