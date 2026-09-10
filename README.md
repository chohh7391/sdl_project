# SDL - cuTAMP

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This project utilizes cuTAMP & LLM & Tool Changer within Isaac Sim to perform **S**elf-**D**riving **L**aboratories


# Installation

## Requirements

- Set up IsaacSim ROS2 Environment
  - install [Isaac Sim ROS Workspace](https://github.com/isaac-sim/IsaacSim-ros_workspaces)

    ```bash
    cd ~/
    git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git
    ```

    - Copy `tamp_interfaces` package and paste into `~/IsaacSim-ros_workspaces/humble_ws/src`

    - build python3.11 ros2 packages for IsaacSim
        ```bash
        cd ~/IsaacSim-ros_workspaces

        ./build_ros.sh -d humble -v 22.04
        ```

## Setup Workspace

```bash
sudo apt-get install build-essential
sudo apt-get install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
```

- install isaacsim

    ```bash
    cd ~/
    git clone -b v5.1.0 https://github.com/isaac-sim/IsaacSim.git isaacsim
    ```

    ```bash
    cd isaacsim
    git lfs install
    git lfs pull
    ```

    ```bash
    ./build.sh
    ```


- create workspace

    ```bash
    cd ~/sdl_ws/src
    git clone git@github.com:chohh7391/sdl_project.git
    ```

    - create sdl conda env
    ```bash
    cd ~/sdl_ws/src/sdl_project
    conda env create -f environment.yml
    conda activate sdl
    ```

    ```bash
    pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    cd ~/sdl_ws/src/sdl_project/TAMP/cuTAMP
    pip install -e .
    ```

    ```bash
    sudo apt install git-lfs
    git lfs install
    ```

    ```bash
    cd curobo

    # This can take up to 20 minutes to install
    # make sure cuda version 12.8
    pip install -e . --no-build-isolation
    ```

    ```bash
    conda deactivate
    cd ~/sdl_ws
    colcon build
    ```


# Demo

## Isaac Sim + TAMP launch

One-time: build the custom ROS 2 interfaces (tamp_interfaces, perception_interfaces)
for Isaac Sim's bundled CPython 3.12:

```bash
bash ros2_isaacsim_ws/build_interfaces.sh
```

Then run each side in its own terminal. The scripts derive all paths from their
own location (no hardcoded home paths) and wire up the correct ROS environment:

```bash
# Terminal 1 -- Isaac Sim standalone node (.venv-sdl, py3.12, internal Humble)
bash scripts/run_isaacsim.sh          # add --check to validate the ROS env only
```

```bash
# Terminal 2 -- TAMP / perception node (conda `sdl`, py3.10, system Humble)
bash scripts/run_tamp.sh              # add --parser to launch the XDL parser
```

## Perception Manager
- Run Perception Manager node

(if you want to use GT Pose data, please change `set_tamp_env_cb` function in `sdl_project/TAMP/tamp/scripts/server/tamp_server.py` and `check_entity_distances` function in `sdl_project/TAMP/tamp/scripts/xdl/tamp_xdl_parser.py`)
```bash
source ~/sdl_ws/install/setup.bash
ros2 launch perception_manager perception_manager.launch.py
```

## TAMP

- Run TAMP Server
```bash
source ~/sdl_ws/install/local_setup.bash
conda activate sdl
export SYSTEM_LIBSTDCXX_PATH="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
```

```bash
LD_PRELOAD="${SYSTEM_LIBSTDCXX_PATH}" ros2 run tamp tamp_server.py
```

- Run TAMP with Xdl Parser
```bash
source ~/sdl_ws/install/local_setup.bash
ros2 run tamp tamp_xdl_parser.py
```

- Run TAMP Client

    - run client

    ```bash
    source /opt/ros/humble/setup.bash
    source ~/sdl_ws/install/local_setup.bash
    ```

    ```bash
    ros2 run tamp tamp_client.py
    ```
    
    - set tamp env

      - transfer
        ```bash
        (csuite) set_tamp_env transfer
        ```

      - pouring
        ```bash
        (csuite) set_tamp_env pouring
        ```

      - stirring
        ```bash
        (csuite) set_tamp_env stirring
        ```

    - plan
    ```bash
    (csuite) plan
    ```

    - execute
    ```bash
    (csuite) execute
    ```

    - tool change
      - ag95
        ```bash
        (csuite) tool_change ag95
        ```

      - 2f85
        ```bash
        (csuite) tool_change 2f85
        ```
    
      - vgc10
        ```bash
        (csuite) tool_change vgc10
        ```

      - empty
        ```bash
        (csuite) tool_change empty
        ```

# Trouble Shooting
- can not import llama
In sdl_project/TAMP/tamp/scripts/xdl/tamp_xdl_parser.py, the interpreter shebang must
point at the `sdl` conda env's python. Do not hard-code an absolute per-machine path;
resolve it from the active environment instead.