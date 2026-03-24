#!/bin/bash

# --- 설정 변수 ---
TIMEOUT_SEC=600  # 10 minutes
ISAACSIM_PATH="$HOME/isaacsim/_build/linux-x86_64/release"
SDL_WS_SETUP="$HOME/sdl_ws/install/local_setup.bash"
SYSTEM_LIBSTDCXX_PATH="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# Conda 초기화 (스크립트 내에서 conda activate를 사용하기 위함)
# conda 설치 경로가 다를 경우 수정이 필요할 수 있습니다.
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh

while true; do
    echo "=========================================================="
    echo "새로운 사이클을 시작합니다: $(date)"
    echo "=========================================================="

    # 1. Isaac Sim 실행 (백그라운드)
    echo "[1/3] Isaac Sim 실행 중..."
    source /opt/ros/humble/setup.bash
    source "$SDL_WS_SETUP"
    
    ros2 launch isaacsim run_isaacsim.launch.py \
        standalone:=$HOME/sdl_ws/src/sdl_project/isaacsim/scripts/standalone/simulation.py \
        install_path:=$ISAACSIM_PATH \
        exclude_install_path:=home/home/sdl_ws/install \
        ros_installation_path:="/home/home/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash,/home/home/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash" &
    ISAAC_PID=$!

    # Isaac Sim이 뜰 때까지 잠시 대기 (상황에 따라 조절)
    sleep 20

    # 2. TAMP Server 실행 (백그라운드)
    echo "[2/3] TAMP Server 실행 중..."
    conda activate sdl
    export SYSTEM_LIBSTDCXX_PATH="$SYSTEM_LIBSTDCXX_PATH"
    
    LD_PRELOAD="${SYSTEM_LIBSTDCXX_PATH}" ros2 run tamp tamp_server.py &
    SERVER_PID=$!

    # Server 준비 대기
    sleep 10

    # 3. TAMP XDL Parser 실행 (백그라운드)
    echo "[3/3] TAMP XDL Parser 실행 중..."
    ros2 run tamp tamp_xdl_parser.py &
    PARSER_PID=$!

    # --- 모니터링 루프 ---
    echo "모니터링 시작 (Timeout: ${TIMEOUT_SEC}초)"
    
    START_TIME=$(date +%s)
    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        # 1. Parser가 종료되었는지 확인
        if ! kill -0 $PARSER_PID 2>/dev/null; then
            echo ">> TAMP XDL Parser가 정상적으로 종료되었습니다."
            break
        fi

        # 2. Timeout 확인
        if [ $ELAPSED -ge $TIMEOUT_SEC ]; then
            echo ">> [TIMEOUT] 7분이 경과하여 프로그램을 강제 종료합니다."
            break
        fi

        sleep 2
    done

    # --- 정리 (Cleanup) ---
    echo "프로세스 정리 중..."
    
    # 프로세스 그룹 전체를 종료하기 위해 pkill 또는 kill 사용
    # Isaac Sim은 자식 프로세스를 많이 생성하므로 pkill이 효과적일 수 있음
    kill $PARSER_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    kill $ISAAC_PID 2>/dev/null
    
    # 남아있을 수 있는 Isaac Sim 관련 프로세스 강제 종료
    pkill -f "isaacsim"
    pkill -f "tamp_server.py"
    
    echo "사이클 종료. 5초 후 재시작합니다..."
    sleep 5
done