#!/bin/bash
# ROS2 IsaacSim 录制脚本启动器
# 这个脚本会先 source ROS2 环境，然后运行录制脚本

# 激活 Conda 环境
source /home/amit/conda/etc/profile.d/conda.sh
conda activate lerobot-ros2

# Source ROS2 Humble 环境
source /opt/ros/humble/setup.bash

# 将 src 目录添加到 Python 路径
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 运行录制脚本
python ros2_isaacsim_recorder.py
