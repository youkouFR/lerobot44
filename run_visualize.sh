#!/bin/bash
# 数据集可视化脚本启动器

# 激活 Conda 环境
source /home/amit/conda/etc/profile.d/conda.sh
conda activate lerobot-ros2

# 将 src 目录添加到 Python 路径
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 运行可视化脚本
python visualize_recording.py "$@"
