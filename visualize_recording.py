#!/usr/bin/env python3
"""
可视化录制的 LeRobot 数据集

这个脚本用于可视化查看使用 ros2_isaacsim_recorder.py 录制的数据

使用方法:
    python visualize_recording.py --dataset-path ./isaacsim_dataset --episode 0

依赖:
    - rerun-sdk
    - lerobot
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset


def main():
    parser = argparse.ArgumentParser(
        description="可视化 LeRobot 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 可视化本地数据集
    python visualize_recording.py --dataset-path ./isaacsim_dataset --episode 0

    # 保存为 .rrd 文件
    python visualize_recording.py --dataset-path ./isaacsim_dataset --episode 0 --save --output-dir ./visualizations

    # 指定数据集 ID
    python visualize_recording.py --dataset-path ./isaacsim_dataset --repo-id isaacsim/robot_recording --episode 0
        """
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="数据集本地路径 (例如: ./isaacsim_dataset)"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="isaacsim/robot_recording",
        help="数据集 ID (默认: isaacsim/robot_recording)"
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="要可视化的 episode 索引 (默认: 0)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="保存为 .rrd 文件而不是直接显示"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="保存 .rrd 文件的目录 (默认: ./visualizations)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="数据加载的 batch size (默认: 32)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="可视化帧率 (默认: 30)"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("LeRobot 数据集可视化")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print(f"数据集 ID: {args.repo_id}")
    print(f"Episode: {args.episode}")
    print(f"模式: {'保存为 .rrd 文件' if args.save else '直接显示'}")
    print("=" * 60)

    # 加载数据集
    print("正在加载数据集...")
    try:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=dataset_path,
            episodes=[args.episode]
        )
        print(f"数据集加载成功!")
        print(f"  - 总 episodes: {dataset.meta.total_episodes}")
        print(f"  - 总 frames: {dataset.meta.total_frames}")
        print(f"  - FPS: {dataset.fps}")
        print(f"  - 相机: {dataset.meta.camera_keys}")
        print(f"  - 动作维度: {dataset.features.get('action', {}).get('shape', 'N/A')}")
        print(f"  - 状态维度: {dataset.features.get('observation.state', {}).get('shape', 'N/A')}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)

    # 可视化
    print("\n启动可视化...")
    print("提示: 按 Ctrl+C 退出")
    print("=" * 60)

    try:
        output_path = visualize_dataset(
            dataset=dataset,
            episode_index=args.episode,
            batch_size=args.batch_size,
            save=args.save,
            output_dir=Path(args.output_dir) if args.save else None,
            mode="local"
        )

        if output_path:
            print(f"\n可视化文件已保存: {output_path}")
            print(f"使用以下命令查看:")
            print(f"  rerun {output_path}")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
