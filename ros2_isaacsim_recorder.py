#!/usr/bin/env python3
"""
ROS2 IsaacSim 数据录制脚本

这个脚本用于录制 IsaacSim 中的 ROS2 topic 数据，并保存为 LeRobot 数据集格式。
录制的 topic 包括:
- /Head: 头部相机 RGB 图像
- /left: 左手腕相机 RGB 图像
- /right: 右手腕相机 RGB 图像
- /joint_states: 机器人关节状态
- /joint_command: 机器人关节控制命令 (action)

使用方法:
    1. 启动 IsaacSim 和 ROS2 节点
    2. 运行 ros2_robot_arm_controller.py 发布控制命令
    3. 运行此脚本: python ros2_isaacsim_recorder.py

依赖:
    - rclpy
    - sensor_msgs
    - cv_bridge
    - lerobot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import threading
import signal
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR


# ==================== 配置参数 ====================
# 数据集配置
DATASET_REPO_ID = "isaacsim/robot_recording"  # 数据集名称
DATASET_ROOT = "./isaacsim_dataset"  # 本地保存路径
FPS = 30  # 录制帧率
TASK_DESCRIPTION = "IsaacSim robot manipulation task"  # 任务描述

# 图像配置
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# 关节配置 (根据你的机器人调整)
# 从 ros2_robot_arm_controller.py 中获取的关节名称
L_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"
]
R_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
]
ALL_JOINTS = L_JOINTS + R_JOINTS
NUM_JOINTS = len(ALL_JOINTS)

# ROS2 Topic 名称
HEAD_CAMERA_TOPIC = "/Head"
LEFT_CAMERA_TOPIC = "/left"
RIGHT_CAMERA_TOPIC = "/right"
JOINT_STATES_TOPIC = "/joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"


class IsaacSimRecorder(Node):
    """IsaacSim 数据录制节点"""

    def __init__(self, dataset: LeRobotDataset):
        super().__init__("isaacsim_recorder")

        self.dataset = dataset
        self.bridge = CvBridge()

        # 数据缓存
        self.head_image = None
        self.left_image = None
        self.right_image = None
        self.joint_states = None
        self.joint_command = None

        # 数据锁
        self.data_lock = threading.Lock()
        self.recording = False
        self.episode_frame_count = 0

        # 创建订阅者
        self.head_sub = self.create_subscription(
            Image, HEAD_CAMERA_TOPIC, self.head_callback, 10
        )
        self.left_sub = self.create_subscription(
            Image, LEFT_CAMERA_TOPIC, self.left_callback, 10
        )
        self.right_sub = self.create_subscription(
            Image, RIGHT_CAMERA_TOPIC, self.right_callback, 10
        )
        self.joint_states_sub = self.create_subscription(
            JointState, JOINT_STATES_TOPIC, self.joint_states_callback, 10
        )
        self.joint_command_sub = self.create_subscription(
            JointState, JOINT_COMMAND_TOPIC, self.joint_command_callback, 10
        )

        # 创建录制定时器
        timer_period = 1.0 / FPS
        self.timer = self.create_timer(timer_period, self.record_frame)

        self.get_logger().info("=" * 60)
        self.get_logger().info("IsaacSim 数据录制器已启动")
        self.get_logger().info(f"订阅的 Topics:")
        self.get_logger().info(f"  - 头部相机: {HEAD_CAMERA_TOPIC}")
        self.get_logger().info(f"  - 左手相机: {LEFT_CAMERA_TOPIC}")
        self.get_logger().info(f"  - 右手相机: {RIGHT_CAMERA_TOPIC}")
        self.get_logger().info(f"  - 关节状态: {JOINT_STATES_TOPIC}")
        self.get_logger().info(f"  - 关节命令: {JOINT_COMMAND_TOPIC}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("按 's' 开始录制, 'q' 停止录制并保存")

    def head_callback(self, msg: Image):
        """接收头部相机图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.data_lock:
                self.head_image = cv_image
        except Exception as e:
            self.get_logger().error(f"头部相机图像转换失败: {e}")

    def left_callback(self, msg: Image):
        """接收左手腕相机图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.data_lock:
                self.left_image = cv_image
        except Exception as e:
            self.get_logger().error(f"左手相机图像转换失败: {e}")

    def right_callback(self, msg: Image):
        """接收右手腕相机图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.data_lock:
                self.right_image = cv_image
        except Exception as e:
            self.get_logger().error(f"右手相机图像转换失败: {e}")

    def joint_states_callback(self, msg: JointState):
        """接收关节状态"""
        try:
            # 将关节状态按名称排序，确保顺序一致
            joint_positions = self._extract_joint_positions(msg)
            with self.data_lock:
                self.joint_states = joint_positions
        except Exception as e:
            self.get_logger().error(f"关节状态处理失败: {e}")

    def joint_command_callback(self, msg: JointState):
        """接收关节控制命令 (作为 action)"""
        try:
            # 将关节命令按名称排序，确保顺序一致
            joint_positions = self._extract_joint_positions(msg)
            with self.data_lock:
                self.joint_command = joint_positions
        except Exception as e:
            self.get_logger().error(f"关节命令处理失败: {e}")

    def _extract_joint_positions(self, msg: JointState) -> np.ndarray:
        """从 JointState 消息中提取关节位置，按预定义顺序排列"""
        positions = np.zeros(NUM_JOINTS, dtype=np.float32)

        # 创建名称到位置的映射
        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}

        # 按 ALL_JOINTS 顺序填充
        for i, joint_name in enumerate(ALL_JOINTS):
            if joint_name in name_to_pos:
                positions[i] = name_to_pos[joint_name]
            else:
                self.get_logger().warn(f"关节 {joint_name} 未在消息中找到")

        return positions

    def record_frame(self):
        """录制一帧数据"""
        if not self.recording:
            return

        with self.data_lock:
            # 检查所有数据是否就绪
            if (self.head_image is None or
                self.left_image is None or
                self.right_image is None or
                self.joint_states is None or
                self.joint_command is None):
                self.get_logger().warn("数据未完全就绪，跳过此帧")
                return

            # 调试信息：检查图像数据
            if self.episode_frame_count == 0:
                self.get_logger().info(f"头部图像形状: {self.head_image.shape}, 数据类型: {self.head_image.dtype}")
                self.get_logger().info(f"左手图像形状: {self.left_image.shape}, 数据类型: {self.left_image.dtype}")
                self.get_logger().info(f"右手图像形状: {self.right_image.shape}, 数据类型: {self.right_image.dtype}")
                
                # 保存第一帧图像用于检查
                import cv2
                import os
                debug_dir = "./debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "head_0.jpg"), self.head_image)
                cv2.imwrite(os.path.join(debug_dir, "left_0.jpg"), self.left_image)
                cv2.imwrite(os.path.join(debug_dir, "right_0.jpg"), self.right_image)
                self.get_logger().info(f"已保存调试图像到 {debug_dir}")

            # 构建帧数据 (注意：OpenCV 是 BGR 格式，需要转换为 RGB)
            import cv2
            frame = {
                "observation.images.head": cv2.cvtColor(self.head_image, cv2.COLOR_BGR2RGB),
                "observation.images.left": cv2.cvtColor(self.left_image, cv2.COLOR_BGR2RGB),
                "observation.images.right": cv2.cvtColor(self.right_image, cv2.COLOR_BGR2RGB),
                "observation.state": self.joint_states.copy(),
                "action": self.joint_command.copy(),
                "task": TASK_DESCRIPTION,
            }

        # 添加帧到数据集
        try:
            self.dataset.add_frame(frame)
            self.episode_frame_count += 1

            # 每 30 帧打印一次日志
            if self.episode_frame_count % 30 == 0:
                self.get_logger().info(f"已录制 {self.episode_frame_count} 帧")

        except Exception as e:
            self.get_logger().error(f"添加帧失败: {e}")

    def start_recording(self):
        """开始录制"""
        if not self.recording:
            self.recording = True
            self.episode_frame_count = 0
            self.get_logger().info("=" * 60)
            self.get_logger().info("开始录制!")
            self.get_logger().info("=" * 60)

    def stop_recording(self):
        """停止录制并保存"""
        if self.recording:
            self.recording = False
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"停止录制，共录制 {self.episode_frame_count} 帧")
            self.get_logger().info("保存 episode...")

            try:
                self.dataset.save_episode()
                self.get_logger().info("Episode 保存成功!")
            except Exception as e:
                self.get_logger().error(f"保存 episode 失败: {e}")

            self.episode_frame_count = 0


def create_dataset_features():
    """创建数据集特征定义"""
    # 动作特征: 关节位置命令
    action_features = {
        joint_name: float for joint_name in ALL_JOINTS
    }

    # 观察特征: 关节状态 + 相机图像
    observation_features = {
        joint_name: float for joint_name in ALL_JOINTS
    }
    observation_features.update({
        "head": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),  # RGB 图像
        "left": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "right": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    })

    # 转换为 LeRobot 格式
    action_dataset_features = hw_to_dataset_features(action_features, ACTION)
    obs_dataset_features = hw_to_dataset_features(observation_features, OBS_STR)

    # 合并特征
    features = {**action_dataset_features, **obs_dataset_features}

    return features


def main():
    """主函数"""
    # 创建数据集
    features = create_dataset_features()

    print("=" * 60)
    print("创建 LeRobot 数据集...")
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"数据集 ID: {DATASET_REPO_ID}")
    print(f"帧率: {FPS} FPS")
    print("=" * 60)

    # 如果数据集已存在则加载，否则创建新的
    dataset_root = Path(DATASET_ROOT)
    if dataset_root.exists():
        print("检测到已有数据集，将追加新数据...")
        dataset = LeRobotDataset(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
        )
    else:
        print("创建新数据集...")
        print("使用 H.264 编码器（兼容性更好）")
        try:
            dataset = LeRobotDataset.create(
                repo_id=DATASET_REPO_ID,
                fps=FPS,
                features=features,
                robot_type="isaacsim_dual_arm",
                use_videos=True,  # 使用视频格式存储图像
                image_writer_threads=4,
            )
            # 设置视频编码器为 H.264，兼容性更好
            dataset.vcodec = "h264"
        except FileExistsError:
            # 如果默认路径已存在，使用自定义路径
            custom_repo_id = f"isaacsim/robot_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"默认路径已存在，使用自定义路径: {custom_repo_id}")
            dataset = LeRobotDataset.create(
                repo_id=custom_repo_id,
                fps=FPS,
                features=features,
                robot_type="isaacsim_dual_arm",
                use_videos=True,
                image_writer_threads=4,
            )
            # 设置视频编码器为 H.264，兼容性更好
            dataset.vcodec = "h264"

    # 初始化 ROS2
    rclpy.init()
    recorder = IsaacSimRecorder(dataset)

    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n接收到中断信号，正在退出...")
        recorder.stop_recording()
        recorder.destroy_node()
        rclpy.shutdown()

        # 完成数据集
        print("完成数据集...")
        dataset.finalize()
        print(f"数据集已保存到: {DATASET_ROOT}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 简单的键盘控制
    print("\n控制命令:")
    print("  s - 开始录制")
    print("  q - 停止录制并保存 episode")
    print("  x - 退出程序")
    print("=" * 60)

    # 在单独的线程中运行 ROS2  spin
    spin_thread = threading.Thread(target=rclpy.spin, args=(recorder,))
    spin_thread.start()

    try:
        while True:
            cmd = input().strip().lower()

            if cmd == 's':
                recorder.start_recording()
            elif cmd == 'q':
                recorder.stop_recording()
            elif cmd == 'x':
                break
            else:
                print("未知命令，使用: s(开始), q(停止), x(退出)")

    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        recorder.stop_recording()
        recorder.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

        # 完成数据集
        print("=" * 60)
        print("完成数据集...")
        dataset.finalize()
        print(f"数据集已保存到: {DATASET_ROOT}")
        print(f"总 episodes: {dataset.meta.total_episodes}")
        print(f"总 frames: {dataset.meta.total_frames}")
        print("=" * 60)


if __name__ == "__main__":
    main()
