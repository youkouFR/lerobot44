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
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import threading
import signal
import sys
from collections import deque

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR


# ==================== 配置参数 ====================
# 数据集配置
DATASET_REPO_ID = "isaacsim/robot_recording"  # 数据集名称
DATASET_ROOT = "./isaacsim_dataset"  # 本地保存路径
FPS = 30  # 录制帧率

# 预定义任务列表
AVAILABLE_TASKS = [
    {
        "task": "remove_buffer",
        "language_instruction": "Remove the buffer material"
    },
    {
        "task": "place_buffer",
        "language_instruction": "Place the buffer material down"
    },
    {
        "task": "pick_cutting_tool",
        "language_instruction": "Pick up the cutting tool"
    },
    {
        "task": "align_steel_strip",
        "language_instruction": "Align with the steel strip"
    },
    {
        "task": "place_cutting_tool",
        "language_instruction": "Put down the cutting tool"
    }
]

# 图像配置
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# 关节配置 (根据你的机器人调整)
# 从 ros2_robot_arm_controller.py 中获取的关节名称
L_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "left_hand_index_bend_joint","left_hand_index_joint1","left_hand_index_joint2","left_hand_mid_joint1",
    "left_hand_mid_joint2","left_hand_pinky_joint1","left_hand_pinky_joint2","left_hand_ring_joint1",
    "left_hand_ring_joint2","left_hand_thumb_bend_joint","left_hand_thumb_rota_joint1","left_hand_thumb_rota_joint2"

]
R_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "right_hand_index_bend_joint","right_hand_index_joint1","right_hand_index_joint2","right_hand_mid_joint1",
    "right_hand_mid_joint2","right_hand_pinky_joint1","right_hand_pinky_joint2","right_hand_ring_joint1",
    "right_hand_ring_joint2","right_hand_thumb_bend_joint","right_hand_thumb_rota_joint1","right_hand_thumb_rota_joint2"
]
ALL_JOINTS = L_JOINTS + R_JOINTS
NUM_JOINTS = len(ALL_JOINTS)

# ROS2 Topic 名称
HEAD_CAMERA_TOPIC = "/Head"
LEFT_CAMERA_TOPIC = "/left"
RIGHT_CAMERA_TOPIC = "/right"
JOINT_STATES_TOPIC = "/joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"
HAND_EE_TOPIC = "/hand_ee"
ARM_EE_TOPIC = "/arm_ee"  # 目标末端执行器位姿（action）
LEROBOT_COMMAND_TOPIC = "/lerobot_command"  # 录制控制命令


class IsaacSimRecorder(Node):
    """IsaacSim 数据录制节点"""

    def __init__(self, dataset: LeRobotDataset, current_task: dict):
        super().__init__("isaacsim_recorder")

        self.dataset = dataset
        self.current_task = current_task  # 当前选择的任务
        self.bridge = CvBridge()

        # 带时间戳的数据缓冲区（用于时间戳对齐）
        self.head_buffer = deque(maxlen=10)
        self.left_buffer = deque(maxlen=10)
        self.right_buffer = deque(maxlen=10)
        self.joint_states_buffer = deque(maxlen=10)
        self.joint_command_buffer = deque(maxlen=10)
        self.hand_ee_buffer = deque(maxlen=10)  # 当前末端执行器位姿（observation）
        self.arm_ee_buffer = deque(maxlen=10)   # 目标末端执行器位姿（action）

        # 数据锁
        self.data_lock = threading.Lock()
        self.recording = False
        self.episode_frame_count = 0

        # 时间戳对齐的最大延迟（秒）
        self.max_timestamp_delay = 0.05

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
        self.hand_ee_sub = self.create_subscription(
            TFMessage, HAND_EE_TOPIC, self.hand_ee_callback, 10
        )
        self.arm_ee_sub = self.create_subscription(
            TFMessage, ARM_EE_TOPIC, self.arm_ee_callback, 10
        )
        self.lerobot_command_sub = self.create_subscription(
            String, LEROBOT_COMMAND_TOPIC, self.lerobot_command_callback, 10
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
        self.get_logger().info(f"  - 当前末端执行器: {HAND_EE_TOPIC}")
        self.get_logger().info(f"  - 目标末端执行器: {ARM_EE_TOPIC}")
        self.get_logger().info(f"  - 录制控制命令: {LEROBOT_COMMAND_TOPIC}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("按 's' 开始录制, 'q' 停止录制并保存")
        self.get_logger().info("或通过 /lerobot_command 发送 'record_start'/'record_stop' 控制")

    def lerobot_command_callback(self, msg: String):
        """接收录制控制命令"""
        command = msg.data.strip().lower()
        self.get_logger().info(f"收到录制命令: {command}")
        
        if command == "record_start":
            self.start_recording()
        elif command == "record_stop":
            self.stop_recording()
        else:
            self.get_logger().warn(f"未知命令: {command}，支持的命令: record_start, record_stop")

    def head_callback(self, msg: Image):
        """接收头部相机图像"""
        try:
            # 检查图像数据是否为空
            if not msg.data:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 检查转换后的图像是否为空
            if cv_image.size == 0:
                return
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.data_lock:
                self.head_buffer.append((timestamp, cv_image))
        except Exception as e:
            # 只在录制时记录错误，避免启动时的错误刷屏
            if self.recording:
                self.get_logger().error(f"头部相机图像转换失败: {e}")

    def left_callback(self, msg: Image):
        """接收左手腕相机图像"""
        try:
            # 检查图像数据是否为空
            if not msg.data:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 检查转换后的图像是否为空
            if cv_image.size == 0:
                return
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.data_lock:
                self.left_buffer.append((timestamp, cv_image))
        except Exception as e:
            # 只在录制时记录错误，避免启动时的错误刷屏
            if self.recording:
                self.get_logger().error(f"左手相机图像转换失败: {e}")

    def right_callback(self, msg: Image):
        """接收右手腕相机图像"""
        try:
            # 检查图像数据是否为空
            if not msg.data:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 检查转换后的图像是否为空
            if cv_image.size == 0:
                return
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.data_lock:
                self.right_buffer.append((timestamp, cv_image))
        except Exception as e:
            # 只在录制时记录错误，避免启动时的错误刷屏
            if self.recording:
                self.get_logger().error(f"右手相机图像转换失败: {e}")

    def joint_states_callback(self, msg: JointState):
        """接收关节状态"""
        try:
            joint_positions = self._extract_joint_positions(msg)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.data_lock:
                self.joint_states_buffer.append((timestamp, joint_positions))
        except Exception as e:
            self.get_logger().error(f"关节状态处理失败: {e}")

    def joint_command_callback(self, msg: JointState):
        """接收关节控制命令 (作为 action)"""
        try:
            joint_positions = self._extract_joint_positions(msg)
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.data_lock:
                self.joint_command_buffer.append((timestamp, joint_positions))
        except Exception as e:
            self.get_logger().error(f"关节命令处理失败: {e}")

    def hand_ee_callback(self, msg: TFMessage):
        """接收末端执行器位姿 (left_hand_ee_link 和 right_hand_ee_link)"""
        try:
            # 提取左右手末端执行器的位姿
            left_hand_ee = None
            right_hand_ee = None
            timestamp = None

            for transform in msg.transforms:
                if timestamp is None:
                    timestamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9

                if transform.child_frame_id == "left_hand_ee_link":
                    left_hand_ee = self._extract_transform(transform)
                elif transform.child_frame_id == "right_hand_ee_link":
                    right_hand_ee = self._extract_transform(transform)

            if left_hand_ee is not None and right_hand_ee is not None and timestamp is not None:
                # 合并左右手位姿: [left_pos(3), left_axis_angle(3), right_pos(3), right_axis_angle(3)] = 12维
                combined_ee = np.concatenate([left_hand_ee, right_hand_ee])
                with self.data_lock:
                    self.hand_ee_buffer.append((timestamp, combined_ee))
        except Exception as e:
            self.get_logger().error(f"末端执行器处理失败: {e}")

    def arm_ee_callback(self, msg: TFMessage):
        """接收目标末端执行器位姿 (left_hand_ee_link 和 right_hand_ee_link)"""
        try:
            left_arm_ee = None
            right_arm_ee = None
            timestamp = None

            for transform in msg.transforms:
                if timestamp is None:
                    timestamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9

                if transform.child_frame_id == "left_hand_ee_link":
                    left_arm_ee = self._extract_transform(transform)
                elif transform.child_frame_id == "right_hand_ee_link":
                    right_arm_ee = self._extract_transform(transform)

            if left_arm_ee is not None and right_arm_ee is not None and timestamp is not None:
                # 合并左右手目标位姿: [left_pos(3), left_axis_angle(3), right_pos(3), right_axis_angle(3)] = 12维
                combined_ee = np.concatenate([left_arm_ee, right_arm_ee])
                with self.data_lock:
                    self.arm_ee_buffer.append((timestamp, combined_ee))
        except Exception as e:
            self.get_logger().error(f"目标末端执行器处理失败: {e}")

    def _extract_transform(self, transform: TransformStamped) -> np.ndarray:
        """从 TransformStamped 中提取位姿 (位置 + 四元数)"""
        # 提取位置 (x, y, z)
        pos = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ], dtype=np.float32)

        # 提取旋转四元数 (x, y, z, w)
        quat = np.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ], dtype=np.float32)

        # 将四元数转换为轴角表示
        axis_angle = self._quat_to_axis_angle(quat)

        # 返回位置和轴角 (共6维)
        return np.concatenate([pos, axis_angle])

    def _quat_to_axis_angle(self, quat: np.ndarray) -> np.ndarray:
        """将四元数 (x, y, z, w) 转换为轴角表示 (rx, ry, rz)
        
        轴角表示: 旋转轴 * 旋转角度
        返回的向量方向是旋转轴，长度是旋转角度（弧度）
        """
        x, y, z, w = quat
        
        # 计算旋转角度的2倍
        sin_half_angle = np.sqrt(x*x + y*y + z*z)
        
        # 如果旋转角度接近0，返回零向量
        if sin_half_angle < 1e-8:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 计算旋转角度
        angle = 2.0 * np.arctan2(sin_half_angle, w)
        
        # 计算轴角: 旋转轴 * 旋转角度
        # 旋转轴 = (x, y, z) / sin_half_angle
        # 轴角 = 旋转轴 * 角度
        scale = angle / sin_half_angle
        return np.array([x * scale, y * scale, z * scale], dtype=np.float32)

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

    def _find_closest_data(self, buffer, target_time):
        """在缓冲区中找到时间戳最接近目标时间的数据"""
        if not buffer:
            return None

        closest_data = None
        min_diff = float('inf')

        for timestamp, data in buffer:
            diff = abs(timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_data = data

        # 如果时间差超过最大延迟，认为数据无效
        if min_diff > self.max_timestamp_delay:
            return None

        return closest_data

    def record_frame(self):
        """录制一帧数据（基于时间戳对齐）"""
        if not self.recording:
            return

        with self.data_lock:
            # 以头部相机的时间戳为基准
            if not self.head_buffer:
                return

            # 获取最新的头部相机数据作为基准
            head_timestamp, head_image = self.head_buffer[-1]

            # 根据头部相机的时间戳，找到其他 topic 中最接近的数据
            left_image = self._find_closest_data(self.left_buffer, head_timestamp)
            right_image = self._find_closest_data(self.right_buffer, head_timestamp)
            joint_states = self._find_closest_data(self.joint_states_buffer, head_timestamp)
            joint_command = self._find_closest_data(self.joint_command_buffer, head_timestamp)
            hand_ee = self._find_closest_data(self.hand_ee_buffer, head_timestamp)
            arm_ee = self._find_closest_data(self.arm_ee_buffer, head_timestamp)

            # 检查所有数据是否都找到
            if left_image is None or right_image is None or joint_states is None or joint_command is None or hand_ee is None or arm_ee is None:
                self.get_logger().warn(f"数据对齐失败，跳过此帧 (head_timestamp: {head_timestamp:.3f}s)")
                return

            # 调试信息：检查图像数据
            if self.episode_frame_count == 0:
                self.get_logger().info(f"头部图像形状: {head_image.shape}, 数据类型: {head_image.dtype}")
                self.get_logger().info(f"左手图像形状: {left_image.shape}, 数据类型: {left_image.dtype}")
                self.get_logger().info(f"右手图像形状: {right_image.shape}, 数据类型: {right_image.dtype}")
                self.get_logger().info(f"关节状态形状: {joint_states.shape}, 数据类型: {joint_states.dtype}")
                self.get_logger().info(f"关节命令形状: {joint_command.shape}, 数据类型: {joint_command.dtype}")
                self.get_logger().info(f"当前末端执行器形状: {hand_ee.shape}, 数据类型: {hand_ee.dtype}")
                self.get_logger().info(f"目标末端执行器形状: {arm_ee.shape}, 数据类型: {arm_ee.dtype}")
                self.get_logger().info(f"基准时间戳: {head_timestamp:.3f}s")
                
                # 保存第一帧图像用于检查
                import cv2
                import os
                debug_dir = "./debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, "head_0.jpg"), head_image)
                cv2.imwrite(os.path.join(debug_dir, "left_0.jpg"), left_image)
                cv2.imwrite(os.path.join(debug_dir, "right_0.jpg"), right_image)
                self.get_logger().info(f"已保存调试图像到 {debug_dir}")

            # 构建帧数据 (注意：OpenCV 是 BGR 格式，需要转换为 RGB)
            import cv2
            
            # 将末端执行器数据添加到关节状态中
            combined_state = np.concatenate([joint_states, hand_ee])
            
            # 将目标末端执行器数据添加到关节命令中
            combined_action = np.concatenate([joint_command, arm_ee])
            
            frame = {
                "observation.images.head": cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB),
                "observation.images.left": cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB),
                "observation.images.right": cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB),
                "observation.state": combined_state.copy(),
                "action": combined_action.copy(),
                "task": self.current_task["task"],
                # "language_instruction": self.current_task["language_instruction"],
            }

        # 添加帧到数据集
        try:
            self.dataset.add_frame(frame)
            self.episode_frame_count += 1

            # 每 30 帧打印一次日志
            if self.episode_frame_count % 30 == 0:
                self.get_logger().info(f"已录制 {self.episode_frame_count} 帧 (时间戳: {head_timestamp:.3f}s)")

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
                # 强制 flush metadata buffer，确保 self.meta.latest_episode 被更新
                # 这样下一个 episode 的视频不会被覆盖
                self.dataset.meta._flush_metadata_buffer()
                self.get_logger().info("Episode 保存成功!")
            except Exception as e:
                self.get_logger().error(f"保存 episode 失败: {e}")

            self.episode_frame_count = 0


def create_dataset_features():
    """创建数据集特征定义"""
    # 动作特征: 关节位置命令 + 目标末端执行器位姿
    # 注意：action 现在包含末端执行器数据（位置3维 + 轴角3维），所以长度是 len(ALL_JOINTS) + 12
    action_features = {
        joint_name: float for joint_name in ALL_JOINTS
    }
    # 添加目标末端执行器相关的特征名称（使用轴角表示，每手6维：位置3 + 轴角3）
    action_ee_feature_names = [
        # 左手目标末端执行器
        "left_ee_x", "left_ee_y", "left_ee_z",
        "left_ee_rx", "left_ee_ry", "left_ee_rz",
        # 右手目标末端执行器
        "right_ee_x", "right_ee_y", "right_ee_z",
        "right_ee_rx", "right_ee_ry", "right_ee_rz"
    ]
    for name in action_ee_feature_names:
        action_features[name] = float

    # 观察特征: 关节状态 + 相机图像 + 当前末端执行器位姿
    # 注意：observation.state 现在包含末端执行器数据（位置3维 + 轴角3维），所以长度是 len(ALL_JOINTS) + 12
    observation_features = {
        joint_name: float for joint_name in ALL_JOINTS
    }
    # 添加当前末端执行器相关的特征名称（使用轴角表示，每手6维：位置3 + 轴角3）
    obs_ee_feature_names = [
        # 左手当前末端执行器
        "left_ee_x", "left_ee_y", "left_ee_z",
        "left_ee_rx", "left_ee_ry", "left_ee_rz",
        # 右手当前末端执行器
        "right_ee_x", "right_ee_y", "right_ee_z",
        "right_ee_rx", "right_ee_ry", "right_ee_rz"
    ]
    for name in obs_ee_feature_names:
        observation_features[name] = float
    
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


def select_task():
    """让用户选择任务"""
    print("\n" + "=" * 60)
    print("请选择要录制的任务:")
    print("=" * 60)
    for i, task in enumerate(AVAILABLE_TASKS, 1):
        print(f"  {i}. {task['task']}")
        print(f"     指令: {task['language_instruction']}")
    print("=" * 60)

    while True:
        try:
            choice = input(f"请输入任务编号 (1-{len(AVAILABLE_TASKS)}): ").strip()
            task_idx = int(choice) - 1
            if 0 <= task_idx < len(AVAILABLE_TASKS):
                selected_task = AVAILABLE_TASKS[task_idx]
                print(f"\n已选择任务: {selected_task['task']}")
                print(f"语言指令: {selected_task['language_instruction']}")
                return selected_task
            else:
                print(f"请输入 1 到 {len(AVAILABLE_TASKS)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")


def main():
    """主函数"""
    # 先让用户选择任务
    current_task = select_task()

    # 创建数据集
    features = create_dataset_features()

    print("\n" + "=" * 60)
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
    recorder = IsaacSimRecorder(dataset, current_task)

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
