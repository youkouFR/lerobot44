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
import shutil
from pathlib import Path
from datetime import datetime
import threading
import signal
import sys
import select
from collections import deque

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR


# ==================== 配置参数 ====================
# 数据集配置
DATASET_BASE_ROOT = "./isaacsim"  # 数据集本地保存根路径
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

    def __init__(self, current_task: dict = None, dataset: LeRobotDataset = None):
        super().__init__("isaacsim_recorder")

        self.dataset = dataset
        self.current_task = current_task or {"task": "isaacsim_vla"}
        self.bridge = CvBridge()

        # 数据集管理（实例级，供 lerobot_command 和 main() 键盘控制公用）
        self.current_dataset = dataset  # 当前活动的 LeRobotDataset
        self.current_dataset_root = None  # 当前数据集根路径
        self.should_exit = False  # record_exit 命令标志

        # 带时间戳的数据缓冲区
        self.head_buffer = deque(maxlen=30)
        self.left_buffer = deque(maxlen=30)
        self.right_buffer = deque(maxlen=30)
        self.joint_states_buffer = deque(maxlen=30)
        self.joint_command_buffer = deque(maxlen=30)
        self.hand_ee_buffer = deque(maxlen=30)  # 当前末端执行器位姿（observation）
        self.arm_ee_buffer = deque(maxlen=30)   # 目标末端执行器位姿（action）

        # 数据锁
        self.data_lock = threading.Lock()
        self.recording = False
        self.episode_frame_count = 0

        # 记录每个 buffer 上一次录制的数据时间戳，用于去重
        # key: buffer 名称, value: 上次录制的时间戳
        self._last_recorded_ts = {
            "head": -1.0,
            "left": -1.0,
            "right": -1.0,
            "joint_states": -1.0,
            "joint_command": -1.0,
            "hand_ee": -1.0,
            "arm_ee": -1.0,
        }

        # 必需的 buffer（必须全部有数据才能录制）
        self._required_buffers = {"head", "left", "right", "joint_states", "joint_command"}
        # 可选的 buffer（有数据就用，没有就填零）
        self._optional_buffers = {"hand_ee", "arm_ee"}
        # 可选数据的维度（首次收到后记录，用于缺失时填零）
        self._optional_data_dim = {
            "hand_ee": 12,   # left(6) + right(6)
            "arm_ee": 12,    # left(6) + right(6)
        }

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
        self.get_logger().info("按 's-<name>' 创建新数据集并开始录制, 'q' 停止录制并保存")
        self.get_logger().info("或通过 /lerobot_command 发送 's-<name>'/'record_stop' 控制")

    def set_dataset(self, dataset: LeRobotDataset):
        """设置/切换当前数据集"""
        self.dataset = dataset

    def lerobot_command_callback(self, msg: String):
        """接收录制控制命令

        支持的命令:
          record_start   - 开始录制
          record_stop    - 停止录制并 finalize 当前数据集
          record_discard - 停止录制并删除当前数据集（用于失败放弃）
          s-<name>       - 创建 isaacsim/<name>/ 数据集并开始录制 (如 s-step10, s-step12)
          record_exit    - 停止录制并退出程序
        """
        command = msg.data.strip().lower()
        self.get_logger().info(f"收到录制命令: {command}")

        if command == "record_start":
            self.start_recording()
        elif command == "record_stop":
            ds = self.stop_recording()
            if ds is not None:
                self.get_logger().info("完成数据集...")
                ds.finalize()
                self.get_logger().info(f"数据集已保存到: {self.current_dataset_root}")
                self.current_dataset = None
                self.current_dataset_root = None
        elif command == "record_discard":
            # 停止录制并将失败数据集移到 fail 目录（保留用于问题排查）
            ds = self.stop_recording()
            if ds is not None and self.current_dataset_root is not None:
                discard_path = Path(self.current_dataset_root)
                dataset_name = discard_path.name  # e.g. robot_recording_20260711_203030
                fail_dir = Path(DATASET_BASE_ROOT) / "fail"
                fail_dir.mkdir(parents=True, exist_ok=True)
                target_path = fail_dir / dataset_name
                self.get_logger().info(f"步骤失败，数据集移至: {target_path}")
                try:
                    shutil.move(str(discard_path), str(target_path))
                    self.get_logger().info(f"数据集已移至: {target_path}")
                except Exception as e:
                    self.get_logger().error(f"移动数据集失败: {e}")
                self.current_dataset = None
                self.current_dataset_root = None
        elif command.startswith("s-"):
            sub_dir = command[2:]  # "s-step10" → "step10"
            # 如果正在录制，先停止并完成上一个数据集
            if self.recording:
                old_ds = self.stop_recording()
                if old_ds is not None:
                    self.get_logger().info("完成上一个数据集...")
                    old_ds.finalize()
                    self.get_logger().info(f"上一个数据集已保存到: {self.current_dataset_root}")
            # 创建新数据集并开始录制
            self.current_dataset, self.current_dataset_root = _create_new_dataset(sub_dir)
            self.set_dataset(self.current_dataset)
            self.start_recording()
        elif command == "record_exit":
            self.get_logger().info("收到退出命令，正在关闭...")
            if self.recording:
                ds = self.stop_recording()
                if ds is not None:
                    self.get_logger().info("完成数据集...")
                    ds.finalize()
                    self.get_logger().info(f"数据集已保存到: {self.current_dataset_root}")
            self.should_exit = True
        else:
            self.get_logger().warn(f"未知命令: {command}，支持: record_start, record_stop, record_discard, s-<name>, record_exit")

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

    def _get_empty_buffers(self, required_only=True):
        """返回当前为空的 buffer 名称列表

        Args:
            required_only: True 只检查必需的 buffer，False 检查全部
        """
        empty = []
        buffers_to_check = self._required_buffers if required_only else (
            self._required_buffers | self._optional_buffers
        )
        if "head" in buffers_to_check and not self.head_buffer:
            empty.append("head")
        if "left" in buffers_to_check and not self.left_buffer:
            empty.append("left")
        if "right" in buffers_to_check and not self.right_buffer:
            empty.append("right")
        if "joint_states" in buffers_to_check and not self.joint_states_buffer:
            empty.append("joint_states")
        if "joint_command" in buffers_to_check and not self.joint_command_buffer:
            empty.append("joint_command")
        if "hand_ee" in buffers_to_check and not self.hand_ee_buffer:
            empty.append("hand_ee")
        if "arm_ee" in buffers_to_check and not self.arm_ee_buffer:
            empty.append("arm_ee")
        return empty

    def _get_optional_data(self, buffer, dim):
        """从可选 buffer 获取数据，buffer 为空时返回零向量"""
        if buffer:
            return buffer[-1][1]  # (timestamp, data) -> data
        else:
            return np.zeros(dim, dtype=np.float32)

    def _all_buffers_ready(self):
        """检查所有必需的 buffer 是否都有数据"""
        return len(self._get_empty_buffers(required_only=True)) == 0

    def record_frame(self):
        """录制一帧数据（使用各 buffer 最新数据，不做时间戳对齐）

        策略：
        - 必需的 topic (head/left/right/joint_states/joint_command): 全部就绪才录制
        - 可选的 topic (hand_ee/arm_ee): 有数据就用，没有就填零
        - 至少一个 topic 产生新数据才录制（去重）
        """
        if not self.recording:
            return

        if self.dataset is None:
            return

        with self.data_lock:
            # 检查必需的 buffer 是否就绪
            empty_required = self._get_empty_buffers(required_only=True)
            if empty_required:
                now = self.get_clock().now().nanoseconds / 1e9
                if not hasattr(self, '_last_empty_log_time'):
                    self._last_empty_log_time = 0.0
                if now - self._last_empty_log_time > 1.0:
                    self.get_logger().warn(
                        f"等待必需 topic 数据... 缺失: {', '.join(empty_required)}"
                    )
                    self._last_empty_log_time = now
                return

            # 从必需 buffer 取最新数据
            head_ts, head_image = self.head_buffer[-1]
            left_ts, left_image = self.left_buffer[-1]
            right_ts, right_image = self.right_buffer[-1]
            js_ts, joint_states = self.joint_states_buffer[-1]
            jc_ts, joint_command = self.joint_command_buffer[-1]

            # 从可选 buffer 取数据（缺失时填零）
            hand_ee = self._get_optional_data(self.hand_ee_buffer, self._optional_data_dim["hand_ee"])
            arm_ee = self._get_optional_data(self.arm_ee_buffer, self._optional_data_dim["arm_ee"])

            # 检查可选 buffer 是否首次变为可用（或仍缺失）
            missing_optional = [name for name in self._optional_buffers
                               if not getattr(self, f"{name}_buffer")]
            if missing_optional:
                now = self.get_clock().now().nanoseconds / 1e9
                if not hasattr(self, '_last_optional_log_time'):
                    self._last_optional_log_time = 0.0
                if now - self._last_optional_log_time > 5.0:
                    self.get_logger().warn(
                        f"可选 topic 仍无数据，将用零填充: {', '.join(missing_optional)}"
                    )
                    self._last_optional_log_time = now

            # 检查是否至少有一个 topic 产生了新数据（防止重复录制同一帧）
            current_ts = {
                "head": head_ts, "left": left_ts, "right": right_ts,
                "joint_states": js_ts, "joint_command": jc_ts,
            }
            # 可选 topic 的时间戳也参与去重（如果有数据的话）
            if self.hand_ee_buffer:
                current_ts["hand_ee"] = self.hand_ee_buffer[-1][0]
            if self.arm_ee_buffer:
                current_ts["arm_ee"] = self.arm_ee_buffer[-1][0]

            has_new_data = any(
                current_ts.get(key, -1.0) > self._last_recorded_ts.get(key, -1.0)
                for key in current_ts
            )
            if not has_new_data:
                return

            # 更新已录制的时间戳
            for key in current_ts:
                self._last_recorded_ts[key] = current_ts[key]

            # 检查图像是否为空
            if head_image.size == 0 or left_image.size == 0 or right_image.size == 0:
                self.get_logger().warn(f"图像数据为空，跳过此帧")
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
                self.get_logger().info(f"已录制 {self.episode_frame_count} 帧")

        except Exception as e:
            self.get_logger().error(f"添加帧失败: {e}")

    def start_recording(self):
        """开始录制"""
        if not self.recording:
            self.recording = True
            self.episode_frame_count = 0
            # 重置时间戳追踪，确保新 episode 从第一帧开始录制
            for key in self._last_recorded_ts:
                self._last_recorded_ts[key] = -1.0
            self._last_empty_log_time = 0.0

            # 立即报告各 topic 数据就绪状态
            empty_required = self._get_empty_buffers(required_only=True)
            empty_optional = [name for name in self._optional_buffers
                            if not getattr(self, f"{name}_buffer")]
            if empty_required:
                self.get_logger().warn(
                    f"必需 topic 尚无数据，录制将等待: {', '.join(empty_required)}"
                )
            else:
                self.get_logger().info("所有必需 topic 数据就绪，开始录制")
            if empty_optional:
                self.get_logger().warn(
                    f"可选 topic 尚无数据，将用零填充: {', '.join(empty_optional)}"
                )
            self.get_logger().info("=" * 60)
            self.get_logger().info("开始录制!")
            self.get_logger().info("=" * 60)

    def stop_recording(self):
        """停止录制并保存，返回当前 dataset（用于后续 finalize）"""
        if self.recording:
            self.recording = False
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"停止录制，共录制 {self.episode_frame_count} 帧")
            self.get_logger().info("保存 episode...")

            dataset_to_finalize = self.dataset
            try:
                if self.dataset is not None:
                    self.dataset.save_episode()
                    # 强制 flush metadata buffer，确保 self.meta.latest_episode 被更新
                    # 这样下一个 episode 的视频不会被覆盖
                    self.dataset.meta._flush_metadata_buffer()
                    self.get_logger().info("Episode 保存成功!")
            except Exception as e:
                self.get_logger().error(f"保存 episode 失败: {e}")

            self.episode_frame_count = 0
            return dataset_to_finalize
        return None


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


def _create_new_dataset(sub_dir: str) -> tuple:
    """创建新的 LeRobot 数据集

    Args:
        sub_dir: 子目录名 (如 "1", "2", "3")

    Returns:
        (dataset, dataset_root_path) 元组
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_dir_name = f"robot_recording_{timestamp}"
    repo_id = f"isaacsim/{sub_dir}/{dataset_dir_name}"
    root = str(Path(DATASET_BASE_ROOT) / sub_dir / dataset_dir_name)

    print(f"\n创建新数据集:")
    print(f"  路径: {root}")
    print(f"  ID: {repo_id}")

    features = create_dataset_features()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=FPS,
        features=features,
        robot_type="isaacsim_dual_arm",
        use_videos=True,
        image_writer_threads=4,
    )
    dataset.vcodec = "h264"
    return dataset, root


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("IsaacSim 数据录制器")
    print(f"数据集根路径: {DATASET_BASE_ROOT}")
    print(f"帧率: {FPS} FPS")
    print("=" * 60)

    # 初始化 ROS2（不再需要手动选任务，通过 /lerobot_command 控制录制）
    rclpy.init()
    recorder = IsaacSimRecorder(current_task=None, dataset=None)

    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n接收到中断信号，正在退出...")
        ds = recorder.stop_recording()
        if ds is not None:
            print("完成数据集...")
            ds.finalize()
            print(f"数据集已保存到: {recorder.current_dataset_root}")
        recorder.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 简单的键盘控制
    print("\n控制命令:")
    print("  s-<name> - 创建新数据集并开始录制 (如 s-step10, s-step12)")
    print("            数据集保存路径: isaacsim/<name>/robot_recording_YYYYmmdd_HHMMSS")
    print("  q        - 停止录制并保存当前数据集")
    print("  x        - 退出程序")
    print("=" * 60)
    print("提示: 也可通过 /lerobot_command topic 发送命令 (s-<name>/record_discard/record_exit)")
    print("=" * 60)

    # 在单独的线程中运行 ROS2 spin
    spin_thread = threading.Thread(target=rclpy.spin, args=(recorder,))
    spin_thread.start()

    try:
        while True:
            # 检查 /lerobot_command 是否请求退出
            if recorder.should_exit:
                print("收到 /lerobot_command record_exit 命令，正在退出...")
                break

            # 使用 select 实现非阻塞 stdin 检查（1秒超时），
            # 确保 should_exit 标志能被及时检测到
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                cmd = sys.stdin.readline().strip().lower()
            else:
                continue  # 超时，重新检查 should_exit

            if cmd.startswith("s-"):
                sub_dir = cmd[2:]  # "s-step10" → "step10"
                # 如果正在录制，先停止并完成上一个数据集
                if recorder.recording:
                    old_dataset = recorder.stop_recording()
                    if old_dataset is not None:
                        print("完成上一个数据集...")
                        old_dataset.finalize()
                        print(f"上一个数据集已保存到: {recorder.current_dataset_root}")

                # 创建新数据集并开始录制
                recorder.current_dataset, recorder.current_dataset_root = _create_new_dataset(sub_dir)
                recorder.set_dataset(recorder.current_dataset)
                recorder.start_recording()

            elif cmd == 'q':
                ds = recorder.stop_recording()
                if ds is not None:
                    print("完成数据集...")
                    ds.finalize()
                    print(f"数据集已保存到: {recorder.current_dataset_root}")
                    recorder.current_dataset = None
                    recorder.current_dataset_root = None
                else:
                    print("未在录制中")

            elif cmd == 'x':
                break
            else:
                print("未知命令，使用: s-<name>(创建数据集并录制), q(停止), x(退出)")

    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        ds = recorder.stop_recording()
        if ds is not None:
            print("完成数据集...")
            ds.finalize()
            print(f"数据集已保存到: {recorder.current_dataset_root}")
        recorder.destroy_node()
        rclpy.shutdown()
        spin_thread.join()

        print("=" * 60)
        print("录制器已退出")
        print("=" * 60)


if __name__ == "__main__":
    main()
