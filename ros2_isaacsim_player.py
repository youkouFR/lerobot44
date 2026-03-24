#!/usr/bin/env python3
"""
ROS2 IsaacSim 数据集播放脚本

这个脚本用于将录制为 LeRobot 格式的数据集回放到 ROS2 topic 中。

功能:
1. 从指定的 LeRobot 数据集中读取数据 (按指定的 episode)。
2. 将录制时被压缩/变换的数据还原:
   - RGB 图像转换回 BGR8 的 ROS Image
   - 将 末端执行器(EE) 的轴角(axis-angle) 变换回 四元数 (quaternion, [x, y, z, w])
3. 重新发布到原始的 Topic (/Head, /left, /right, /joint_states, /joint_command, /hand_ee, /arm_ee)。

使用方法:
    python ros2_isaacsim_player.py --dataset-path ./isaacsim_dataset --episode 0 --fps 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

# 添加 src 到 Python 路径以便导入 lerobot
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# 关节名称 (与 ros2_isaacsim_recorder.py 保持一致)
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
ARM_EE_TOPIC = "/arm_ee"


class IsaacSimPlayer(Node):
    """IsaacSim 数据回放节点"""

    def __init__(self, dataset: LeRobotDataset, fps: int):
        super().__init__("isaacsim_player")
        self.dataset = dataset
        self.fps = fps
        self.frame_idx = 0
        self.bridge = CvBridge()
        
        self.total_frames = len(self.dataset)

        # 创建发布者
        self.head_pub = self.create_publisher(Image, HEAD_CAMERA_TOPIC, 10)
        self.left_pub = self.create_publisher(Image, LEFT_CAMERA_TOPIC, 10)
        self.right_pub = self.create_publisher(Image, RIGHT_CAMERA_TOPIC, 10)
        self.joint_states_pub = self.create_publisher(JointState, JOINT_STATES_TOPIC, 10)
        self.joint_command_pub = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        self.hand_ee_pub = self.create_publisher(TFMessage, HAND_EE_TOPIC, 10)
        self.arm_ee_pub = self.create_publisher(TFMessage, ARM_EE_TOPIC, 10)

        # 创建播放定时器
        self.timer_period = 1.0 / self.fps
        self.timer = self.create_timer(self.timer_period, self.play_frame)

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"开始播放数据集，共 {self.total_frames} 帧，目标帧率: {self.fps} FPS")
        self.get_logger().info("=" * 60)

    def _axis_angle_to_quat(self, axis_angle: np.ndarray) -> np.ndarray:
        """将轴角表示 (rx, ry, rz) 转换回四元数 (x, y, z, w)"""
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        axis = axis_angle / angle
        sin_half_angle = np.sin(angle / 2.0)
        x = axis[0] * sin_half_angle
        y = axis[1] * sin_half_angle
        z = axis[2] * sin_half_angle
        w = np.cos(angle / 2.0)
        
        return np.array([x, y, z, w], dtype=np.float32)

    def _create_tf_message(self, left_data: np.ndarray, right_data: np.ndarray, frame_id: str = "base_link") -> TFMessage:
        """根据位姿+轴角数组，组装左右手的 TFMessage"""
        timestamp = self.get_clock().now().to_msg()
        
        # 提取左手数据
        left_pos = left_data[0:3]
        left_axis_angle = left_data[3:6]
        left_quat = self._axis_angle_to_quat(left_axis_angle)
        
        left_tf = TransformStamped()
        left_tf.header.stamp = timestamp
        left_tf.header.frame_id = frame_id
        left_tf.child_frame_id = "left_hand_ee_link"
        left_tf.transform.translation.x = float(left_pos[0])
        left_tf.transform.translation.y = float(left_pos[1])
        left_tf.transform.translation.z = float(left_pos[2])
        left_tf.transform.rotation.x = float(left_quat[0])
        left_tf.transform.rotation.y = float(left_quat[1])
        left_tf.transform.rotation.z = float(left_quat[2])
        left_tf.transform.rotation.w = float(left_quat[3])

        # 提取右手数据
        right_pos = right_data[0:3]
        right_axis_angle = right_data[3:6]
        right_quat = self._axis_angle_to_quat(right_axis_angle)
        
        right_tf = TransformStamped()
        right_tf.header.stamp = timestamp
        right_tf.header.frame_id = frame_id
        right_tf.child_frame_id = "right_hand_ee_link"
        right_tf.transform.translation.x = float(right_pos[0])
        right_tf.transform.translation.y = float(right_pos[1])
        right_tf.transform.translation.z = float(right_pos[2])
        right_tf.transform.rotation.x = float(right_quat[0])
        right_tf.transform.rotation.y = float(right_quat[1])
        right_tf.transform.rotation.z = float(right_quat[2])
        right_tf.transform.rotation.w = float(right_quat[3])

        tf_msg = TFMessage()
        tf_msg.transforms = [left_tf, right_tf]
        return tf_msg

    def _process_image_to_msg(self, img_data) -> Image:
        """将 LeRobot 图像转换回 ROS2 的 bgr8 格式 Image"""
        import torch
        if isinstance(img_data, torch.Tensor):
            img = img_data.numpy()
        else:
            img = np.array(img_data)
            
        # LeRobot 通常会保存为 (C, H, W) 或 (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
            
        # 处理 float32 的 0-1 格式
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            
        # 原始是从 bgr8 转换成了 RGB，在此转回去
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 转换为 Image Message
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        return msg

    def play_frame(self):
        """定时器回调，发布一帧数据"""
        if self.frame_idx >= self.total_frames:
            self.get_logger().info("播放结束。")
            self.timer.cancel()
            rclpy.shutdown()
            return

        try:
            frame = self.dataset[self.frame_idx]
            
            # 提取数据并转化为 numpy array 方便处理
            import torch
            state_data = frame["observation.state"].numpy() if isinstance(frame["observation.state"], torch.Tensor) else np.array(frame["observation.state"])
            action_data = frame["action"].numpy() if isinstance(frame["action"], torch.Tensor) else np.array(frame["action"])
            
            timestamp = self.get_clock().now().to_msg()

            # 1. 恢复并发布关节状态 (/joint_states)
            js_msg = JointState()
            js_msg.header.stamp = timestamp
            js_msg.name = ALL_JOINTS
            js_msg.position = state_data[0:NUM_JOINTS].tolist()
            self.joint_states_pub.publish(js_msg)

            # 2. 恢复并发布关节命令 (/joint_command)
            jc_msg = JointState()
            jc_msg.header.stamp = timestamp
            jc_msg.name = ALL_JOINTS
            jc_msg.position = action_data[0:NUM_JOINTS].tolist()
            self.joint_command_pub.publish(jc_msg)

            # 3. 恢复并发布当前末端执行器 (/hand_ee) (从 state 恢复)
            hand_ee_data = state_data[NUM_JOINTS:NUM_JOINTS+12]
            hand_ee_msg = self._create_tf_message(left_data=hand_ee_data[0:6], right_data=hand_ee_data[6:12])
            self.hand_ee_pub.publish(hand_ee_msg)

            # 4. 恢复并发布目标末端执行器 (/arm_ee) (从 action 恢复)
            arm_ee_data = action_data[NUM_JOINTS:NUM_JOINTS+12]
            arm_ee_msg = self._create_tf_message(left_data=arm_ee_data[0:6], right_data=arm_ee_data[6:12])
            self.arm_ee_pub.publish(arm_ee_msg)

            # 5. 恢复并发布图像
            if "observation.images.head" in frame:
                self.head_pub.publish(self._process_image_to_msg(frame["observation.images.head"]))
            if "observation.images.left" in frame:
                self.left_pub.publish(self._process_image_to_msg(frame["observation.images.left"]))
            if "observation.images.right" in frame:
                self.right_pub.publish(self._process_image_to_msg(frame["observation.images.right"]))

            # 进度报告
            if self.frame_idx % 30 == 0:
                self.get_logger().info(f"正在播放帧 {self.frame_idx}/{self.total_frames}...")

            self.frame_idx += 1

        except Exception as e:
            self.get_logger().error(f"播放第 {self.frame_idx} 帧时出错: {e}")
            import traceback
            traceback.print_exc()
            self.timer.cancel()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="播放录制的 LeRobot 数据集到 ROS2 topic")
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
        help="要播放的 episode 索引 (默认: 0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="播放帧率 (默认: 30)"
    )
    args = parser.parse_args()

    print("加载数据集中...")
    try:
        # 指定 episodes 后，读取出来的数据集长度会自动仅包含该 episode 的数据帧
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.dataset_path,
            episodes=[args.episode]
        )
    except Exception as e:
        print(f"无法加载数据集: {e}")
        sys.exit(1)

    rclpy.init()
    player = IsaacSimPlayer(dataset, args.fps)
    
    try:
        rclpy.spin(player)
    except KeyboardInterrupt:
        print("\n用户中断播放...")
    finally:
        player.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()