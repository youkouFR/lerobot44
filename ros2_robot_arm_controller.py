#!/usr/bin/env python3
"""
ROS2 机器人手臂控制器

这个脚本用于向 Isaac Sim 中的机器人发布关节控制命令，模拟 hello_world_isaaclab.py 中的手臂运动模式。
它会创建一个 ROS2 节点，并向 /joint_command topic 发布 JointState 消息。

使用方法：
    python ros2_robot_arm_controller.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import time
import math
import numpy as np

class RobotArmController(Node):
    """机器人手臂控制器节点"""
    
    def __init__(self):
        super().__init__("robot_arm_controller")
        # 创建发布者，发布到 /joint_command topic
        self.publisher_ = self.create_publisher(JointState, "/joint_command", 10)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz 发布频率
        
        # 订阅 /clock topic 获取模拟时钟
        self.clock_sub = self.create_subscription(
            Clock, "/clock", self.clock_callback, 10
        )
        
        # 模拟时钟时间
        self.sim_time = 0.0
        
        # 关节名称（从 hello_world_isaaclab.py 中获取）
        self.l_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
            "left_hand_index_bend_joint","left_hand_index_joint1","left_hand_index_joint2","left_hand_mid_joint1",
            "left_hand_mid_joint2","left_hand_pinky_joint1","left_hand_pinky_joint2","left_hand_ring_joint1",
            "left_hand_ring_joint2","left_hand_thumb_bend_joint","left_hand_thumb_rota_joint1","left_hand_thumb_rota_joint2"
        ]
        self.r_joints = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
            "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
            "right_hand_index_bend_joint","right_hand_index_joint1","right_hand_index_joint2","right_hand_mid_joint1",
            "right_hand_mid_joint2","right_hand_pinky_joint1","right_hand_pinky_joint2","right_hand_ring_joint1",
            "right_hand_ring_joint2","right_hand_thumb_bend_joint","right_hand_thumb_rota_joint1","right_hand_thumb_rota_joint2"
        ]
        
        # 合并左右手臂关节
        self.joint_names = self.l_joints + self.r_joints
        
        # 手臂目标姿态（从 hello_world_isaaclab.py 中获取）
        # 前7个是手臂关节，后面是手部关节
        self.l_arm_targets = np.array([
            -1.3, -0.2, 1.0, -1.5, 0.0, -1.0, 0.0,  # 手臂关节
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 手指关节
            0.0, 0.0, 0.0  # 拇指关节
        ])
        self.r_arm_targets = np.array([
            -0.5, -0.2, -1.0, -1.5, 0.0, -1.0, 0.0,  # 手臂关节
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 手指关节
            0.0, 0.0, 0.0  # 拇指关节
        ])
        
        # 运动参数 - 为每个关节设计独立的运动模式
        self.freqs = {
            # 左臂关节频率
            "left_shoulder_pitch_joint": 1.0,    # 肩部俯仰
            "left_shoulder_roll_joint": 1.2,      # 肩部翻滚
            "left_arm_yaw_joint": 0.8,           # 手臂偏航
            "left_elbow_pitch_joint": 1.5,       # 肘部俯仰
            "left_elbow_yaw_joint": 0.6,         # 肘部偏航
            "left_wrist_pitch_joint": 1.8,       # 腕部俯仰
            "left_wrist_roll_joint": 0.9,        # 腕部翻滚
            # 左手手指关节频率
            "left_hand_index_bend_joint": 2.0,   # 食指弯曲
            "left_hand_index_joint1": 2.2,       # 食指第一节
            "left_hand_index_joint2": 2.4,       # 食指第二节
            "left_hand_mid_joint1": 2.1,         # 中指第一节
            "left_hand_mid_joint2": 2.3,         # 中指第二节
            "left_hand_pinky_joint1": 1.9,       # 小指第一节
            "left_hand_pinky_joint2": 2.1,       # 小指第二节
            "left_hand_ring_joint1": 2.0,        # 无名指第一节
            "left_hand_ring_joint2": 2.2,        # 无名指第二节
            "left_hand_thumb_bend_joint": 2.3,   # 拇指弯曲
            "left_hand_thumb_rota_joint1": 2.5,  # 拇指旋转1
            "left_hand_thumb_rota_joint2": 2.7,  # 拇指旋转2
            # 右臂关节频率
            "right_shoulder_pitch_joint": 1.1,   # 肩部俯仰
            "right_shoulder_roll_joint": 1.3,    # 肩部翻滚
            "right_arm_yaw_joint": 0.7,          # 手臂偏航
            "right_elbow_pitch_joint": 1.4,      # 肘部俯仰
            "right_elbow_yaw_joint": 0.5,        # 肘部偏航
            "right_wrist_pitch_joint": 1.7,      # 腕部俯仰
            "right_wrist_roll_joint": 0.8,       # 腕部翻滚
            # 右手手指关节频率
            "right_hand_index_bend_joint": 2.0,  # 食指弯曲
            "right_hand_index_joint1": 2.2,      # 食指第一节
            "right_hand_index_joint2": 2.4,      # 食指第二节
            "right_hand_mid_joint1": 2.1,        # 中指第一节
            "right_hand_mid_joint2": 2.3,        # 中指第二节
            "right_hand_pinky_joint1": 1.9,      # 小指第一节
            "right_hand_pinky_joint2": 2.1,      # 小指第二节
            "right_hand_ring_joint1": 2.0,       # 无名指第一节
            "right_hand_ring_joint2": 2.2,       # 无名指第二节
            "right_hand_thumb_bend_joint": 2.3,  # 拇指弯曲
            "right_hand_thumb_rota_joint1": 2.5, # 拇指旋转1
            "right_hand_thumb_rota_joint2": 2.7, # 拇指旋转2
        }
        
        self.amplitudes = {
            # 左臂关节振幅
            "left_shoulder_pitch_joint": 0.5,    # 肩部俯仰
            "left_shoulder_roll_joint": 0.3,     # 肩部翻滚
            "left_arm_yaw_joint": 0.6,           # 手臂偏航
            "left_elbow_pitch_joint": 0.4,       # 肘部俯仰
            "left_elbow_yaw_joint": 0.3,         # 肘部偏航
            "left_wrist_pitch_joint": 0.4,       # 腕部俯仰
            "left_wrist_roll_joint": 0.3,        # 腕部翻滚
            # 左手手指关节振幅
            "left_hand_index_bend_joint": 0.5,   # 食指弯曲
            "left_hand_index_joint1": 0.4,       # 食指第一节
            "left_hand_index_joint2": 0.4,       # 食指第二节
            "left_hand_mid_joint1": 0.4,         # 中指第一节
            "left_hand_mid_joint2": 0.4,         # 中指第二节
            "left_hand_pinky_joint1": 0.3,       # 小指第一节
            "left_hand_pinky_joint2": 0.3,       # 小指第二节
            "left_hand_ring_joint1": 0.35,       # 无名指第一节
            "left_hand_ring_joint2": 0.35,       # 无名指第二节
            "left_hand_thumb_bend_joint": 0.5,   # 拇指弯曲
            "left_hand_thumb_rota_joint1": 0.4,  # 拇指旋转1
            "left_hand_thumb_rota_joint2": 0.4,  # 拇指旋转2
            # 右臂关节振幅
            "right_shoulder_pitch_joint": 0.5,   # 肩部俯仰
            "right_shoulder_roll_joint": 0.3,    # 肩部翻滚
            "right_arm_yaw_joint": 0.6,          # 手臂偏航
            "right_elbow_pitch_joint": 0.4,      # 肘部俯仰
            "right_elbow_yaw_joint": 0.3,        # 肘部偏航
            "right_wrist_pitch_joint": 0.4,      # 腕部俯仰
            "right_wrist_roll_joint": 0.3,       # 腕部翻滚
            # 右手手指关节振幅
            "right_hand_index_bend_joint": 0.5,  # 食指弯曲
            "right_hand_index_joint1": 0.4,      # 食指第一节
            "right_hand_index_joint2": 0.4,      # 食指第二节
            "right_hand_mid_joint1": 0.4,        # 中指第一节
            "right_hand_mid_joint2": 0.4,        # 中指第二节
            "right_hand_pinky_joint1": 0.3,      # 小指第一节
            "right_hand_pinky_joint2": 0.3,      # 小指第二节
            "right_hand_ring_joint1": 0.35,      # 无名指第一节
            "right_hand_ring_joint2": 0.35,      # 无名指第二节
            "right_hand_thumb_bend_joint": 0.5,  # 拇指弯曲
            "right_hand_thumb_rota_joint1": 0.4, # 拇指旋转1
            "right_hand_thumb_rota_joint2": 0.4, # 拇指旋转2
        }
        
        # 相位偏移 - 避免所有关节同时运动
        self.phases = {
            # 左臂关节相位
            "left_shoulder_pitch_joint": 0.0,    # 肩部俯仰
            "left_shoulder_roll_joint": 0.2,     # 肩部翻滚
            "left_arm_yaw_joint": 0.4,           # 手臂偏航
            "left_elbow_pitch_joint": 0.6,       # 肘部俯仰
            "left_elbow_yaw_joint": 0.8,         # 肘部偏航
            "left_wrist_pitch_joint": 1.0,       # 腕部俯仰
            "left_wrist_roll_joint": 1.2,        # 腕部翻滚
            # 左手手指关节相位
            "left_hand_index_bend_joint": 1.4,   # 食指弯曲
            "left_hand_index_joint1": 1.6,       # 食指第一节
            "left_hand_index_joint2": 1.8,       # 食指第二节
            "left_hand_mid_joint1": 2.0,         # 中指第一节
            "left_hand_mid_joint2": 2.2,         # 中指第二节
            "left_hand_pinky_joint1": 2.4,       # 小指第一节
            "left_hand_pinky_joint2": 2.6,       # 小指第二节
            "left_hand_ring_joint1": 2.8,        # 无名指第一节
            "left_hand_ring_joint2": 3.0,        # 无名指第二节
            "left_hand_thumb_bend_joint": 3.2,   # 拇指弯曲
            "left_hand_thumb_rota_joint1": 3.4,  # 拇指旋转1
            "left_hand_thumb_rota_joint2": 3.6,  # 拇指旋转2
            # 右臂关节相位
            "right_shoulder_pitch_joint": 0.1,   # 肩部俯仰
            "right_shoulder_roll_joint": 0.3,    # 肩部翻滚
            "right_arm_yaw_joint": 0.5,          # 手臂偏航
            "right_elbow_pitch_joint": 0.7,      # 肘部俯仰
            "right_elbow_yaw_joint": 0.9,        # 肘部偏航
            "right_wrist_pitch_joint": 1.1,      # 腕部俯仰
            "right_wrist_roll_joint": 1.3,       # 腕部翻滚
            # 右手手指关节相位
            "right_hand_index_bend_joint": 1.5,  # 食指弯曲
            "right_hand_index_joint1": 1.7,      # 食指第一节
            "right_hand_index_joint2": 1.9,      # 食指第二节
            "right_hand_mid_joint1": 2.1,        # 中指第一节
            "right_hand_mid_joint2": 2.3,        # 中指第二节
            "right_hand_pinky_joint1": 2.5,      # 小指第一节
            "right_hand_pinky_joint2": 2.7,      # 小指第二节
            "right_hand_ring_joint1": 2.9,       # 无名指第一节
            "right_hand_ring_joint2": 3.1,       # 无名指第二节
            "right_hand_thumb_bend_joint": 3.3,  # 拇指弯曲
            "right_hand_thumb_rota_joint1": 3.5, # 拇指旋转1
            "right_hand_thumb_rota_joint2": 3.7, # 拇指旋转2
        }
        
        self.get_logger().info(f"机器人手臂控制器已启动")
        self.get_logger().info(f"控制关节数量: {len(self.joint_names)}")
        self.get_logger().info(f"左臂关节: {self.l_joints}")
        self.get_logger().info(f"右臂关节: {self.r_joints}")
        self.get_logger().info(f"已订阅 /clock topic 获取模拟时钟")
    
    def clock_callback(self, msg: Clock):
        """接收模拟时钟"""
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
    
    def timer_callback(self):
        """定时发布关节命令"""
        msg = JointState()
        # 使用模拟时钟作为时间戳
        from builtin_interfaces.msg import Time
        time_msg = Time()
        time_msg.sec = int(self.sim_time)
        time_msg.nanosec = int((self.sim_time - int(self.sim_time)) * 1e9)
        msg.header.stamp = time_msg
        msg.name = self.joint_names
        
        # 生成关节位置命令
        msg.position = []
        msg.velocity = []
        msg.effort = []
        
        # 为每个关节生成独立的运动
        for i, joint_name in enumerate(self.joint_names):
            # 确定关节属于左臂还是右臂
            if joint_name in self.l_joints:
                base_position = self.l_arm_targets[self.l_joints.index(joint_name)]
            else:
                base_position = self.r_arm_targets[self.r_joints.index(joint_name)]
            
            # 获取关节的运动参数
            freq = self.freqs[joint_name]
            amp = self.amplitudes[joint_name]
            phase = self.phases[joint_name]
            
            # 计算关节位置（基础位置 + 正弦运动）
            position = base_position + amp * math.sin(2.0 * math.pi * freq * self.sim_time + phase)
            
            msg.position.append(position)
            msg.velocity.append(0.0)  # 速度设置为 0
            msg.effort.append(0.0)    # 力/力矩设置为 0
        
        self.publisher_.publish(msg)
        
        # 每 1 秒打印一次主要关节的位置
        if int(self.sim_time) != int(self.sim_time + 0.01):
            self.get_logger().info(f"发布关节命令 - 时间: {self.sim_time:.2f}s")
            self.get_logger().info(f"左臂肩部: {msg.position[0]:.4f} rad")
            self.get_logger().info(f"左臂肘部: {msg.position[3]:.4f} rad")
            self.get_logger().info(f"左臂腕部: {msg.position[5]:.4f} rad")
            self.get_logger().info(f"左臂食指: {msg.position[7]:.4f} rad")
            self.get_logger().info(f"左臂拇指: {msg.position[17]:.4f} rad")
            self.get_logger().info(f"右臂肩部: {msg.position[20]:.4f} rad")
            self.get_logger().info(f"右臂肘部: {msg.position[23]:.4f} rad")
            self.get_logger().info(f"右臂腕部: {msg.position[25]:.4f} rad")
            self.get_logger().info(f"右臂食指: {msg.position[27]:.4f} rad")
            self.get_logger().info(f"右臂拇指: {msg.position[37]:.4f} rad")
        
        self.sim_time += 0.01

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    robot_arm_controller = RobotArmController()
    
    try:
        rclpy.spin(robot_arm_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_arm_controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
