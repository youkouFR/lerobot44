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
        self.sim_time = 0.0
        
        # 关节名称（从 hello_world_isaaclab.py 中获取）
        self.l_joints = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_arm_yaw_joint",
                        "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"]
        self.r_joints = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_arm_yaw_joint",
                        "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"]
        
        # 合并左右手臂关节
        self.joint_names = self.l_joints + self.r_joints
        
        # 手臂目标姿态（从 hello_world_isaaclab.py 中获取）
        self.l_arm_targets = np.array([-1.3, -0.2, 1.0, -1.5, 0.0, -1.0, 0.0])
        self.r_arm_targets = np.array([-0.5, -0.2, -1.0, -1.5, 0.0, -1.0, 0.0])
        
        # 运动参数
        self.freq = 1.5  # 频率
        self.amp = 0.4    # 振幅
        
        self.get_logger().info(f"机器人手臂控制器已启动")
        self.get_logger().info(f"控制关节数量: {len(self.joint_names)}")
        self.get_logger().info(f"左臂关节: {self.l_joints}")
        self.get_logger().info(f"右臂关节: {self.r_joints}")
    
    def timer_callback(self):
        """定时发布关节命令"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # 计算震荡值
        oscillation = self.amp * math.sin(2.0 * math.pi * self.freq * self.sim_time)
        
        # 生成关节位置命令
        msg.position = []
        msg.velocity = []
        msg.effort = []
        
        # 左臂目标（第一个关节叠加周期震荡）
        for i, joint_name in enumerate(self.l_joints):
            if i == 0:
                # 第一个关节（肩部俯仰）叠加震荡
                position = self.l_arm_targets[i] + oscillation
            else:
                # 其他关节保持固定目标值
                position = self.l_arm_targets[i]
            
            msg.position.append(position)
            msg.velocity.append(0.0)  # 速度设置为 0
            msg.effort.append(0.0)    # 力/力矩设置为 0
        
        # 右臂目标（第一个关节叠加周期震荡）
        for i, joint_name in enumerate(self.r_joints):
            if i == 0:
                # 第一个关节（肩部俯仰）叠加震荡
                position = self.r_arm_targets[i] + oscillation
            else:
                # 其他关节保持固定目标值
                position = self.r_arm_targets[i]
            
            msg.position.append(position)
            msg.velocity.append(0.0)  # 速度设置为 0
            msg.effort.append(0.0)    # 力/力矩设置为 0
        
        self.publisher_.publish(msg)
        
        # 每 1 秒打印一次
        if int(self.sim_time) != int(self.sim_time + 0.01):
            self.get_logger().info(f"发布关节命令 - 时间: {self.sim_time:.2f}s")
            self.get_logger().info(f"左臂肩部目标: {msg.position[0]:.4f} rad")
            self.get_logger().info(f"右臂肩部目标: {msg.position[7]:.4f} rad")
        
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
