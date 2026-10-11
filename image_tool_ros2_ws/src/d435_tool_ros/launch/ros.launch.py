      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # LaunchConfiguration 是 substitution，不能直接 int()/float()/bool()
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    enable_depth = LaunchConfiguration("enable_depth")

    pose_topic = LaunchConfiguration("pose_topic")
    action_topic = LaunchConfiguration("action_topic")

    daemon_host = LaunchConfiguration("daemon_host")
    daemon_port = LaunchConfiguration("daemon_port")
    min_confidence = LaunchConfiguration("min_confidence")

    near_distance_m = LaunchConfiguration("near_distance_m")
    far_distance_m = LaunchConfiguration("far_distance_m")
    min_safe_distance_m = LaunchConfiguration("min_safe_distance_m")

    # 用 ParameterValue 指定参数类型（推荐）
    daemon_port_int = ParameterValue(daemon_port, value_type=int)
    min_confidence_float = ParameterValue(min_confidence, value_type=float)
    enable_depth_bool = ParameterValue(enable_depth, value_type=bool)

    near_distance_float = ParameterValue(near_distance_m, value_type=float)
    far_distance_float = ParameterValue(far_distance_m, value_type=float)
    min_safe_distance_float = ParameterValue(min_safe_distance_m, value_type=float)

    return LaunchDescription([
        # ---------- arguments ----------
        DeclareLaunchArgument(
            "color_topic",
            default_value="/camera/camera/color/image_raw",
            description="RealSense RGB image topic",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="RealSense aligned depth topic (16UC1 mm)",
        ),
        DeclareLaunchArgument(
            "enable_depth",
            default_value="true",
            description="Enable aligned depth subscription",
        ),
        DeclareLaunchArgument(
            "pose_topic",
            default_value="/human_pose/raw",
            description="Pose JSON output topic",
        ),
        DeclareLaunchArgument(
            "action_topic",
            default_value="/human_action",
            description="Action JSON output topic",
        ),
        DeclareLaunchArgument(
            "daemon_host",
            default_value="127.0.0.1",
            description="LeRobot daemon host",
        ),
        DeclareLaunchArgument(
            "daemon_port",
            default_value="8765",
            description="LeRobot daemon TCP port",
        ),
        DeclareLaunchArgument(
            "min_confidence",
            default_value="0.65",
            description="Min confidence to trigger robot action",
        ),
        # 深度控制阈值（带滞回）
        DeclareLaunchArgument(
            "near_distance_m",
            default_value="0.75",
            description="< near => PROX_RAISE",
        ),
        DeclareLaunchArgument(
            "far_distance_m",
            default_value="0.95",
            description="> far => PROX_HOME",
        ),
        DeclareLaunchArgument(
            "min_safe_distance_m",
            default_value="0.45",
            description="< min_safe => ESTOP",
        ),

        # ---------- nodes ----------
        Node(
            package="d435_tool_ros",
            executable="pose_node",
            name="pose_node",
            output="screen",
            parameters=[{
                # 注意：这里要用 pose_node_updated_v2 对应的参数名
                "color_topic": color_topic,
                "depth_topic": depth_topic,
                "enable_depth": enable_depth_bool,

                # 你已经确认 encoding=16UC1，通常是 mm
                "depth_unit_is_mm": True,
                "depth_roi": 9,
                "depth_sync_tolerance_ms": 200,

                "pose_topic": pose_topic,
                "log_every_n": 30,
                "publish_on_error": True,
            }],
        ),

        Node(
            package="d435_tool_ros",
            executable="action_node",
            name="action_node",
            output="screen",
            parameters=[{
                "pose_topic": pose_topic,
                "action_topic": action_topic,

                # 深度靠近/远离
                "enable_depth_control": True,
                "near_distance_m": near_distance_float,
                "far_distance_m": far_distance_float,
                "min_safe_distance_m": min_safe_distance_float,
            }],
        ),

        Node(
            package="d435_tool_ros",
            executable="lerobot_client_node",
            name="lerobot_client_node",
            output="screen",
            parameters=[{
                "action_topic": action_topic,
                "daemon_host": daemon_host,
                "daemon_port": daemon_port_int,
                "min_confidence": min_confidence_float,

                # per-action cooldown（新版本参数名）
                "cooldown_wave_sec": 1.0,
                "cooldown_turn_sec": 0.8,
                "cooldown_raise_sec": 0.8,
                "cooldown_home_sec": 0.2,
                "cooldown_estop_sec": 0.0,

                # wave
                "wave_joint": "Motor_3.pos",
                "wave_amp": 25.0,
                "wave_repeat": 3,
                "wave_step_sleep": 0.35,

                # raise_arm
                "raise_lift_m1": 25.0,
                "raise_elbow_m2": 15.0,
                "raise_motion_time_ms": 1200,

                # turn
                "turn_delta_m0": 20.0,
                "turn_motion_time_ms": 900,
            }],
        ),
    ])

    