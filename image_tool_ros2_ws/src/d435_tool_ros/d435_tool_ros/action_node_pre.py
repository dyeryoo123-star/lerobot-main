      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action_node.py
---------------------------------
功能：
- 订阅 /human_pose/raw（std_msgs/String，JSON）
- 取出 landmarks（33点）与 quality
- 基于简单规则识别动作：
  1) 抬手 RAISE_HAND：手腕在肩膀之上（y 更小）
  2) 挥手 WAVE：手腕 x 相对肩中心摆动（标准差 + 符号变化次数）
- 发布 /human_action（std_msgs/String，JSON）
- 带 triggered + cooldown 防止连续触发机械臂动作
"""

import json
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# MediaPipe Pose 常用关键点索引
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16


def get_landmark_with_validation(landmarks: List[Dict[str, Any]], index: int) -> Optional[Dict[str, float]]:
    """
    安全获取某个 landmark 点，并确保返回包含 x/y/z/score。
    若 index 越界或格式不对，则返回 None。
    """
    if not isinstance(landmarks, list):
        return None
    if index < 0 or index >= len(landmarks):
        return None
    point = landmarks[index]
    if not isinstance(point, dict):
        return None
    return {
        "x": float(point.get("x", 0.0)),
        "y": float(point.get("y", 0.0)),
        "z": float(point.get("z", 0.0)),
        "score": float(point.get("score", 0.0)),
    }


def compute_mean(values: List[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def compute_standard_deviation(values: List[float]) -> float:
    """
    计算标准差，用于衡量摆动幅度（挥手识别的一种简单特征）。
    """
    if len(values) <= 1:
        return 0.0
    mean_value = compute_mean(values)
    variance = sum((v - mean_value) ** 2 for v in values) / float(len(values))
    return math.sqrt(variance)


class HumanActionRecognitionNode(Node):
    """
    ROS2 节点：订阅姿态 JSON -> 识别动作 -> 发布动作 JSON
    """

    def __init__(self) -> None:
        super().__init__("action_node")

        # -------------------------
        # 1) 参数：topic
        # -------------------------
        self.declare_parameter("pose_topic", "/human_pose/raw")
        self.declare_parameter("action_topic", "/human_action")

        # -------------------------
        # 2) 参数：门控阈值（重要！）
        # -------------------------
        # min_quality：姿态整体质量不够就直接清空窗口并跳过（防误触）
        self.declare_parameter("min_quality", 0.25)

        # min_keypoint_score：肩/腕关键点 score 不够就跳过（建议对齐 vis 语义，如 0.5）
        self.declare_parameter("min_keypoint_score", 0.50)

        # -------------------------
        # 3) 参数：滑窗与冷却
        # -------------------------
        self.declare_parameter("window_size", 15)
        self.declare_parameter("cooldown_frames", 30)

        # -------------------------
        # 4) 参数：动作阈值（可按实际调参）
        # -------------------------
        # 抬手：shoulder_y - wrist_y >= raise_hand_min_delta_y
        self.declare_parameter("raise_hand_min_delta_y", 0.10)

        # 挥手：轨迹标准差 >= wave_min_std 且 符号变化次数 >= wave_min_sign_changes
        self.declare_parameter("wave_min_std", 0.015)
        self.declare_parameter("wave_min_sign_changes", 3)

        # -------------------------
        # 5) 参数：日志频率
        # -------------------------
        self.declare_parameter("log_every_n_frames", 30)

        # 读取参数
        self.pose_topic_name = str(self.get_parameter("pose_topic").value)
        self.action_topic_name = str(self.get_parameter("action_topic").value)

        self.min_quality = float(self.get_parameter("min_quality").value)
        self.min_keypoint_score = float(self.get_parameter("min_keypoint_score").value)

        self.window_size = int(self.get_parameter("window_size").value)
        self.cooldown_frames = int(self.get_parameter("cooldown_frames").value)

        self.raise_hand_min_delta_y = float(self.get_parameter("raise_hand_min_delta_y").value)

        self.wave_min_std = float(self.get_parameter("wave_min_std").value)
        self.wave_min_sign_changes = int(self.get_parameter("wave_min_sign_changes").value)

        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)

        # -------------------------
        # 运行状态：滑窗 + 冷却
        # window 存 (frame_index, landmarks, quality, stamp)
        # -------------------------
        self.pose_window: Deque[Tuple[int, List[Dict[str, Any]], float, Dict[str, Any]]] = deque(maxlen=self.window_size)
        self.last_triggered_frame_index: int = -10**9

        # -------------------------
        # ROS 通信
        # -------------------------
        self.pose_subscription = self.create_subscription(String, self.pose_topic_name, self.on_pose_json_received, 10)
        self.action_publisher = self.create_publisher(String, self.action_topic_name, 10)

        self.get_logger().info(
            f"[action_node] 启动成功：订阅={self.pose_topic_name} 发布={self.action_topic_name} "
            f"min_quality={self.min_quality} min_keypoint_score={self.min_keypoint_score} window={self.window_size}"
        )

    def publish_action_json_payload(self, payload: Dict[str, Any]) -> None:
        """
        发布动作 JSON。
        """
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.action_publisher.publish(message)

    def is_cooldown_satisfied(self, frame_index: int) -> bool:
        """
        冷却判断：避免动作连续触发导致机械臂频繁执行。
        """
        return (frame_index - self.last_triggered_frame_index) >= self.cooldown_frames

    def on_pose_json_received(self, msg: String) -> None:
        """
        接收姿态 JSON，更新滑窗并尝试识别动作。
        """
        # 1) 解析 JSON
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        # 2) 过滤非 ok 帧
        if data.get("status") != "ok":
            self.pose_window.clear()
            return

        landmarks = data.get("landmarks", [])
        quality = float(data.get("quality", 0.0))
        frame_index = int(data.get("frame_index", 0))
        stamp = data.get("stamp", {"sec": 0, "nanosec": 0})

        # 3) 门控：质量不足则清窗口并跳过
        if quality < self.min_quality or not isinstance(landmarks, list) or len(landmarks) < 17:
            self.pose_window.clear()
            return

        # 4) 入窗
        self.pose_window.append((frame_index, landmarks, quality, stamp))

        # 5) 窗口不够长，不做识别
        if len(self.pose_window) < max(6, self.window_size // 2):
            return

        # 6) 识别动作
        action_name, confidence, debug_info = self.predict_action_from_pose_window()

        # 7) 冷却 + 触发
        triggered = False
        if action_name != "IDLE" and confidence > 0.0 and self.is_cooldown_satisfied(frame_index):
            triggered = True
            self.last_triggered_frame_index = frame_index

        # 8) 发布动作 JSON
        payload = {
            "status": "ok",
            "stamp": stamp,
            "frame_index": frame_index,
            "action": action_name,
            "confidence": float(confidence),
            "triggered": bool(triggered),
            "debug": debug_info,
        }
        self.publish_action_json_payload(payload)

        # 9) 日志
        if self.log_every_n_frames > 0 and (frame_index % self.log_every_n_frames == 0):
            self.get_logger().info(
                f"[action_node] action={action_name} conf={confidence:.2f} triggered={triggered} debug={debug_info}"
            )

    def predict_action_from_pose_window(self) -> Tuple[str, float, Dict[str, Any]]:
        """
        从滑窗中预测动作。
        返回：
          (action_name, confidence, debug_info)
        """
        # 取最新帧
        latest_frame_index, latest_landmarks, latest_quality, _ = self.pose_window[-1]

        # -------------------------
        # A) 抬手识别：左右手分别判断，只要一边满足就触发
        # -------------------------
        left_shoulder = get_landmark_with_validation(latest_landmarks, LANDMARK_LEFT_SHOULDER)
        right_shoulder = get_landmark_with_validation(latest_landmarks, LANDMARK_RIGHT_SHOULDER)
        left_wrist = get_landmark_with_validation(latest_landmarks, LANDMARK_LEFT_WRIST)
        right_wrist = get_landmark_with_validation(latest_landmarks, LANDMARK_RIGHT_WRIST)

        def is_landmark_pair_confident(shoulder: Optional[Dict[str, float]], wrist: Optional[Dict[str, float]]) -> bool:
            return (
                shoulder is not None
                and wrist is not None
                and shoulder["score"] >= self.min_keypoint_score
                and wrist["score"] >= self.min_keypoint_score
            )

        left_delta_y = 0.0
        right_delta_y = 0.0
        left_raise = False
        right_raise = False

        if is_landmark_pair_confident(left_shoulder, left_wrist):
            # y 越小表示越靠上；shoulder_y - wrist_y 越大表示手抬得越高
            left_delta_y = left_shoulder["y"] - left_wrist["y"]
            left_raise = left_delta_y >= self.raise_hand_min_delta_y

        if is_landmark_pair_confident(right_shoulder, right_wrist):
            right_delta_y = right_shoulder["y"] - right_wrist["y"]
            right_raise = right_delta_y >= self.raise_hand_min_delta_y

        if left_raise or right_raise:
            best_delta_y = max(left_delta_y, right_delta_y)

            # 简单 confidence：抬得越高越可信，同时结合 quality
            # 这里做一个温和映射：超过阈值后逐渐上升
            ratio = best_delta_y / max(1e-6, self.raise_hand_min_delta_y)
            base_conf = min(1.0, max(0.0, (ratio - 1.0)))  # 刚好等于阈值时为0，超过才增长
            confidence = min(1.0, 0.6 + 0.4 * base_conf) * min(1.0, latest_quality + 0.2)

            debug = {
                "type": "raise_hand",
                "quality": float(latest_quality),
                "left_delta_y": float(left_delta_y),
                "right_delta_y": float(right_delta_y),
                "min_delta_y": float(self.raise_hand_min_delta_y),
            }
            return "RAISE_HAND", float(confidence), debug

        # -------------------------
        # B) 挥手识别：手腕 x 相对肩中心的摆动
        # -------------------------
        # 选择可信的一只手（左/右）
        use_left_wrist = left_wrist is not None and left_wrist["score"] >= self.min_keypoint_score
        use_right_wrist = right_wrist is not None and right_wrist["score"] >= self.min_keypoint_score
        if not (use_left_wrist or use_right_wrist):
            return "IDLE", 0.0, {"type": "no_wrist_confident"}

        # 肩中心 x（用于消除整体平移）
        shoulder_x_list: List[float] = []
        if left_shoulder is not None and left_shoulder["score"] >= self.min_keypoint_score:
            shoulder_x_list.append(left_shoulder["x"])
        if right_shoulder is not None and right_shoulder["score"] >= self.min_keypoint_score:
            shoulder_x_list.append(right_shoulder["x"])
        shoulder_center_x = compute_mean(shoulder_x_list) if shoulder_x_list else 0.5

        # 构造窗口内 wrist_x 相对 shoulder_center_x 的序列
        relative_wrist_x_series: List[float] = []
        quality_series: List[float] = []

        target_wrist_index = LANDMARK_LEFT_WRIST if use_left_wrist else LANDMARK_RIGHT_WRIST

        for _, window_landmarks, window_quality, _ in self.pose_window:
            wrist_point = get_landmark_with_validation(window_landmarks, target_wrist_index)
            if wrist_point is None or wrist_point["score"] < self.min_keypoint_score:
                continue
            relative_wrist_x_series.append(wrist_point["x"] - shoulder_center_x)
            quality_series.append(window_quality)

        if len(relative_wrist_x_series) < max(6, self.window_size // 2):
            return "IDLE", 0.0, {"type": "wave_window_too_short", "n": len(relative_wrist_x_series)}

        # 计算符号变化次数（粗略判断左右摆动）
        sign_change_count = 0
        previous_sign = 0
        for x in relative_wrist_x_series:
            current_sign = 1 if x > 0 else (-1 if x < 0 else 0)
            if previous_sign != 0 and current_sign != 0 and current_sign != previous_sign:
                sign_change_count += 1
            if current_sign != 0:
                previous_sign = current_sign

        series_std = compute_standard_deviation(relative_wrist_x_series)
        mean_quality = compute_mean(quality_series)

        if series_std >= self.wave_min_std and sign_change_count >= self.wave_min_sign_changes:
            # confidence：std + sign_changes 综合，再乘质量
            conf_by_std = min(1.0, series_std / max(1e-6, self.wave_min_std))
            conf_by_sign = min(1.0, sign_change_count / max(1, self.wave_min_sign_changes))
            confidence = 0.5 * conf_by_std + 0.5 * conf_by_sign
            confidence = min(1.0, confidence) * min(1.0, mean_quality + 0.2)

            debug = {
                "type": "wave",
                "use_left_wrist": bool(use_left_wrist),
                "std": float(series_std),
                "sign_changes": int(sign_change_count),
                "mean_quality": float(mean_quality),
                "min_std": float(self.wave_min_std),
                "min_sign_changes": int(self.wave_min_sign_changes),
            }
            return "WAVE", float(confidence), debug

        # 默认无动作
        return "IDLE", 0.0, {
            "type": "no_match",
            "std": float(series_std),
            "sign_changes": int(sign_change_count),
            "min_std": float(self.wave_min_std),
            "min_sign_changes": int(self.wave_min_sign_changes),
            "mean_quality": float(mean_quality),
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanActionRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

    