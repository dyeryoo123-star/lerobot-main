      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pose_node.py
=========================================================
功能：
- 订阅 RealSense 彩色图像（RGB）topic
- （可选）订阅 aligned depth（对齐到彩色图的深度）topic
- MediaPipe Pose 推理得到 33 个关键点
- 发布 /human_pose/raw（std_msgs/String，JSON）

你现在的目标是“实时控制机械臂”，所以这里做了两件关键增强：
1) 兼容 estimator 返回 tuple 的情况（你之前遇到过 result_type=tuple）
2) 深度融合：从 aligned depth 图中取人体距离、左右手腕距离（单位：米）

注意：
- 你反馈 depth encoding 是 16UC1，通常单位是 mm（毫米）。
  本节点默认把 16UC1 当作 mm -> m 时乘 0.001。
- 深度取值建议用 ROI 的中位数（median），抗噪更稳。
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import numpy as np

# 你的核心库（保持复用）
from d435_tool.mediapipe_pose.pose_estimator import MediaPipePoseEstimator


# BlazePose landmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24


def _stamp_to_ms(stamp) -> int:
    """ROS header.stamp -> 毫秒"""
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return sec * 1000 + nanosec // 1_000_000


def _extract_landmarks_generic(result: Any) -> List[Dict[str, float]]:
    """
    将 PoseEstimator 的输出尽量转成统一的 landmarks 列表：
    [{"x":..., "y":..., "z":..., "score":...}, ...]  (期望 33 个)

    兼容：
    - result 是 tuple：取第 0 个元素作为 landmarks（你之前遇到的情况）
    - result 是 list：list[dict] 或 list[mediapipe landmark]
    - result 是 dict 且包含 "landmarks"
    - result 是 mediapipe results，含 pose_landmarks.landmark
    """
    if result is None:
        return []

    # 0) tuple 兼容：你的 estimator 常见返回 (has_pose: bool, landmarks: list)
    #    - 如果是这种形式，应该取第 1 个元素（landmarks）
    #    - 也兼容 (landmarks, extra) 的形式
    if isinstance(result, tuple):
        if len(result) >= 2 and isinstance(result[0], bool) and isinstance(result[1], list):
            result = result[1]
        elif len(result) >= 1:
            result = result[0]

    # 1) list
    if isinstance(result, list):
        out: List[Dict[str, float]] = []
        for p in result:
            if isinstance(p, dict):
                out.append(
                    {
                        "x": float(p.get("x", 0.0)),
                        "y": float(p.get("y", 0.0)),
                        "z": float(p.get("z", 0.0)),
                        # score 优先 score，否则 visibility/presence
                        "score": float(p.get("score", p.get("vis", p.get("visibility", p.get("presence", 0.0))))),
                    }
                )
            else:
                out.append(
                    {
                        "x": float(getattr(p, "x", 0.0)),
                        "y": float(getattr(p, "y", 0.0)),
                        "z": float(getattr(p, "z", 0.0)),
                        "score": float(getattr(p, "visibility", getattr(p, "presence", 0.0))),
                    }
                )
        return out

    # 2) dict 包含 landmarks
    if isinstance(result, dict) and "landmarks" in result:
        return _extract_landmarks_generic(result.get("landmarks"))

    # 3) mediapipe result: result.pose_landmarks.landmark
    pose_lms = getattr(result, "pose_landmarks", None)
    if pose_lms is not None:
        lm_list = getattr(pose_lms, "landmark", None)
        if lm_list is not None:
            return _extract_landmarks_generic(list(lm_list))

    return []


def _compute_quality(landmarks: List[Dict[str, float]]) -> float:
    """将所有点的 score 做平均，作为整体质量分数（0~1）"""
    if not landmarks:
        return 0.0
    scores = [float(p.get("score", 0.0)) for p in landmarks]
    return float(sum(scores) / max(len(scores), 1))


class PoseNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_node")

        # ===== 参数（可在 launch 中覆盖）=====
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("pose_topic", "/human_pose/raw")

        self.declare_parameter("enable_depth", True)

        # 深度编码与单位：你现在是 16UC1，一般是 mm
        self.declare_parameter("depth_encoding_is_mm", True)
        self.declare_parameter("depth_roi_radius_px", 3)  # ROI 半径：3 -> 7x7
        self.declare_parameter("depth_sync_tolerance_ms", 80)  # RGB 与 depth 最大允许时间差

        self.declare_parameter("log_every_n", 30)
        self.declare_parameter("publish_on_error", True)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        self.enable_depth = bool(self.get_parameter("enable_depth").value)
        self.depth_encoding_is_mm = bool(self.get_parameter("depth_encoding_is_mm").value)
        self.depth_roi_radius_px = int(self.get_parameter("depth_roi_radius_px").value)
        self.depth_sync_tolerance_ms = int(self.get_parameter("depth_sync_tolerance_ms").value)

        self.log_every_n = int(self.get_parameter("log_every_n").value)
        self.publish_on_error = bool(self.get_parameter("publish_on_error").value)

        # ===== 初始化 =====
        self.bridge = CvBridge()
        self.estimator = MediaPipePoseEstimator()

        self.pub_pose = self.create_publisher(String, self.pose_topic, 10)
        self.sub_color = self.create_subscription(Image, self.color_topic, self.on_color, qos_profile_sensor_data)

        # 缓存最近一帧 depth（aligned depth）
        self._last_depth: Optional[np.ndarray] = None
        self._last_depth_stamp_ms: int = -1
        if self.enable_depth:
            self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos_profile_sensor_data)
        else:
            self.sub_depth = None

        self.frame_count = 0

        public = [m for m in dir(self.estimator) if not m.startswith("_")]
        self.get_logger().info(f"PoseEstimator public attrs: {public}")
        self.get_logger().info(
            f"pose_node started. color={self.color_topic} depth={self.depth_topic} enable_depth={self.enable_depth} "
            f"publish={self.pose_topic}"
        )

    def _publish_json(self, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_pose.publish(msg)

    def _call_estimator(self, bgr_image: np.ndarray) -> Any:
        """
        适配 PoseEstimator 的推理入口（优先用 estimate_from_bgr）。
        """
        fn = getattr(self.estimator, "estimate_from_bgr", None)
        if callable(fn):
            return fn(bgr_image)

        # 兜底：尝试常见方法名
        for name in ["infer", "estimate", "process", "predict", "run", "forward", "detect"]:
            f = getattr(self.estimator, name, None)
            if callable(f):
                return f(bgr_image)

        # 最后兜底：类本身可调用
        if callable(self.estimator):
            return self.estimator(bgr_image)

        raise RuntimeError("PoseEstimator has no callable inference method.")

    # ---------------- depth ----------------
    def on_depth(self, msg: Image) -> None:
        """缓存 aligned depth（16UC1 或 32FC1）"""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self._last_depth = depth.copy()
            self._last_depth_stamp_ms = _stamp_to_ms(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f"depth cv_bridge failed: {e}")

    def _depth_at_norm_xy(self, norm_x: float, norm_y: float) -> Optional[float]:
        """
        取某个归一化坐标点的深度（米）。
        - 使用 ROI 中位数（median）更稳
        - 自动过滤 0 深度
        """
        if self._last_depth is None:
            return None

        depth = self._last_depth
        h, w = depth.shape[:2]
        px = int(np.clip(norm_x * w, 0, w - 1))
        py = int(np.clip(norm_y * h, 0, h - 1))

        r = max(1, int(self.depth_roi_radius_px))
        x0 = max(0, px - r)
        x1 = min(w, px + r + 1)
        y0 = max(0, py - r)
        y1 = min(h, py + r + 1)

        roi = depth[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        # 过滤无效深度（0）
        valid = roi[roi > 0]
        if valid.size < 10:
            return None

        d = float(np.median(valid))

        # 16UC1 常见是 mm；32FC1 常见是 m（但也可能是 mm）
        if depth.dtype == np.uint16:
            if self.depth_encoding_is_mm:
                d = d * 0.001
        else:
            # float32/float64：一般就是米，如果你确实是 mm，也可以把参数设为 True
            if self.depth_encoding_is_mm:
                d = d * 0.001

        return d

    def _person_distance(self, landmarks: List[Dict[str, float]]) -> Optional[float]:
        """
        估算“人体距离”：优先用双髋中心，否则用双肩中心。
        """
        if len(landmarks) < 25:
            return None

        def ok(idx: int) -> bool:
            try:
                return float(landmarks[idx].get("score", 0.0)) > 0.2
            except Exception:
                return False

        # hip center
        if ok(L_HIP) and ok(R_HIP):
            cx = 0.5 * (float(landmarks[L_HIP]["x"]) + float(landmarks[R_HIP]["x"]))
            cy = 0.5 * (float(landmarks[L_HIP]["y"]) + float(landmarks[R_HIP]["y"]))
            return self._depth_at_norm_xy(cx, cy)

        # shoulder center
        if ok(L_SHOULDER) and ok(R_SHOULDER):
            cx = 0.5 * (float(landmarks[L_SHOULDER]["x"]) + float(landmarks[R_SHOULDER]["x"]))
            cy = 0.5 * (float(landmarks[L_SHOULDER]["y"]) + float(landmarks[R_SHOULDER]["y"]))
            return self._depth_at_norm_xy(cx, cy)

        return None

    # ---------------- color ----------------
    def on_color(self, msg: Image) -> None:
        self.frame_count += 1
        stamp_ms = _stamp_to_ms(msg.header.stamp)

        # 1) ROS Image -> BGR
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"color cv_bridge failed: {e}")
            if self.publish_on_error:
                self._publish_json({"status": "cv_bridge_error", "error": str(e)})
            return

        # 2) MediaPipe 推理
        try:
            result = self._call_estimator(bgr)
        except Exception as e:
            self.get_logger().error(f"pose estimator failed: {e}")
            if self.publish_on_error:
                self._publish_json({"status": "estimator_error", "error": str(e)})
            return

        # 3) landmarks + quality
        landmarks = _extract_landmarks_generic(result)
        quality = _compute_quality(landmarks)

        # 4) 深度融合（只要 depth 可用且时间差合理）
        depth_payload: Dict[str, Any] = {"available": False}
        if self.enable_depth and self._last_depth is not None and self._last_depth_stamp_ms > 0:
            age_ms = abs(stamp_ms - self._last_depth_stamp_ms)
            if age_ms <= self.depth_sync_tolerance_ms and landmarks:
                person_d = self._person_distance(landmarks)

                left_wrist_d = None
                right_wrist_d = None
                if len(landmarks) > L_WRIST:
                    left_wrist_d = self._depth_at_norm_xy(float(landmarks[L_WRIST]["x"]), float(landmarks[L_WRIST]["y"]))
                if len(landmarks) > R_WRIST:
                    right_wrist_d = self._depth_at_norm_xy(float(landmarks[R_WRIST]["x"]), float(landmarks[R_WRIST]["y"]))

                depth_payload = {
                    "available": True,
                    "age_ms": int(age_ms),
                    "encoding": str(msg.encoding),
                    "person_distance_m": person_d,
                    "left_wrist_distance_m": left_wrist_d,
                    "right_wrist_distance_m": right_wrist_d,
                }
            else:
                depth_payload = {"available": False, "age_ms": int(age_ms)}

        payload = {
            "status": "ok",
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "frame_id": str(msg.header.frame_id),
            "frame": int(self.frame_count),
            "image": {"width": int(msg.width), "height": int(msg.height), "encoding": str(msg.encoding)},
            "landmarks": landmarks,
            "quality": float(quality),
            "depth": depth_payload,
        }

        self._publish_json(payload)

        if self.log_every_n > 0 and (self.frame_count % self.log_every_n == 0):
            self.get_logger().info(
                f"published pose: frame={self.frame_count}, lm={len(landmarks)}, quality={quality:.3f}, "
                f"depth={bool(depth_payload.get('available', False))}"
            )


def main() -> None:
    rclpy.init()
    node = PoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

    