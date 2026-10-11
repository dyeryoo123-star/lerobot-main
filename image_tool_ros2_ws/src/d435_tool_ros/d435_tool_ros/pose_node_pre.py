      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pose_node.py
---------------------------------
功能：
- 订阅 RealSense 彩色图像 topic（sensor_msgs/Image）
- 调用你“离线代码库”里的 MediaPipePoseEstimator 进行姿态估计
- 将离线输出（通常是 vis）适配为在线链路需要的 score
- 计算 quality（用于 action_node 门控）
- 发布到 /human_pose/raw（std_msgs/String，JSON）

为什么要做适配：
- 离线 estimator 常返回：(raw_pose_ok: bool, landmarks: list[dict])
  且 landmarks 点里是 {"x","y","z","vis"}（没有 score）
- 在线 action_node 会使用 landmarks[i]["score"] 和 quality 来做过滤与触发判断
  如果不把 vis->score，quality 会始终 0，导致 action_node 不发布动作。
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

# 复用代码库里的 estimator
# from d435_tool.mediapipe_pose.pose_estimator import MediaPipePoseEstimator
from d435_tool.mediapipe_pose.pose_estimator import MediaPipePoseEstimator


def convert_offline_landmark_list_to_ros_landmark_list(
    offline_landmarks: Any
) -> List[Dict[str, float]]:
    """
    将离线 estimator 的 landmarks 列表，转换为 ROS 在线链路统一格式。

    离线常见格式（每个点）：
      {"x":..., "y":..., "z":..., "vis":...}

    在线统一格式（每个点）：
      {"x":..., "y":..., "z":..., "score":...}

    注意：
    - action_node 依赖 score（以及 quality = score 平均值）
    - 若离线点没有 vis/score 等字段，默认给 1.0（表示“点存在，可用”）
      这样不会因为缺字段导致全链路被 quality 门控卡死。
    """
    if not isinstance(offline_landmarks, list):
        return []

    normalized: List[Dict[str, float]] = []
    for point in offline_landmarks:
        if not isinstance(point, dict):
            continue

        # score 的来源优先级：
        # 1) 如果离线已经给了 score，就直接用
        # 2) 否则用 vis（离线最常见）
        # 3) 否则尝试 visibility/presence（有些 mediapipe 输出会用这个名字）
        # 4) 都没有就默认 1.0（表示“点存在但未提供置信度字段”）
        if "score" in point:
            score_value = point["score"]
        elif "vis" in point:
            score_value = point["vis"]
        elif "visibility" in point:
            score_value = point["visibility"]
        elif "presence" in point:
            score_value = point["presence"]
        else:
            score_value = 1.0

        normalized.append(
            {
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "z": float(point.get("z", 0.0)),
                "score": float(score_value),
            }
        )
    return normalized


def compute_pose_quality_from_landmark_scores(
    landmarks: List[Dict[str, float]]
) -> float:
    """
    计算 quality：用所有关键点 score 的平均值来表示“这一帧姿态估计的整体可信度/可用度”。

    逻辑：
    - 如果 landmarks 为空 -> quality = 0
    - 否则 quality = 平均(score)
    - 兜底：如果所有 score 都是 0，但点数量足够（例如 25+），说明“检测到了点但没有置信度字段”
      这种情况不应该让 action_node 永久卡死，因此返回 1.0 作为“可用”。
    """
    if not landmarks:
        return 0.0

    scores = [float(p.get("score", 0.0)) for p in landmarks]
    if not scores:
        return 0.0

    max_score = max(scores)
    if max_score <= 0.0:
        # ✅ 兜底：点很多但 score 全 0，一般是“字段缺失导致默认 0”
        return 1.0 if len(landmarks) >= 25 else 0.0

    avg_score = sum(scores) / float(len(scores))

    # 语义上建议限制在 0~1
    if avg_score < 0.0:
        avg_score = 0.0
    if avg_score > 1.0:
        avg_score = 1.0
    return float(avg_score)


class PoseEstimationPublisherNode(Node):
    """
    ROS2 节点：订阅图像 -> 估计人体姿态 -> 发布 JSON（/human_pose/raw）
    """

    def __init__(self) -> None:
        super().__init__("pose_node")

        # -------------------------
        # 1) 参数：topic 与运行配置
        # -------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("pose_topic", "/human_pose/raw")

        # 模型路径：如果你的离线 estimator 内部默认找 models/*.task，也可以留空
        self.declare_parameter("model_path", "")

        # 日志输出频率：每 N 帧打印一次
        self.declare_parameter("log_every_n_frames", 30)

        # 出错时是否仍发布 JSON（便于你用 ros2 topic echo 看到错误原因）
        self.declare_parameter("publish_error_payload", True)

        self.image_topic_name: str = str(self.get_parameter("image_topic").value)
        self.pose_topic_name: str = str(self.get_parameter("pose_topic").value)
        self.model_path: str = str(self.get_parameter("model_path").value).strip()
        self.log_every_n_frames: int = int(self.get_parameter("log_every_n_frames").value)
        self.publish_error_payload: bool = bool(self.get_parameter("publish_error_payload").value)

        # -------------------------
        # 2) 初始化工具：cv_bridge 与 estimator
        # -------------------------
        self.cv_bridge = CvBridge()

        # ✅ 这里复用离线 estimator
        # 如果你的 MediaPipePoseEstimator 不接受 model_path 参数，就改成 MediaPipePoseEstimator()
        try:
            if self.model_path:
                self.pose_estimator = MediaPipePoseEstimator(self.model_path)
            else:
                self.pose_estimator = MediaPipePoseEstimator()
        except Exception as e:
            self.get_logger().error(f"初始化 MediaPipePoseEstimator 失败：{e}")
            raise

        # -------------------------
        # 3) ROS 通信：订阅图像，发布姿态 JSON
        # -------------------------
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic_name,
            self.on_image_message_received,
            qos_profile_sensor_data,
        )
        self.pose_publisher = self.create_publisher(String, self.pose_topic_name, 10)

        # -------------------------
        # 4) 运行状态
        # -------------------------
        self.frame_index_counter: int = 0
        self.get_logger().info(
            f"[pose_node] 启动成功：订阅={self.image_topic_name} 发布={self.pose_topic_name} model_path='{self.model_path}'"
        )

    def publish_pose_json_payload(self, payload: Dict[str, Any]) -> None:
        """
        将 payload dict 序列化为 JSON 字符串并发布。
        """
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.pose_publisher.publish(message)

    def on_image_message_received(self, image_msg: Image) -> None:
        """
        图像回调：每收到一帧图像就执行一次姿态估计并发布结果。
        """
        self.frame_index_counter += 1
        frame_index = self.frame_index_counter

        # 1) ROS Image -> OpenCV BGR
        try:
            bgr_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"[pose_node] cv_bridge 转换失败：{e}")
            if self.publish_error_payload:
                self.publish_pose_json_payload(
                    {
                        "status": "cv_bridge_error",
                        "error": str(e),
                        "image_topic": self.image_topic_name,
                    }
                )
            return

        # 2) 调用离线 estimator（离线 estimator 一般就是 estimate_from_bgr(bgr)）
        #    返回格式通常为：(raw_pose_ok: bool, offline_landmarks: list[dict])
        try:
            estimation_result = self.pose_estimator.estimate_from_bgr(bgr_image)
        except Exception as e:
            self.get_logger().error(f"[pose_node] pose_estimator 推理失败：{e}")
            if self.publish_error_payload:
                self.publish_pose_json_payload(
                    {
                        "status": "estimator_error",
                        "error": str(e),
                        "image_topic": self.image_topic_name,
                    }
                )
            return

        # 3) 解析返回：确保支持 tuple 契约
        raw_pose_ok: bool = False
        offline_landmarks: Any = None

        if isinstance(estimation_result, tuple) and len(estimation_result) == 2 and isinstance(estimation_result[0], (bool, int)):
            raw_pose_ok = bool(estimation_result[0])
            offline_landmarks = estimation_result[1]
        else:
            # 如果这里发生，说明 estimator 的输出契约与你离线版本不一致
            # 你可以先打印类型，定位是哪一种结构
            self.get_logger().warn(
                f"[pose_node] estimator 返回结构非预期：type={type(estimation_result)}，将按失败处理"
            )
            raw_pose_ok = False
            offline_landmarks = []

        # 4) 离线 landmarks -> 在线统一 landmarks（vis -> score）
        normalized_landmarks = convert_offline_landmark_list_to_ros_landmark_list(offline_landmarks)

        # 5) 计算质量分 quality（给 action_node 过滤用）
        quality_value = compute_pose_quality_from_landmark_scores(normalized_landmarks)

        # 6) 构造 JSON payload
        payload: Dict[str, Any] = {
            "status": "ok",
            "stamp": {"sec": int(image_msg.header.stamp.sec), "nanosec": int(image_msg.header.stamp.nanosec)},
            "frame_id": image_msg.header.frame_id,
            "image_width": int(bgr_image.shape[1]),
            "image_height": int(bgr_image.shape[0]),
            "frame_index": int(frame_index),

            # raw_pose_ok：来自离线 estimator 的“原始检测是否成功”
            "raw_pose_ok": bool(raw_pose_ok),

            # landmarks：统一后的 33 点，每点包含 x/y/z/score
            "landmarks": normalized_landmarks,

            # quality：score 的均值（或兜底可用性）
            "quality": float(quality_value),
        }

        self.publish_pose_json_payload(payload)

        # 7) 适度日志：每 N 帧打印一次
        if self.log_every_n_frames > 0 and (frame_index % self.log_every_n_frames == 0):
            self.get_logger().info(
                f"[pose_node] 发布姿态：frame={frame_index}, lm={len(normalized_landmarks)}, "
                f"quality={quality_value:.3f}, raw_pose_ok={raw_pose_ok}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseEstimationPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

    