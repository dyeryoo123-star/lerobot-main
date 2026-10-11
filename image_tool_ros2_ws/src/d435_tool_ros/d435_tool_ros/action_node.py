      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action_node.py
=========================================================
作用：把 /human_pose/raw（PoseNode 输出的 JSON）转换成“可控、可解释、可抢占”的动作指令 /human_action。

你现在的目标动作集：
- 抬臂：RAISE_ARM（举手）
- 挥臂：WAVE（手左右摆动）
- 转向：TURN_LEFT / TURN_RIGHT（手水平伸向左右）
- 深度靠近：PROX_RAISE（人靠近 -> 抬臂）
- 深度远离：PROX_HOME（人远离 -> 回位）
- 紧急停止：ESTOP（双手同时举过肩 -> 锁存）
- 安全回位：HOME（丢失人体/质量过低/ESTOP 解除后）

设计原则（防止“动作混乱/卡死”）：
1) 明确优先级：ESTOP > fail-safe HOME > 深度靠近/远离 > WAVE > TURN > RAISE
2) 动作独立 cooldown：避免 wave 后 HOME 被挡住
3) ESTOP 锁存 + 复位手势：防止误触后马上恢复又继续乱动
4) 低质量/丢失人体超时：自动 HOME（防失控）
"""

import json
import math
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# BlazePose indices
L_SHOULDER = 11
R_SHOULDER = 12
L_WRIST = 15
R_WRIST = 16


def _safe_landmark(landmarks: Any, idx: int) -> Optional[Dict[str, float]]:
    if not isinstance(landmarks, list):
        return None
    if idx < 0 or idx >= len(landmarks):
        return None
    p = landmarks[idx]
    if not isinstance(p, dict):
        return None
    return {
        "x": float(p.get("x", 0.0)),
        "y": float(p.get("y", 0.0)),
        "z": float(p.get("z", 0.0)),
        "score": float(p.get("score", 0.0)),
    }


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: List[float]) -> float:
    m = _mean(xs)
    v = _mean([(x - m) ** 2 for x in xs])
    return math.sqrt(v)


class ActionNode(Node):
    def __init__(self) -> None:
        super().__init__("action_node")

        # ---------- topics ----------
        self.declare_parameter("pose_topic", "/human_pose/raw")
        self.declare_parameter("action_topic", "/human_action")

        # ---------- 基础门槛 ----------
        self.declare_parameter("window_size", 15)
        self.declare_parameter("min_quality", 0.25)    # 低于此值，认为姿态不可靠
        self.declare_parameter("min_kpt_score", 0.35)  # wrist/shoulder 可见性门槛

        # ---------- 失效安全 ----------
        self.declare_parameter("lost_pose_timeout_sec", 0.7)  # 超时无可靠 pose -> HOME

        # ---------- 抬臂 / 转向 ----------
        self.declare_parameter("raise_min_dy", 0.08)     # shoulder_y - wrist_y
        self.declare_parameter("turn_min_abs_dx", 0.18)  # wrist_x - shoulder_x

        # ---------- 挥手 ----------
        self.declare_parameter("wave_min_sign_changes", 4)
        self.declare_parameter("wave_min_std", 0.03)

        # ---------- 深度靠近/远离（滞回） ----------
        self.declare_parameter("enable_depth", True)
        self.declare_parameter("near_distance_m", 0.75)  # 小于 near -> 抬臂
        self.declare_parameter("far_distance_m", 0.95)   # 大于 far -> 回位（>near）
        self.declare_parameter("min_safe_distance_m", 0.45)  # 小于该距离：直接 ESTOP（防碰撞）
        self.declare_parameter("depth_required", False)  # True=没有深度就不做靠近/远离

        # ---------- 紧急停止 ----------
        self.declare_parameter("estop_min_dy", 0.10)           # 双手同时举过肩
        self.declare_parameter("estop_release_min_frames", 12) # 双手放下持续帧数 -> 解除

        # ---------- 动作冷却（独立） ----------
        self.declare_parameter("cooldown_wave_sec", 1.0)
        self.declare_parameter("cooldown_turn_sec", 0.8)
        self.declare_parameter("cooldown_raise_sec", 0.8)
        self.declare_parameter("cooldown_prox_sec", 1.2)
        self.declare_parameter("cooldown_home_sec", 0.3)

        # ---------- 读取参数 ----------
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.action_topic = str(self.get_parameter("action_topic").value)

        self.window_size = int(self.get_parameter("window_size").value)
        self.min_quality = float(self.get_parameter("min_quality").value)
        self.min_kpt_score = float(self.get_parameter("min_kpt_score").value)

        self.lost_pose_timeout_sec = float(self.get_parameter("lost_pose_timeout_sec").value)

        self.raise_min_dy = float(self.get_parameter("raise_min_dy").value)
        self.turn_min_abs_dx = float(self.get_parameter("turn_min_abs_dx").value)

        self.wave_min_sign_changes = int(self.get_parameter("wave_min_sign_changes").value)
        self.wave_min_std = float(self.get_parameter("wave_min_std").value)

        self.enable_depth = bool(self.get_parameter("enable_depth").value)
        self.near_distance_m = float(self.get_parameter("near_distance_m").value)
        self.far_distance_m = float(self.get_parameter("far_distance_m").value)
        self.min_safe_distance_m = float(self.get_parameter("min_safe_distance_m").value)
        self.depth_required = bool(self.get_parameter("depth_required").value)

        self.estop_min_dy = float(self.get_parameter("estop_min_dy").value)
        self.estop_release_min_frames = int(self.get_parameter("estop_release_min_frames").value)

        self.cooldown_wave_sec = float(self.get_parameter("cooldown_wave_sec").value)
        self.cooldown_turn_sec = float(self.get_parameter("cooldown_turn_sec").value)
        self.cooldown_raise_sec = float(self.get_parameter("cooldown_raise_sec").value)
        self.cooldown_prox_sec = float(self.get_parameter("cooldown_prox_sec").value)
        self.cooldown_home_sec = float(self.get_parameter("cooldown_home_sec").value)

        # ---------- ROS 通信 ----------
        self.sub = self.create_subscription(String, self.pose_topic, self.on_pose, 10)
        self.pub = self.create_publisher(String, self.action_topic, 10)

        # ---------- 内部状态 ----------
        self._last_valid_pose_time = time.time()

        # 用于挥手/转向/抬臂的窗口（以右手为主）
        self._window_relx: List[float] = []   # wrist_x - shoulder_x
        self._window_dx: List[float] = []     # 同 relx，但用于 turn
        self._window_dy: List[float] = []     # shoulder_y - wrist_y

        # 深度滞回状态
        self._prox_state = "FAR"  # FAR/NEAR

        # ESTOP 锁存与释放计数
        self._estop_latched = False
        self._estop_release_counter = 0

        # per-action cooldown 记录
        self._last_sent: Dict[str, float] = {}

        self.get_logger().info(f"action_node started. subscribe={self.pose_topic} publish={self.action_topic}")

    # ---------- publish helpers ----------
    def _publish_action(self, action: str, confidence: float, triggered: bool, reason: str, debug: Dict[str, Any]) -> None:
        payload = {
            "status": "ok",
            "ts_ms": int(time.time() * 1000),
            "action": action,
            "confidence": float(confidence),
            "triggered": bool(triggered),
            "reason": reason,
            "debug": debug,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)

    def _cooldown_ok(self, key: str, cooldown_sec: float) -> bool:
        now = time.time()
        last = self._last_sent.get(key, 0.0)
        if cooldown_sec <= 0.0:
            self._last_sent[key] = now
            return True
        if (now - last) >= cooldown_sec:
            self._last_sent[key] = now
            return True
        return False

    # ---------- wave helper ----------
    def _count_sign_changes(self, xs: List[float]) -> int:
        if len(xs) < 3:
            return 0
        m = _mean(xs)
        centered = [x - m for x in xs]
        eps = 1e-4
        signs: List[int] = []
        for v in centered:
            if abs(v) < eps:
                continue
            signs.append(1 if v > 0 else -1)
        if len(signs) < 3:
            return 0
        changes = 0
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                changes += 1
        return changes

    def _update_window(self, relx: float, dx: float, dy: float) -> None:
        self._window_relx.append(float(relx))
        self._window_dx.append(float(dx))
        self._window_dy.append(float(dy))
        if len(self._window_relx) > self.window_size:
            self._window_relx.pop(0)
            self._window_dx.pop(0)
            self._window_dy.pop(0)

    def _get_person_distance(self, data: Dict[str, Any]) -> Optional[float]:
        depth = data.get("depth", {})
        if not isinstance(depth, dict):
            return None
        if not bool(depth.get("available", False)):
            return None
        d = depth.get("person_distance_m", None)
        if d is None:
            return None
        try:
            return float(d)
        except Exception:
            return None

    # ---------- callback ----------
    def on_pose(self, msg: String) -> None:
        # 1) parse json
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        if data.get("status") != "ok":
            return

        quality = float(data.get("quality", 0.0))
        landmarks = data.get("landmarks", [])

        # 2) fail-safe: low quality / invalid landmarks
        if quality < self.min_quality or not isinstance(landmarks, list) or len(landmarks) < 17:
            if (time.time() - self._last_valid_pose_time) > self.lost_pose_timeout_sec:
                if self._cooldown_ok("HOME_FAILSAFE", self.cooldown_home_sec):
                    self._publish_action(
                        action="HOME",
                        confidence=0.95,
                        triggered=True,
                        reason="lost_pose_or_low_quality",
                        debug={"quality": quality},
                    )
            return

        # 可靠 pose
        self._last_valid_pose_time = time.time()

        # 3) landmarks
        rs = _safe_landmark(landmarks, R_SHOULDER)
        rw = _safe_landmark(landmarks, R_WRIST)
        ls = _safe_landmark(landmarks, L_SHOULDER)
        lw = _safe_landmark(landmarks, L_WRIST)
        if rs is None or rw is None or ls is None or lw is None:
            return

        # visibility filter
        if min(rs["score"], rw["score"], ls["score"], lw["score"]) < self.min_kpt_score:
            return

        # 4) features
        dy_r = rs["y"] - rw["y"]  # >0 means right hand up
        dy_l = ls["y"] - lw["y"]  # >0 means left hand up
        dx_r = rw["x"] - rs["x"]  # >0 right extend, <0 left extend
        relx_r = dx_r

        self._update_window(relx=relx_r, dx=dx_r, dy=dy_r)

        # 5) depth safety (too close -> ESTOP)
        person_d = self._get_person_distance(data)
        if self.enable_depth:
            if person_d is None and self.depth_required:
                pass
            elif person_d is not None:
                if person_d < self.min_safe_distance_m:
                    # 立即 ESTOP（不做 cooldown）
                    if not self._estop_latched:
                        self._estop_latched = True
                        self._estop_release_counter = 0
                    self._publish_action(
                        action="ESTOP",
                        confidence=1.0,
                        triggered=True,
                        reason="too_close_depth_estop",
                        debug={"person_distance_m": person_d},
                    )
                    return

        # 6) ESTOP gesture: both hands up -> latch
        both_hands_up = (dy_r >= self.estop_min_dy) and (dy_l >= self.estop_min_dy)
        if both_hands_up and not self._estop_latched:
            self._estop_latched = True
            self._estop_release_counter = 0
            self._publish_action(
                action="ESTOP",
                confidence=1.0,
                triggered=True,
                reason="both_hands_up_estop",
                debug={"dy_r": dy_r, "dy_l": dy_l, "quality": quality},
            )
            return

        # 7) if latched: only allow release gesture (both hands down)
        if self._estop_latched:
            both_down = (dy_r < self.estop_min_dy * 0.5) and (dy_l < self.estop_min_dy * 0.5)
            if both_down:
                self._estop_release_counter += 1
            else:
                self._estop_release_counter = 0

            if self._estop_release_counter >= self.estop_release_min_frames:
                self._estop_latched = False
                self._estop_release_counter = 0
                # 解除后先 HOME，确保安全
                self._publish_action(
                    action="HOME",
                    confidence=0.95,
                    triggered=True,
                    reason="estop_released_home",
                    debug={},
                )
            return

        # 8) depth proximity control (hysteresis)
        if self.enable_depth and person_d is not None:
            # FAR -> NEAR
            if self._prox_state == "FAR" and person_d < self.near_distance_m:
                self._prox_state = "NEAR"
                if self._cooldown_ok("PROX_RAISE", self.cooldown_prox_sec):
                    conf = min(1.0, 0.60 + (self.near_distance_m - person_d) * 0.9)
                    self._publish_action(
                        action="PROX_RAISE",
                        confidence=conf,
                        triggered=True,
                        reason="person_close_raise",
                        debug={"person_distance_m": person_d},
                    )
                    return

            # NEAR -> FAR
            if self._prox_state == "NEAR" and person_d > self.far_distance_m:
                self._prox_state = "FAR"
                if self._cooldown_ok("PROX_HOME", self.cooldown_prox_sec):
                    conf = min(1.0, 0.60 + (person_d - self.far_distance_m) * 0.9)
                    self._publish_action(
                        action="PROX_HOME",
                        confidence=conf,
                        triggered=True,
                        reason="person_far_home",
                        debug={"person_distance_m": person_d},
                    )
                    return

        # 9) WAVE
        if len(self._window_relx) >= max(8, self.window_size // 2):
            sign_changes = self._count_sign_changes(self._window_relx)
            s = _std(self._window_relx)
            if sign_changes >= self.wave_min_sign_changes and s >= self.wave_min_std:
                if self._cooldown_ok("WAVE", self.cooldown_wave_sec):
                    conf = min(1.0, 0.55 + 0.08 * (sign_changes - self.wave_min_sign_changes) + 2.5 * (s - self.wave_min_std))
                    self._publish_action(
                        action="WAVE",
                        confidence=conf,
                        triggered=True,
                        reason="wave_relx_oscillation",
                        debug={"sign_changes": sign_changes, "std": s},
                    )
                    return

        # 10) TURN
        if len(self._window_dx) >= 6:
            dx_mean = _mean(self._window_dx[-6:])
            if abs(dx_mean) >= self.turn_min_abs_dx:
                if dx_mean > 0:
                    if self._cooldown_ok("TURN_RIGHT", self.cooldown_turn_sec):
                        conf = min(1.0, 0.55 + (abs(dx_mean) - self.turn_min_abs_dx) * 2.0)
                        self._publish_action(
                            action="TURN_RIGHT",
                            confidence=conf,
                            triggered=True,
                            reason="arm_extend_right",
                            debug={"dx_mean": dx_mean},
                        )
                        return
                else:
                    if self._cooldown_ok("TURN_LEFT", self.cooldown_turn_sec):
                        conf = min(1.0, 0.55 + (abs(dx_mean) - self.turn_min_abs_dx) * 2.0)
                        self._publish_action(
                            action="TURN_LEFT",
                            confidence=conf,
                            triggered=True,
                            reason="arm_extend_left",
                            debug={"dx_mean": dx_mean},
                        )
                        return

        # 11) RAISE
        if len(self._window_dy) >= 6:
            dy_mean = _mean(self._window_dy[-6:])
            if dy_mean >= self.raise_min_dy:
                if self._cooldown_ok("RAISE_ARM", self.cooldown_raise_sec):
                    conf = min(1.0, 0.55 + (dy_mean - self.raise_min_dy) * 2.0)
                    self._publish_action(
                        action="RAISE_ARM",
                        confidence=conf,
                        triggered=True,
                        reason="right_hand_up",
                        debug={"dy_mean": dy_mean},
                    )
                    return

        # 默认：不触发，不刷屏


def main() -> None:
    rclpy.init()
    node = ActionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

    