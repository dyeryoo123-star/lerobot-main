#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lerobot_client_node.py
=========================================================
ROS2 侧客户端节点：
- 订阅 /human_action（std_msgs/String，JSON）
- 将 action 映射为 preset 指令，通过 TCP 发给本机 lerobot_daemon

1) per-action cooldown：不同动作独立冷却，不会 wave 后挡住 HOME/ESTOP
2) 安全动作穿透：ESTOP/HOME 永远优先，不受 min_confidence 限制
3) 映射更完整：支持 RAISE_ARM、TURN_LEFT/RIGHT、PROX_RAISE/PROX_HOME
"""

import json
import socket
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _now_ms() -> int:
    return int(time.time() * 1000)


class LeRobotClientNode(Node):
    def __init__(self) -> None:
        super().__init__("lerobot_client_node")

        # ---------- ROS 输入 ----------
        self.declare_parameter("action_topic", "/human_action")

        # ---------- daemon 地址 ----------
        self.declare_parameter("daemon_host", "127.0.0.1")
        self.declare_parameter("daemon_port", 8765)

        # ---------- 触发策略 ----------
        self.declare_parameter("min_confidence", 0.65)

        # ---------- per-action cooldown ----------
        self.declare_parameter("cooldown_wave_sec", 1.0)
        self.declare_parameter("cooldown_turn_sec", 0.8)
        self.declare_parameter("cooldown_raise_sec", 0.8)
        self.declare_parameter("cooldown_home_sec", 0.2)
        self.declare_parameter("cooldown_estop_sec", 0.0)  # ESTOP 不限

        # ---------- wave 参数 ----------
        self.declare_parameter("wave_joint", "Motor_3.pos")
        self.declare_parameter("wave_amp", 25.0)
        self.declare_parameter("wave_repeat", 3)
        self.declare_parameter("wave_step_sleep", 0.35)

        # ---------- raise_arm 参数（由 daemon 执行） ----------
        self.declare_parameter("raise_lift_m1", 25.0)
        self.declare_parameter("raise_elbow_m2", 15.0)
        self.declare_parameter("raise_motion_time_ms", 1200)

        # ---------- turn 参数（由 daemon 执行） ----------
        self.declare_parameter("turn_delta_m0", 20.0)
        self.declare_parameter("turn_motion_time_ms", 900)

        # ---------- socket ----------
        self.declare_parameter("socket_timeout_sec", 1.5)
        self.declare_parameter("max_retries", 1)

        self.action_topic = str(self.get_parameter("action_topic").value)
        self.host = str(self.get_parameter("daemon_host").value)
        self.port = int(self.get_parameter("daemon_port").value)

        self.min_conf = float(self.get_parameter("min_confidence").value)

        self.cool_wave = float(self.get_parameter("cooldown_wave_sec").value)
        self.cool_turn = float(self.get_parameter("cooldown_turn_sec").value)
        self.cool_raise = float(self.get_parameter("cooldown_raise_sec").value)
        self.cool_home = float(self.get_parameter("cooldown_home_sec").value)
        self.cool_estop = float(self.get_parameter("cooldown_estop_sec").value)

        self.wave_joint = str(self.get_parameter("wave_joint").value)
        self.wave_amp = float(self.get_parameter("wave_amp").value)
        self.wave_repeat = int(self.get_parameter("wave_repeat").value)
        self.wave_step_sleep = float(self.get_parameter("wave_step_sleep").value)

        self.raise_lift_m1 = float(self.get_parameter("raise_lift_m1").value)
        self.raise_elbow_m2 = float(self.get_parameter("raise_elbow_m2").value)
        self.raise_motion_time_ms = int(self.get_parameter("raise_motion_time_ms").value)

        self.turn_delta_m0 = float(self.get_parameter("turn_delta_m0").value)
        self.turn_motion_time_ms = int(self.get_parameter("turn_motion_time_ms").value)

        self.sock_timeout = float(self.get_parameter("socket_timeout_sec").value)
        self.max_retries = int(self.get_parameter("max_retries").value)

        self._last_sent: Dict[str, float] = {}

        self.sub = self.create_subscription(String, self.action_topic, self.on_action, 10)

        self.get_logger().info(
            f"[lerobot_client_node] subscribe={self.action_topic}, daemon={self.host}:{self.port}, min_conf={self.min_conf}"
        )

    # ---------- TCP ----------
    def _send_line_json(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        for attempt in range(self.max_retries + 1):
            try:
                with socket.create_connection((self.host, self.port), timeout=self.sock_timeout) as s:
                    s.settimeout(self.sock_timeout)
                    s.sendall(line)

                    buf = b""
                    while b"\n" not in buf:
                        chunk = s.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    if b"\n" not in buf:
                        raise RuntimeError("no newline response")

                    resp_line = buf.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
                    return json.loads(resp_line)

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.1)
                    continue
                self.get_logger().error(f"[lerobot_client_node] send failed: {e}")
                return None
        return None

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

    def _map_action(self, action: str) -> Tuple[Optional[str], Dict[str, Any], float]:
        """
        返回：preset_name, args, cooldown_sec
        """
        if action == "ESTOP":
            return "stop", {}, self.cool_estop

        if action in ("HOME", "PROX_HOME"):
            return "home", {}, self.cool_home

        if action in ("RAISE_ARM", "PROX_RAISE"):
            return "raise_arm", {
                "lift_m1": self.raise_lift_m1,
                "elbow_m2": self.raise_elbow_m2,
                "motion_time_ms": self.raise_motion_time_ms,
                "home_first": False,
            }, self.cool_raise

        if action == "WAVE":
            return "wave", {
                "joint": self.wave_joint,
                "amp": self.wave_amp,
                "repeat": self.wave_repeat,
                "step_sleep": self.wave_step_sleep,
                "home_first": True,
            }, self.cool_wave

        if action == "TURN_LEFT":
            return "turn_left", {
                "delta_m0": self.turn_delta_m0,
                "motion_time_ms": self.turn_motion_time_ms,
                "home_first": False,
            }, self.cool_turn

        if action == "TURN_RIGHT":
            return "turn_right", {
                "delta_m0": self.turn_delta_m0,
                "motion_time_ms": self.turn_motion_time_ms,
                "home_first": False,
            }, self.cool_turn

        return None, {}, 0.0

    # ---------- callback ----------
    def on_action(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        action = str(data.get("action", "IDLE"))
        conf = float(data.get("confidence", 0.0))
        triggered = bool(data.get("triggered", False))

        if not triggered:
            return

        preset, args, cooldown_sec = self._map_action(action)
        if preset is None:
            return

        # 安全动作：永远优先
        is_safety = action in ("ESTOP", "HOME", "PROX_HOME")
        if (not is_safety) and conf < self.min_conf:
            self.get_logger().info(f"[lerobot_client_node] ignore low conf: action={action} conf={conf:.2f}")
            return

        if not self._cooldown_ok(preset, cooldown_sec):
            return

        req = {
            "cmd": "preset",
            "name": preset,
            "request_id": f"ros_{uuid.uuid4().hex[:8]}",
            "ts_ms": _now_ms(),
            "args": args,
        }

        resp = self._send_line_json(req)
        if resp is None:
            return

        ok = bool(resp.get("ok", False))
        status = str(resp.get("status", ""))
        if ok:
            self.get_logger().info(f"[lerobot_client_node] sent preset={preset} action={action} conf={conf:.2f}")
        else:
            self.get_logger().error(f"[lerobot_client_node] daemon error: {status} resp={resp}")


def main() -> None:
    rclpy.init()
    node = LeRobotClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
