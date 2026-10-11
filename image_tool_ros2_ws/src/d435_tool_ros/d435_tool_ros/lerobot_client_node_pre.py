#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lerobot_client_node.py
=========================================================
ROS2 侧客户端节点：
- 订阅 /human_action（std_msgs/String，内容为 JSON）
- 将动作转换为 preset 指令，通过 TCP 发给本机 lerobot_daemon
"""

import json
import socket
import time
import uuid
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _now_ms() -> int:
    return int(time.time() * 1000)


class LeRobotClientNode(Node):
    def __init__(self) -> None:
        super().__init__("lerobot_client_node")

        # ===== ROS 输入 =====
        self.declare_parameter("action_topic", "/human_action")

        # ===== daemon 地址 =====
        self.declare_parameter("daemon_host", "127.0.0.1")
        self.declare_parameter("daemon_port", 8765)

        # ===== 触发策略 =====
        self.declare_parameter("min_confidence", 0.6)  # 置信度阈值
        self.declare_parameter("cooldown_sec", 1.0)     # 客户端本地冷却（双保险）

        # ===== action -> preset 映射 =====
        self.declare_parameter("preset_for_wave", "wave")
        self.declare_parameter("preset_for_raise_hand", "home")
        self.declare_parameter("preset_for_stop", "stop")

        # ===== wave 参数（可选，会通过 args 发给 daemon）=====
        self.declare_parameter("wave_joint", "Motor_3.pos")
        self.declare_parameter("wave_amp", 25.0)
        self.declare_parameter("wave_repeat", 3)
        self.declare_parameter("wave_step_sleep", 0.35)

        # ===== socket 超时 / 重试 =====
        self.declare_parameter("socket_timeout_sec", 1.5)
        self.declare_parameter("max_retries", 1)

        self.action_topic: str = self.get_parameter("action_topic").value
        self.host: str = self.get_parameter("daemon_host").value
        self.port: int = int(self.get_parameter("daemon_port").value)

        self.min_conf: float = float(self.get_parameter("min_confidence").value)
        self.cooldown_sec: float = float(self.get_parameter("cooldown_sec").value)

        self.preset_wave: str = self.get_parameter("preset_for_wave").value
        self.preset_raise: str = self.get_parameter("preset_for_raise_hand").value
        self.preset_stop: str = self.get_parameter("preset_for_stop").value

        self.wave_joint: str = self.get_parameter("wave_joint").value
        self.wave_amp: float = float(self.get_parameter("wave_amp").value)
        self.wave_repeat: int = int(self.get_parameter("wave_repeat").value)
        self.wave_step_sleep: float = float(self.get_parameter("wave_step_sleep").value)

        self.sock_timeout: float = float(self.get_parameter("socket_timeout_sec").value)
        self.max_retries: int = int(self.get_parameter("max_retries").value)

        self._last_sent_ts: float = 0.0

        self.sub = self.create_subscription(String, self.action_topic, self.on_action, 10)

        self.get_logger().info(
            f"[lerobot_client_node] subscribe={self.action_topic}, daemon={self.host}:{self.port}, "
            f"min_conf={self.min_conf}, cooldown_sec={self.cooldown_sec}"
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

    # ---------- mapping ----------
    def _map_action_to_preset_and_args(self, action: str) -> tuple[Optional[str], Dict[str, Any]]:
        """
        你可以按需求扩展更多动作：
        - action_node 输出 "WAVE" -> preset "wave"
        - "RAISE_HAND" -> "home"
        - "STOP" -> "stop"
        """
        if action == "WAVE":
            return self.preset_wave, {
                "joint": self.wave_joint,
                "amp": self.wave_amp,
                "repeat": self.wave_repeat,
                "step_sleep": self.wave_step_sleep,
            }
        if action == "RAISE_HAND":
            return self.preset_raise, {}
        if action == "STOP":
            return self.preset_stop, {}
        return None, {}

    # ---------- callback ----------
    def on_action(self, msg: String) -> None:
        # 解析 action_node 的 JSON
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        action = str(data.get("action", "IDLE"))
        conf = float(data.get("confidence", 0.0))
        triggered = bool(data.get("triggered", False))

        # 触发沿：只在 triggered=true 时发送
        if not triggered:
            return

        # 置信度过滤
        if conf < self.min_conf:
            self.get_logger().info(f"[lerobot_client_node] ignore low conf: action={action} conf={conf:.2f}")
            return

        # 本地 cooldown（避免连续触发）
        now = time.time()
        if self.cooldown_sec > 0.0 and (now - self._last_sent_ts) < self.cooldown_sec:
            self.get_logger().info("[lerobot_client_node] ignore due to cooldown")
            return

        preset, args = self._map_action_to_preset_and_args(action)
        if preset is None:
            self.get_logger().info(f"[lerobot_client_node] unmapped action: {action}")
            return

        req = {
            "cmd": "preset",
            "name": preset,
            "request_id": str(uuid.uuid4()),
            "ts_ms": _now_ms(),
            "args": args,
        }

        resp = self._send_line_json(req)
        if resp is None:
            return

        ok = bool(resp.get("ok", False))
        status = str(resp.get("status", ""))
        self._last_sent_ts = now

        self.get_logger().info(f"[lerobot_client_node] sent preset={preset}, resp ok={ok} status={status}")


def main() -> None:
    import rclpy
    rclpy.init()
    node = LeRobotClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()