# 人体视觉交互 ROS2 工作空间

本目录保存 `image_tool_ros2_ws` 的源码，作为 LeRobot 机械臂项目的视觉交互模块。未包含原目录中的 `build/`、`install/`、`log/` 和 Python 缓存。

## 模块

- `pose_node`：订阅 RealSense 彩色与对齐深度图，调用人体姿态估计，发布 `/human_pose/raw`。
- `action_node`：根据人体关键点时序识别挥手、抬臂及左右伸手，并处理深度距离、动作冷却与急停状态。
- `lerobot_client_node`：订阅 `/human_action`，通过 TCP 向机械臂执行服务发送预设动作；默认地址为 `127.0.0.1:8765`。
- `src/d435_tool_ros/launch/ros.launch.py`：三个节点的联合启动配置。

## 当前完整性

这是现有源码快照，尚未验证在当前机器上的运行情况。运行还需要 ROS2、RealSense ROS 图像发布节点、`cv_bridge`、NumPy，以及外部库 `d435_tool.mediapipe_pose.pose_estimator` 和 `lerobot_daemon` 服务端；后两项源码未包含在本目录中。

启动文件与节点存在部分参数名称不一致，依赖声明也需要后续补齐。原始 `*_pre.py` 文件作为历史版本保留，当前入口使用不带 `_pre` 的文件。
