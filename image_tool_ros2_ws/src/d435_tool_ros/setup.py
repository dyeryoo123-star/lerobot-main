from setuptools import setup, find_packages
import os
from glob import glob

package_name = "d435_tool_ros"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ROS2 ament index
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package.xml
        ("share/" + package_name, ["package.xml"]),
        # launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dingyu",
    maintainer_email="dingyu@example.com",
    description="D435 + MediaPipe pose + action + LeRobot client nodes",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            
            "pose_node = d435_tool_ros.pose_node:main",
            "action_node = d435_tool_ros.action_node:main",
            "lerobot_client_node = d435_tool_ros.lerobot_client_node:main",
        ],
    },
)