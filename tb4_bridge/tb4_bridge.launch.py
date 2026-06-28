"""
TurtleBot4 02493 Namespace Bridge — Launch File
================================================
Launches the bridge node that remaps /Turtlebot_02493/* topics
to standard namespace for VfhPlus.

Usage:
    export ROS_DOMAIN_ID=3
    ros2 launch tb4_bridge tb4_bridge.launch.py

    # Or with custom namespace:
    ros2 launch tb4_bridge tb4_bridge.launch.py robot_ns:=/MyOtherRobot
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────
    robot_ns_arg = DeclareLaunchArgument(
        "robot_ns",
        default_value="/Turtlebot_02493",
        description="TurtleBot4 namespace on the network",
    )

    domain_id_arg = DeclareLaunchArgument(
        "domain_id",
        default_value="3",
        description="ROS_DOMAIN_ID for the TurtleBot4",
    )

    # ── Set domain ID ─────────────────────────────────────────
    set_domain_id = SetEnvironmentVariable(
        name="ROS_DOMAIN_ID",
        value=LaunchConfiguration("domain_id"),
    )

    # ── Bridge node ───────────────────────────────────────────
    bridge_node = Node(
        package=None,  # standalone script
        executable=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tb4_bridge_node.py"
        ),
        name="tb4_bridge",
        output="screen",
        emulate_tty=True,
    )

    # ── RViz2 (optional, comment out if not needed) ───────────
    # Uncomment to auto-launch RViz with a config that uses
    # the bridged (standard namespace) topics:
    #
    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     arguments=["-d", os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "tb4_rviz.rviz"
    #     )],
    #     output="screen",
    # )

    return LaunchDescription([
        robot_ns_arg,
        domain_id_arg,
        set_domain_id,
        LogInfo(msg=["Launching TB4 bridge for namespace: ", LaunchConfiguration("robot_ns")]),
        LogInfo(msg=["ROS_DOMAIN_ID: ", LaunchConfiguration("domain_id")]),
        bridge_node,
        # rviz_node,  # uncomment to auto-launch RViz
    ])
