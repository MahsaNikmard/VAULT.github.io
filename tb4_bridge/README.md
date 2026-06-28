# TurtleBot4 02493 Namespace Bridge

Bridges `/Turtlebot_02493/*` namespaced topics to standard namespace so that
VfhPlus can consume them without modification.

## Quick Start

```bash
# Terminal 1: Run the bridge
export ROS_DOMAIN_ID=3
cd ~/VfhPlus/tb4_bridge
./run_bridge.sh

# Terminal 2: Launch RViz with pre-configured display
export ROS_DOMAIN_ID=3
rviz2 -d ~/VfhPlus/tb4_bridge/tb4_rviz.rviz
```

## What It Does

| Namespaced (robot)                              | Standard (your code)           | QoS        | Direction   |
|-------------------------------------------------|-------------------------------|------------|-------------|
| `/Turtlebot_02493/tf`                            | `/tf`                          | Best Effort | robot → std |
| `/Turtlebot_02493/tf_static`                     | `/tf_static`                   | Reliable/TL | robot → std |
| `/Turtlebot_02493/oakd/rgb/preview/image_raw`    | `/oakd/rgb/preview/image_raw`  | Best Effort | robot → std |
| `/Turtlebot_02493/oakd/rgb/preview/camera_info`  | `/oakd/rgb/preview/camera_info`| Best Effort | robot → std |
| `/Turtlebot_02493/stereo/depth`                  | `/stereo/depth`                | Best Effort | robot → std |
| `/Turtlebot_02493/scan`                          | `/scan`                        | Reliable    | robot → std |
| `/Turtlebot_02493/odom`                          | `/odom`                        | Best Effort | robot → std |
| `/Turtlebot_02493/imu`                           | `/imu`                         | Best Effort | robot → std |
| `/Turtlebot_02493/robot_description`             | `/robot_description`           | Reliable/TL | robot → std |
| `/Turtlebot_02493/joint_states`                  | `/joint_states`                | Best Effort | robot → std |
| `/cmd_vel`                                       | `/Turtlebot_02493/cmd_vel`     | Reliable    | std → robot |
| `/Turtlebot_02493/hazard_detection`              | `/hazard_detection`            | Best Effort | robot → std |

## Camera Not Publishing?

The OAK-D driver uses lazy publishers — it only starts streaming when someone subscribes.
The bridge subscribes to the image topics, which should trigger the camera to start publishing.
If images still don't appear:

1. SSH into the robot: `ssh ubuntu@<robot_ip>`
2. Check the OAK-D node: `ros2 node info /Turtlebot_02493/oakd`
3. If node is missing, restart the TB4 bringup:
   ```bash
   # On the robot (Raspberry Pi)
   sudo systemctl restart turtlebot4.service
   ```
4. Check USB connection to OAK-D camera:
   ```bash
   lsusb | grep Movidius
   ```
   If no Movidius device appears, physically re-seat the USB-C cable to the OAK-D.

## Dependencies

Standard ROS2 Humble packages (should all be installed already):
- `rclpy`, `tf2_msgs`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `std_msgs`
- Optional: `irobot_create_msgs` (for hazard detection relay)

## Files

- `tb4_bridge_node.py` — The bridge node (standalone, no colcon build needed)
- `tb4_bridge.launch.py` — ROS2 launch file
- `run_bridge.sh` — Quick-start shell script
- `bridge_config.yaml` — Reference config (human-readable topic mapping)
- `tb4_rviz.rviz` — Pre-configured RViz layout using bridged topics
