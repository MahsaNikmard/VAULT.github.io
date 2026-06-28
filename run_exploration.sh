#!/bin/bash
# Target-agnostic exploration on the real TurtleBot4 (NoMaD + Depth Anything V2 + VFH*).
# Opens four terminals: the ROS 2 bridge, the controller, the navigation node, and RViz.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the environment, source ROS 2, and make the vendored packages importable.
SETUP="source ~/miniconda3/etc/profile.d/conda.sh && conda activate vault \
  && source /opt/ros/humble/setup.bash \
  && export ROS_DOMAIN_ID=3 \
  && export PYTHONPATH=\"$ROOT/deployment/src:$ROOT/train:$ROOT/third_party:\$PYTHONPATH\" \
  && cd $ROOT"

# Terminal 1: ROS bridge
gnome-terminal --title="Bridge" -- bash -c "$SETUP && cd $ROOT/tb4_bridge && ./run_bridge.sh; exec bash"

# Terminal 2: Controller
gnome-terminal --title="Controller" -- bash -c "$SETUP && python deployment/src/pd_controller.py --robot turtlebot4 --control vfh; exec bash"

# Terminal 3: Navigation
gnome-terminal --title="Navigation" -- bash -c "$SETUP && python deployment/src/explore_vfh.py --model nomad --robot turtlebot4; exec bash"

# Terminal 4: RViz
gnome-terminal --title="RViz" -- bash -c "$SETUP && cd $ROOT/tb4_bridge && rviz2 -d tb4_rviz.rviz; exec bash"
