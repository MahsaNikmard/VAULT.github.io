# VAULT: Vision-Aware Unified Layer for safe Traversal

VAULT is a two-layer runtime safety framework for learned robot navigation. It
pairs a learned navigation policy (NoMaD) with a classical reactive avoider
(VFH*) that works on top of monocular metric depth (Depth Anything V2). The
learned policy drives goal directed behaviour, and the reactive layer checks
every proposed action against a depth based obstacle map, overriding it only
when it would violate a local safety constraint.

This repository is the artifact for the paper *When Should a Robot Stop Learning
and Start Reasoning?*. It contains the deployment code used on a real
TurtleBot4, a Webots simulation of the same pipeline, and the evaluation scripts
that reproduce the safety comparison between VAULT and the NoMaD only baseline.

Project page: https://mahsanikmard.github.io/VAULT.github.io/

Archived release (Zenodo): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21009846.svg)](https://doi.org/10.5281/zenodo.21009846)

Demo video: [videos/Introduction_vault.mp4](videos/Introduction_vault.mp4) (also
embedded on the project page).


## What you can do with this artifact

There are two ways to use the code.

1. Simulation. Run the full NoMaD plus Depth Anything V2 plus VFH* loop against a
   Webots simulation arena with a single command. No robot hardware is required.
   This is the path used by the artifact evaluation, and it reproduces the
   central result of the paper, namely that VAULT travels much farther without
   collisions than the learned policy alone.

2. Real robot. Deploy the same pipeline on a TurtleBot4 running ROS 2 Humble with
   a Luxonis OAK-D camera. This reproduces the hardware experiments reported in
   the paper.

A reviewer who only wants to check the claims should follow the simulation path.


## System requirements

Tested configuration:

* Ubuntu 22.04
* ROS 2 Humble
* Webots R2023b or newer (tested with R2025a), for the simulation path
* An NVIDIA GPU with CUDA (tested with an RTX 4060)
* About 6 GB of free disk space for the code and model weights

A GPU is strongly recommended. The depth model and the diffusion policy run in
real time on the GPU, and Webots renders the camera much faster with GPU
acceleration. The simulation also runs without a GPU using software rendering,
but each episode then takes roughly ten times longer.

For the real robot path you additionally need an iRobot TurtleBot4 with a
front facing Luxonis OAK-D camera.


## Repository layout

```
deployment/            Real robot pipeline (the code that runs on the TurtleBot4)
  src/
    explore_vfh.py     Exploration node (NoMaD proposals checked by VFH*)
    pd_controller.py   Single rate PD controller
    utils.py           NoMaD checkpoint loader
    VfhPlus/           Core library (vfh_star, depth_processing, depth_markers, ...)
    Object_detection/  Goal oriented node with a YOLOv8 target state machine
  config/              Robot, camera, model, VFH* and SLAM configuration
  model_weights/       NoMaD and Depth Anything V2 weights (included)
simulation/            Webots simulation and evaluation
  worlds/              multi_robot.wbt (the arena), plus warehouse.wbt
  src/
    webots_controller.py  Webots to ROS 2 bridge with ground-truth collision logging
  config/              Simulation robot and VFH* configuration
  run_sim.sh           One command launcher for a single run
  run_trials.sh        Repeated trial evaluation (VAULT vs NoMaD)
  aggregate_trials.py  Summarises the trials into a results table
train/                 NoMaD configuration and the vint_train package
third_party/           Vendored diffusion_policy modules
Depth-Anything-V2/     Vendored Depth Anything V2 (metric variant)
intrinsic/             Camera intrinsics for the robot and the simulation
tb4_bridge/            ROS 2 bridge and RViz layout for the physical TurtleBot4
run_exploration.sh     Real robot exploration launcher
run_navigation.sh      Real robot goal oriented launcher
```

The navigation backbone (NoMaD and Depth Anything V2) and the diffusion policy
modules are vendored under `train/`, `Depth-Anything-V2/` and `third_party/`, so
no extra clones are needed. The model weights are already in
`deployment/model_weights`.


## Installation

1. Install Webots and ROS 2 Humble using their official instructions.

2. Create the Python environment from the file in this repository.

   ```
   conda env create -f environment.yml
   conda activate vault
   ```

3. Source ROS 2 in the same shell before running anything.

   ```
   source /opt/ros/humble/setup.bash
   ```

The NoMaD and Depth Anything V2 weights are already included under
`deployment/model_weights`, so no extra download step is needed. For the goal
oriented real robot mode you also need a YOLOv8 weights file (for example
`yolov8n.pt`), passed on the command line as shown below.


## Quick start: run the simulation

From the repository root, with the environment active and ROS 2 sourced:

```
bash simulation/run_sim.sh --gpu
```

This opens the simulation arena, starts the Webots to ROS 2 bridge, and runs the
VAULT navigation loop. The robot explores while VFH* checks every action against
the depth based obstacle map and overrides it when needed. Add `--gui` to watch
the simulation in the Webots window, or drop `--gpu` to run with software
rendering.

To run the learned policy without the safety layer (the baseline), use:

```
bash simulation/run_sim.sh --method nomad --gpu
```

Useful options:

* `--method vault` or `--method nomad` selects the system under test
* `--gpu` renders on the GPU instead of software rendering
* `--gui` shows the Webots window
* `--rviz` opens RViz with the depth fan, the NoMaD and VFH* bins, and the camera
  overlay (needs a display, so combine it with `--gui` or `--gpu`)
* `--duration 120` sets the episode length in seconds (0 runs until Ctrl+C)


## Reproducing the paper result

The headline claim is that VAULT achieves a much larger safe traversal distance
than the NoMaD only baseline. To reproduce it, run several trials of each method
and aggregate the ground truth metrics:

```
bash simulation/run_trials.sh --trials 5 --gpu
```

This runs each method five times and records, for every trial, the metrics
reported in the paper: the total path traversed, the number of collisions in the
run, the time to the first collision, and the distance travelled before the
first collision. It then writes a summary to `results/trend_results.md` and
`results/trend_results.csv`.

The metrics are measured from the simulator using the true robot pose and the
known obstacle layout, so they reflect actual collisions rather than perceived
depth. Time and distance to first collision are only defined for runs that
collide, so they read `n/a` when a method completes a run without any collision.

Five trials per method in the default arena give:

| Method | Trials | Traversed path (m) | Collisions | Time to first collision (s) | Distance to first collision (m) |
|--------|:------:|:------------------:|:----------:|:---------------------------:|:-------------------------------:|
| vault  | 5      | 34.35 ± 2.34 | 0.00 ± 0.00 | n/a | n/a |
| nomad  | 5      | 12.93 ± 7.25 | 2.00 ± 0.63 | 35.3 ± 4.4 | 9.07 ± 1.81 |

VAULT covers more than twice the distance and completes every run without a
single collision, while the baseline collides about twice per run and travels
less far. The absolute numbers vary with the start position and the random
sampling of the diffusion policy, but the ordering is consistent across runs.
These simulation numbers reproduce the ordering of the hardware results in
Table I of the paper. They are not expected to match the physical values
exactly, since those were measured on a real robot in different environments.


## Running on the real robot

The deployment scripts assume a TurtleBot4 with a front facing OAK-D camera and
ROS 2 Humble. Both scripts open four gnome-terminal panes: the camera bridge, the
controller, the navigation node, and RViz.

Exploration, target agnostic, pure NoMaD plus VFH*:

```
bash run_exploration.sh
```

Goal oriented, NoMaD plus VFH* plus a YOLOv8 target lock:

```
bash run_navigation.sh
```

The goal oriented launcher expects a YOLOv8 weights file. Edit the
`--yolo-weights` path near the bottom of `run_navigation.sh`, and set the target
class with `--yolo-classes` (for example 56 for a chair, 0 for a person). To stop
everything, close the four terminal windows or press Ctrl+C in each pane.

Both launchers activate the `vault` conda environment, source ROS 2, set
`PYTHONPATH` to the vendored packages, and change into the repository root. If
your setup differs, adjust the `SETUP` line at the top of each script.


## Visualization in RViz

In simulation, add `--rviz` to `run_sim.sh` to open RViz with the
`simulation/config/webots_rviz.rviz` layout. On the real robot the launchers
open `tb4_bridge/tb4_rviz.rviz`. The preconfigured displays are:

| Display | Topic | What it shows |
|---|---|---|
| RobotModel, TF, Map, LaserScan | standard | base TurtleBot4 view |
| MarkerArray | `/vfh/depth_markers` | full polar depth fan, one arrow per bin |
| MarkerArray | `/vfh/nomad_reference_bins` | NoMaD trajectory reference rays |
| MarkerArray | `/vfh/chosen_bin` | the bin VFH* selected |
| MarkerArray | `/vfh/goal_reference_bins` | goal direction bins (goal mode only) |
| MarkerArray | `/vfh/detected_objects_ray` | YOLO detection bins (goal mode only) |
| Image | `/robot2/trajectory_viz` | camera view with the NoMaD and VFH* overlay |


## Recording a demo video

Record the camera overlay published by the pipeline:

```
ros2 run image_view video_recorder --ros-args \
  -r image:=/robot2/trajectory_viz \
  -p filename:=trajectory.avi -p fps:=8.0 -p codec:=MJPG
```

RViz has no built in recorder, so capture its window with a screen grabber:

```
WID=$(xdotool selectwindow)
ffmpeg -f x11grab -framerate 30 -window_id $WID \
       -c:v libx264 -pix_fmt yuv420p rviz_scene.mp4
```


## Configuration and reuse

The behaviour of the safety layer is controlled through configuration files,
with no code changes required.

* `deployment/config/vfh.yaml` and `simulation/config/vfh.yaml` set the VFH*
  parameters: the safety threshold, the sensing range, the robot radius, the
  number of angular bins, the cost weights mu1, mu2 and mu3, and the depth model
  settings (encoder, maximum depth, weights path).
* `deployment/config/robot.yaml` and `simulation/config/robot.yaml` set the
  velocity limits max_v and max_w, the control rate, and the camera intrinsics
  path.
* `deployment/config/models.yaml` points to the NoMaD configuration and
  checkpoint, and to the YOLO settings.
* `train/config/nomad.yaml` describes the NoMaD architecture, read by
  `deployment/src/utils.py` when loading the checkpoint.
* `deployment/config/camera_front.yaml`, `camera_reverse.yaml` and `slam.yaml`
  configure the cameras and the SLAM toolbox on the real robot.

## Adapting to other robots, cameras, and worlds

VAULT is meant to be reused. The three things you are most likely to change are
the robot and camera, the simulation world, and the reference path provider. None
of them require touching the safety layer code.

### A different robot or camera

The pipeline needs a forward facing RGB camera and a way to drive the base. Two
things must match the new hardware, and both are read from the YAML configs:

* Camera intrinsics. The metric depth and the per bin distances depend on the
  camera calibration, so it must be correct for your camera. The reliable and
  reproducible way to obtain it is to read it straight from the camera driver,
  which publishes a `sensor_msgs/CameraInfo` message on a `.../camera_info`
  topic:

  ```
  ros2 topic echo --once /your_camera/camera_info
  ```

  Take the 3x3 matrix from the `k` field, save it as a NumPy array (for example
  `intrinsic/<robot>/intrinsics.npy`), and point `intrinsics_path` in
  `robot.yaml` at it. Using the driver published values keeps the calibration
  exactly consistent with the camera that produced the images, which is what
  makes a run reproducible on a new platform.

* Base limits. Set `max_v`, `max_w` and the control rate in `robot.yaml`, and the
  robot radius in `vfh.yaml`, to match the new platform.

### A different world

The simulation is not tied to one map. Drop another Webots world into
`simulation/worlds/` and select it with the `--world` option:

```
bash simulation/run_sim.sh --gpu --world simulation/worlds/your_world.wbt
```

The only requirement is that the world contains the same extern controlled robot
used by `multi_robot.wbt`: an `<extern>` controller, a camera named `oakd_rgb`,
the wheel motors and sensors, and `supervisor TRUE` so the bridge can connect and
read the ground truth poses. Everything else in the scene, the layout, the
obstacles and the walls, is free to change.

### A different reference path

The safety layer is a plug in. NoMaD only supplies the reference path, that is the
candidate directions the robot would like to follow. VFH* takes that reference,
checks it against the depth based obstacle map, and overrides it when it is
unsafe. Nothing in the safety layer is specific to NoMaD: replace the NoMaD
proposal step in `deployment/src/explore_vfh.py` with any other source of
candidate waypoints or directions, such as a different learned policy, a global
planner, or a fixed goal heading, and the same VFH* layer keeps guarding it. The
depth processing turns a metric depth map into a per bin distance vector, which is
the only input the layer needs.


## Limitations

* The accuracy of the safety layer depends on the monocular depth model. A wrong
  depth calibration can make the robot stop too early or pass too close to
  obstacles.
* The recovery behaviour is a fixed three phase sequence. It handles common dead
  ends but does not cover every trapped configuration, for example a U shaped
  obstacle that surrounds the robot after it turns.
* The simulation reproduces the ordering of the hardware results, not the exact
  physical distances. The real environments and the simulation arena differ.
* Software rendering makes the simulation about ten times slower than real time.
  A GPU is recommended for timely evaluation.
* The deployment pipeline is single camera and single GPU. The goal oriented
  node uses a multithreaded executor so that depth and policy inference do not
  block the reactive layer.


## License

This project is released under the MIT License. See the LICENSE file.


## Citation

If you use this artifact, please cite the paper.

```bibtex
@inproceedings{nikmard2026vault,
  author    = {Nikmard, Mahsa and Pelliccione, Patrizio and D'Angelo, Gianlorenzo},
  title     = {When Should a Robot Stop Learning and Start Reasoning?},
  booktitle = {Proceedings of the 2026 IEEE International Conference on
               Autonomic Computing and Self-Organizing Systems (ACSOS)},
  year      = {2026},
  publisher = {IEEE},
  note      = {To appear}
}
```
