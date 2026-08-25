# active-vision-fep
Pytorch implementation of "Active Vision for Robot Manipulators Using the Free Energy Principle" by Van de Maele et al.

## Installation

The original development environment used Python 3.8 and ROS Noetic. On Ubuntu
20.04 with ROS Noetic already installed, create the Python environment with:

```bash
source /opt/ros/noetic/setup.bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file contains the direct Python dependencies used by the
project. ROS modules such as `rospy`, `sensor_msgs`, and `cv_bridge` come from
the ROS Noetic installation and should not be installed from PyPI. If a local
catkin workspace supplies the robot packages, source its `devel/setup.bash` (or
`install/setup.bash`) before running the data-collection script.

Training does not require ROS. `collect_data.py` does require ROS Noetic and the
robot-specific ROS packages used by that script.
