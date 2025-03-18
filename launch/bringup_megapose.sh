#!/bin/bash

# Set up the user and path to the catkin workspace
user=$(whoami)
catkin_directory="/home/${user}/catkin_ws"
cd ${catkin_directory}
source devel/setup.bash

tmux new-session -s megapose -n alessandro -d

tmux split-window -hf 
tmux split-window -vf
tmux split-window -h

#Launch the camera node driver
tmux send-keys -t 0 "roslaunch realsense2_camera rs_camera.launch" C-m
sleep 1.0

#Launch the image resizer node
tmux send-keys -t 1 "roslaunch ros_imresize imresize_color.launch" C-m
sleep 1.0

#Launch the MegaPose server
tmux send-keys -t 2 "user=${user} &&. /home/${user}/catkin_ws/src/visp_megapose/launch/megapose_env.sh " C-m
sleep 1.0
tmux send-keys -t 2 "roslaunch visp_megapose megapose_server.launch" C-m
sleep 1.0

#Launch the MegaPose client
tmux send-keys -t 3 "roslaunch visp_megapose megapose_client.launch" C-m

tmux attach-session -t megapose