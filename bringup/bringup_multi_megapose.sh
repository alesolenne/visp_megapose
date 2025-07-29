#!/bin/bash

# Set up the user and path to the catkin workspace
user=$(whoami)
catkin_directory="/home/${user}/catkin_ws"
cd ${catkin_directory}
source devel/setup.bash
cd ${catkin_directory}/src/visp_megapose/bringup

tmux new-session -s megapose -n alessandro -d

tmux split-window -h
tmux split-window -h
tmux select-layout even-horizontal
tmux split-window -v -t 0
tmux split-window -v -t 1
tmux split-window -v -t 2

#Launch the camera node driver
tmux send-keys -t 0 "roslaunch vision_tool vision_camera.launch" C-m
sleep 1.0

#Launch the image resizer node
tmux send-keys -t 1 "roslaunch ros_imresize imresize_color.launch" C-m
sleep 2.0
tmux send-keys -t 2 "roslaunch ros_imresize imresize_depth.launch" C-m
sleep 1.0

#Launch the MegaPose server
tmux send-keys -t 4 "user=${user} &&. /home/${user}/catkin_ws/src/visp_megapose/bringup/megapose_env.sh " C-m
sleep 1.0
tmux send-keys -t 4 "roslaunch visp_megapose megapose_server.launch" C-m
sleep 2.0

#Launch the MegaPose client
tmux send-keys -t 5 "roslaunch visp_megapose megapose_client_command.launch" C-m
sleep 1.0
tmux send-keys -t 3 "roslaunch visp_megapose command.launch" C-m

tmux attach-session -t megapose