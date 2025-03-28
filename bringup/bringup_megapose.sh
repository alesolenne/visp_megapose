#!/bin/bash

# Set up the user and path to the catkin workspace
user=$(whoami)
catkin_directory="/home/${user}/catkin_ws"
cd ${catkin_directory}
source devel/setup.bash

tmux new-session -s megapose -n alessandro -d

tmux split-window -h
tmux split-window -h
tmux select-layout even-horizontal
tmux split-window -v -t 0
tmux split-window -v -t 1

#Launch the camera node driver
tmux send-keys -t 0 "roslaunch realsense2_camera rs_camera.launch align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30" C-m
sleep 1.0

#Launch the image resizer node
tmux send-keys -t 1 "roslaunch ros_imresize imresize_color.launch" C-m
sleep 2.0
tmux send-keys -t 2 "roslaunch ros_imresize imresize_depth.launch" C-m
sleep 1.0

#Launch the MegaPose server
tmux send-keys -t 3 "user=${user} &&. /home/${user}/catkin_ws/src/visp_megapose/bringup/megapose_env.sh " C-m
sleep 1.0
tmux send-keys -t 3 "roslaunch visp_megapose megapose_server.launch" C-m
sleep 1.0

#Launch the MegaPose client
tmux send-keys -t 4 "roslaunch visp_megapose megapose_client.launch" C-m

tmux attach-session -t megapose