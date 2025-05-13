#!/bin/bash

SESSION_1="camera"
SESSION_2="megapose"

# Set up the user and path to the catkin workspace
user=$(whoami)
catkin_directory="/home/${user}/catkin_ws"
cd ${catkin_directory}
source devel/setup.bash

# Percorso al setup ROS e workspace
ROS_SETUP="/opt/ros/noetic/setup.bash"  # Cambia con la tua distro se usi un'altra
WORKSPACE_SETUP="$HOME/catkin_ws/devel/setup.bash"

# --- Sessione bringup DARKO con 3 finestre ---
if ! tmux has-session -t $SESSION_1 2>/dev/null; then
  # Crea la sessione per camera
  tmux new-session -d -s $SESSION_1 -n "cam1" \
    "bash -c 'source $ROS_SETUP && source $WORKSPACE_SETUP && roslaunch realsense2_camera rs_camera.launch align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30'"

  sleep 1.0

  # Crea la finestra manip2
  tmux new-window -t $SESSION_1:1 -n "cam2" \
    "bash -c 'source $ROS_SETUP && source $WORKSPACE_SETUP && roslaunch ros_imresize imresize_color.launch'"

  # Crea la finestra manip3
  tmux new-window -t $SESSION_1:2 -n "cam3" \
    "bash -c 'source $ROS_SETUP && source $WORKSPACE_SETUP && roslaunch ros_imresize imresize_depth.launch'"

  sleep 2.0

  # Seleziona la finestra 'manip1'
  tmux select-window -t $SESSION_1:0
fi

# --- Sessione control DARKO con 2 finestre ---
if ! tmux has-session -t $SESSION_2 2>/dev/null; then
  # Crea la sessione megapose
  tmux new-session -d -s $SESSION_2 -n "meg1"

  # Invia i comandi su due righe: prima attiva l'ambiente, poi lancia roslaunch
  tmux send-keys -t $SESSION_2:0 "user=${user} && . /home/${user}/catkin_ws/src/visp_megapose/bringup/megapose_env.sh" C-m
  sleep 1.0

  tmux send-keys -t $SESSION_2:0 "roslaunch visp_megapose megapose_server.launch" C-m

  sleep 2.0

  # Crea la finestra ctrl2
  tmux new-window -t $SESSION_2:1 -n "meg2" \
    "bash -c 'source $ROS_SETUP && source $WORKSPACE_SETUP && roslaunch visp_megapose megapose_client_command.launch'"

  tmux new-window -t $SESSION_2:2 -n "meg3" \
    "bash -c 'source $ROS_SETUP && source $WORKSPACE_SETUP && roslaunch visp_megapose command.launch'"

  # Seleziona la finestra 'ctrl1'
  tmux select-window -t $SESSION_2:0
fi  

# --- Attacchiamo alla sessione di default ---
# In base alla tua preferenza, puoi scegliere la sessione da attaccare
tmux attach -t $SESSION_1
# tmux attach -t $SESSION_2  # Sostituisci con SESSION_2 se desideri attaccarti a control invece di bringup
