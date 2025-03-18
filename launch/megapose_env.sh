#!/bin/bash          

# Set up the environment for the MegaPose ROS package
# This script should be sourced before running the launch file for server
# Set up your user name here

user=$(whoami) #Set if launch file withouth tmux
conda_directory="/home/${user}/miniconda3"

if [ ! -d "${conda_directory}/envs/megapose" ]; then
    echo "Conda not found at ${conda_directory}/envs/megapose. Please install conda and create a conda environment named 'megapose' before running this script."
    return
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$("${conda_directory}/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${conda_directory}/etc/profile.d/conda.sh" ]; then
        . "${conda_directory}/etc/profile.d/conda.sh"
    else
        export PATH="${conda_directory}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate megapose
export LD_LIBRARY_PATH=${conda_directory}/envs/megapose/lib:$LD_LIBRARY_PATH
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'