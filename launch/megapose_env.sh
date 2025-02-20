#!/bin/bash          
conda activate megapose
cd /home/ws
export LD_LIBRARY_PATH=/root/miniconda3/envs/megapose/lib:$LD_LIBRARY_PATH
source devel/setup.bash

