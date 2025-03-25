import os                                                        # Provides functions to interact with the operating system
import getpass                                                   # Provides access to the user's login name
from pathlib import Path                                         # Object-oriented filesystem paths
import json                                                      # Used for working with JSON data
from typing import Dict, Optional                                # Type hinting for dictionaries and optional values
import numpy as np                                               # Library for numerical computations
import transforms3d                                              # Library for 3D transformations (rotations, translations, etc.)
import pandas as pd                                              # Library for data manipulation and analysis
import torch                                                     # PyTorch library for deep learning
import torch.nn as nn                                            # PyTorch module for neural network layers
from torch.fx.experimental.optimization import fuse              # Function for fusing layers in PyTorch models
import rospy                                                     # ROS Python client library for interacting with ROS nodes
from cv_bridge import CvBridge                                   # ROS library for converting between ROS Image messages and OpenCV images