#!/usr/bin/env python3

import rospy
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage
import tf.transformations as tr
import numpy as np
import getpass
import os
import json

def compute_transformation_matrix(T1, q1, position, rotation):
  """Compute the transformation matrix from the camera to the object."""
  RT1 = tr.quaternion_matrix(q1)
  RT1[:3, 3] = T1

  r, p, y = tr.euler_from_quaternion(rotation, 'sxyz')
  RT2 = tr.euler_matrix(r, p, y, 'sxyz')
  RT2[:3, 3] = position[:3]

  return np.matmul(RT1, RT2)

def apply_surface_transformation(T12, object_name):
  """Apply transformation to align the frame with the object's surface."""
  offsets = {
    'box1_1': 0.069, 'box1_2': 0.069, 'box1_3': 0.069, 'box1_6': 0.069, 
    'box2_1': 0.03, 
    'box3_1': 0.06, 'box3_2': 0.06, 'box3_3': 0.06, 'box3_4': 0.06, 'box3_8': 0.06,  
    'box4_1': 0.04,
    'box5_1': 0.06, 'box5_2': 0.06, 'box5_3': 0.06, 'box5_4': 0.06, 'box5_8': 0.06
  }

  offset = offsets.get(object_name)
  if offset is None:
    rospy.logerr("Invalid object name: %s", object_name)
    return None

  T = np.array([[1, 0.0, 0.0, 0.0],
          [0.0, 1, 0.0, offset],
          [0.0, 0.0, 1, 0.0],
          [0.0, 0.0, 0.0, 1]])

  return np.matmul(T12, T)

def load_pose_data(file_path):
  """Load position and rotation data from a JSON file."""
  try:
    with open(file_path, 'r') as f:
      json_data = json.load(f)
      position = np.array(json_data['position'], dtype=np.float64)
      rotation = np.array(json_data['rotation'], dtype=np.float64)
      return position, rotation
  except (KeyError, ValueError, FileNotFoundError) as e:
    rospy.logerr("Error reading pose data from %s: %s", file_path, str(e))
    return None, None

def main():
  rospy.init_node('object_pose')
  pub_tf = rospy.Publisher("/tf", TFMessage, queue_size=10)
  header_frame = 'robot_arm_link0'

  # Camera pose relative to the robot base
  T1 = np.array([-0.468, 0.196, 0.456])
  q1 = [-0.548, 0.063, -0.094, 0.829]

  # Number of objects
  n_object = rospy.get_param("n_object", 1)
  child_frames = []
  poses = np.zeros((n_object, 7))

  user = getpass.getuser()
  base_path = f'/home/{user}/catkin_ws/src/visp_megapose/output/pose/'

  for i in range(n_object):
    object_name = rospy.get_param(f"object_name_{i + 1}", None)
    if not object_name:
      rospy.logerr("Object name parameter 'object_name_%d' is missing", i + 1)
      continue

    rospy.loginfo('Processing object %d: %s', i + 1, object_name)
    file_path = os.path.join(base_path, f'{object_name}_pose.json')

    if not os.path.exists(file_path):
      rospy.logerr("File not found: %s", file_path)
      continue

    position, rotation = load_pose_data(file_path)
    if position is None or rotation is None:
      continue

    T12 = compute_transformation_matrix(T1, q1, position, rotation)
    T12 = apply_surface_transformation(T12, object_name)
    if T12 is None:
      continue

    q12 = tr.quaternion_from_matrix(T12)
    T12 = T12[:3, 3]
    child_frames.append(f'object{i}_pose_0')

    poses[i, :3] = T12
    poses[i, 3:] = q12

  rospy.loginfo('Object poses published on the /tf topic')

  rate = rospy.Rate(100)
  while not rospy.is_shutdown():
    for i, child_frame in enumerate(child_frames):
      t = geometry_msgs.msg.TransformStamped()
      t.header.frame_id = header_frame
      t.header.stamp = rospy.Time.now()
      t.child_frame_id = child_frame
      t.transform.translation.x = poses[i][0]
      t.transform.translation.y = poses[i][1]
      t.transform.translation.z = poses[i][2]
      t.transform.rotation.x = poses[i][3]
      t.transform.rotation.y = poses[i][4]
      t.transform.rotation.z = poses[i][5]
      t.transform.rotation.w = poses[i][6]
      pub_tf.publish(TFMessage([t]))
    rate.sleep()

if __name__ == '__main__':
  main()
