#!/usr/bin/env python3

import rospy
import numpy as np
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage
from visp_megapose.msg import ObjectName, PoseResult
import tf.transformations as tr


def callback(msg):
  """
  Callback function to process pose results and generate grasp poses.
  """
  global current_object_index, process_next_object

  # Extract pose information
  translation = msg.pose.translation
  rotation = msg.pose.rotation

  current_object_name = object_list[current_object_index]
  offsets = object_offsets.get(current_object_name)
  box_count = box_counts.get(current_object_name, (0, 0))

  if offsets is None or box_count == (0, 0):
    rospy.logerr(f"{current_object_name} is not a valid model")
    return

  offset_x, offset_y, offset_z = offsets
  box_x, box_z = box_count

  if msg.skip:
    generate_grasp_pose(translation, rotation, current_object_index, box_x, box_z, offset_x, offset_z, offset_y)

    while current_object_index < num_objects and process_next_object:
      process_next_object = False
      rospy.logwarn(f"Object found and grasp pose for {current_object_name} generated!")
      current_object_index += 1

      if current_object_index < num_objects:
        rospy.loginfo(f"Publish/Generate the bounding box for {object_list[current_object_index]}!")


def generate_grasp_pose(translation, rotation, index, box_x, box_z, offset_x, offset_z, offset_y):
  """
  Generate grasp poses for the given object and store them in the object_poses array.
  """
  # Convert quaternion to Euler angles
  roll, pitch, yaw = tr.euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w], 'sxyz')
  transformation_matrix = tr.euler_matrix(roll, pitch, yaw, 'sxyz')
  transformation_matrix[:3, 3] = [translation.x, translation.y, translation.z]

  for i in range(box_x):
    for j in range(box_z):
      adjustment_matrix = np.eye(4)
      adjustment_matrix[0, 3] = i * offset_x
      adjustment_matrix[1, 3] = offset_y
      adjustment_matrix[2, 3] = j * offset_z

      result = np.matmul(transformation_matrix, adjustment_matrix)
      quaternion = tr.quaternion_from_matrix(result)
      position = result[0:3, 3]

      # Store quaternion and position in a vector
      start_idx = (j + i * box_z) * 7
      object_poses[index, start_idx:start_idx + 3] = position
      object_poses[index, start_idx + 3:start_idx + 7] = quaternion

  # Store the final index for valid poses
  final_idx[index, 0] = (box_x * box_z) * 7


if __name__ == '__main__':
  rospy.init_node('command', anonymous=True)

  # Object-specific offsets and box counts
  object_offsets = {
    'box1_1': (-0.06, 0.069, -0.07), 'box1_2': (-0.06, 0.069, -0.07), 'box1_3': (-0.06, 0.069, -0.07), 'box1_6': (-0.06, 0.069, -0.07), 
    'box2_1': (-0.06, 0.03, -0.08), 
    'box3_1': (-0.035, 0.06, -0.05), 'box3_2': (-0.035, 0.06, -0.05), 'box3_3': (-0.035, 0.06, -0.05), 'box3_4': (-0.035, 0.06, -0.05), 'box3_8': (-0.035, 0.06, -0.05),  
    'box4_1': (-0.055, 0.04, -0.105),
    'box5_1': (-0.045, 0.06, -0.05),  'box5_2': (-0.045, 0.06, -0.05),  'box5_3': (-0.045, 0.06, -0.05),  'box5_4': (-0.045, 0.06, -0.05),  'box5_8': (-0.045, 0.06, -0.05)
  }

  box_counts = {
    'box1_1': (1,1), 'box1_2':(1,2), 'box1_3': (1,3), 'box1_6': (2,3), 
    'box2_1': (1,1), 
    'box3_1': (1,1), 'box3_2': (1,2), 'box3_3': (1,3), 'box3_4': (1,4), 'box3_8':(2,4),  
    'box4_1': (1,1),
    'box5_1': (1,1),  'box5_2': (1,2),  'box5_3': (1,3),  'box5_4': (1,4),  'box5_8': (2,4)
  }

  # Get the number of objects and their names
  num_objects = rospy.get_param("n_object")
  rospy.loginfo(f"Number of objects: {num_objects}")
  object_list = [rospy.get_param(f"object_name_{i+1}") for i in range(num_objects)]

  # Determine the maximum box dimensions
  n_i, n_j = 0, 0
  for idx, obj_name in enumerate(object_list, start=1):
    n_i = max(box_counts.get(obj_name, (0, 0))[0], n_i)
    n_j = max(box_counts.get(obj_name, (0, 0))[1], n_j)
    rospy.loginfo(f"Object number {idx}: {obj_name}")

  rospy.loginfo(f"Publish/Generate the bounding box for {object_list[0]}!")

  # Initialize variables
  current_object_index = 0
  process_next_object = True
  object_poses = np.zeros((num_objects, 7 * n_i * n_j))  # For max n_i * n_j poses
  final_idx = np.zeros((num_objects, 1))
  header_frame = rospy.get_param('camera_tf')

  # Publishers and subscribers
  object_publisher = rospy.Publisher('ObjectList', ObjectName, queue_size=10)
  pose_subscriber = rospy.Subscriber('PoseResult', PoseResult, callback)
  tf_publisher = rospy.Publisher("/tf", TFMessage, queue_size=10)

  rate = rospy.Rate(10)  # 10 Hz

  while not rospy.is_shutdown():
    if current_object_index < num_objects:
      process_next_object = True
      msg = ObjectName()
      msg.obj_name = object_list[current_object_index]
      msg.number = num_objects
      object_publisher.publish(msg)

    # Publish transformations for already processed objects
    for idx in range(current_object_index):
      poses = object_poses[idx]
      valid_poses = poses[:int(final_idx[idx, 0])]

      for i in range(len(valid_poses) // 7):
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.frame_id = header_frame
        transform.header.stamp = rospy.Time.now()
        transform.child_frame_id = f"object_{idx}_pose_{i}"
        transform.transform.translation.x = valid_poses[i * 7 + 0]
        transform.transform.translation.y = valid_poses[i * 7 + 1]
        transform.transform.translation.z = valid_poses[i * 7 + 2]
        transform.transform.rotation.x = valid_poses[i * 7 + 3]
        transform.transform.rotation.y = valid_poses[i * 7 + 4]
        transform.transform.rotation.z = valid_poses[i * 7 + 5]
        transform.transform.rotation.w = valid_poses[i * 7 + 6]
        tf_message = TFMessage([transform])
        tf_publisher.publish(tf_message)

    rate.sleep()