#!/usr/bin/env python3

import rospy
import numpy as np
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage
from visp_megapose.msg import ObjectName, PoseResult, Result
import tf.transformations as tr


def callback(msg):
  global current_object_index
  global process_next_object

  # Extract pose information
  translation = msg.pose.translation
  rotation = msg.pose.rotation

  if msg.skip:
    generate_grasp_pose(translation, rotation, object_list[current_object_index], current_object_index)

    while current_object_index < num_objects and process_next_object:
      process_next_object = False
      rospy.logwarn(f"Object found and grasp pose for {object_list[current_object_index]} generated!")
      current_object_index += 1

      if current_object_index < num_objects:
        rospy.loginfo(f"Publish/Generate the bounding box for {object_list[current_object_index]}!")


def generate_grasp_pose(translation, rotation, object_name, index):
  # Convert quaternion to Euler angles
  roll, pitch, yaw = tr.euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w], 'sxyz')
  transformation_matrix = tr.euler_matrix(roll, pitch, yaw, 'sxyz')
  transformation_matrix[0, 3] = translation.x
  transformation_matrix[1, 3] = translation.y
  transformation_matrix[2, 3] = translation.z

  # Adjust frame to the surface of the object
  object_offsets = {
    'box1_1': 0.063, 'box1_6': 0.063, 'box1_3': 0.063,
    'box2': 0.024, 'box3': 0.049, 'box4': 0.04,
    'box5': 0.06, 'cube': 0.06, 'cubo_verde': 0.06
  }
  offset = object_offsets.get(object_name, None)

  if offset is None:
    rospy.logerr(f"{object_name} is not a valid model")
    return

  adjustment_matrix = np.array([
    [1, 0.0, 0.0, 0.0],
    [0.0, 1, 0.0, offset],
    [0.0, 0.0, 1, 0.0],
    [0.0, 0.0, 0.0, 1]
  ])

  final_transformation = np.matmul(transformation_matrix, adjustment_matrix)
  quaternion = tr.quaternion_from_matrix(final_transformation)
  position = final_transformation[0:3, 3]

  # Store the transformation
  object_poses[index, 0:3] = position
  object_poses[index, 3:] = quaternion


if __name__ == '__main__':
  rospy.init_node('command', anonymous=True)

  # Get the number of objects and their names
  num_objects = rospy.get_param("n_object")
  rospy.loginfo(f"Number of objects: {num_objects}")

  object_list = [rospy.get_param(f"object_name_{i+1}") for i in range(num_objects)]
  for idx, obj_name in enumerate(object_list, start=1):
    rospy.loginfo(f"Object number {idx}: {obj_name}")

  rospy.loginfo(f"Publish/Generate the bounding box for {object_list[0]}!")

  # Initialize variables
  current_object_index = 0
  process_next_object = True
  object_poses = np.zeros((num_objects, 7))
  header_frame = 'robot_arm_link0'

  # Publishers and subscribers
  object_publisher = rospy.Publisher('ObjectList', ObjectName, queue_size=10)
  object_result = rospy.Publisher('Result', Result, queue_size=10)
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
      transform = geometry_msgs.msg.TransformStamped()
      transform.header.frame_id = header_frame
      transform.header.stamp = rospy.Time.now()
      transform.child_frame_id = f"object_{idx}"
      transform.transform.translation.x = object_poses[idx][0]
      transform.transform.translation.y = object_poses[idx][1]
      transform.transform.translation.z = object_poses[idx][2]
      transform.transform.rotation.x = object_poses[idx][3]
      transform.transform.rotation.y = object_poses[idx][4]
      transform.transform.rotation.z = object_poses[idx][5]
      transform.transform.rotation.w = object_poses[idx][6]

      tf_message = TFMessage([transform])
      tf_publisher.publish(tf_message)

      b = Result()
      b.name = object_list[idx]
      b.pose.translation.x = object_poses[idx][0]
      b.pose.translation.y = object_poses[idx][1]
      b.pose.translation.z = object_poses[idx][2]
      b.pose.rotation.x = object_poses[idx][3]
      b.pose.rotation.y = object_poses[idx][4]
      b.pose.rotation.z = object_poses[idx][5]
      b.pose.rotation.w = object_poses[idx][6]

      object_result.publish(b)

    rate.sleep()