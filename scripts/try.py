#!/usr/bin/env python3

import rospy
from visp_megapose.msg import Result
import rospy
import numpy as np
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage
import tf.transformations as tr

def callback(data):
    # Adjust frame to the surface of the object
    object_offsets_x = {
        'box1_1': 0.063, 'box1_6': 0.063, 'box1_3': 0.063, 'box1_2' : 0.063,
        'box2': 0.024, 'box3': 0.049, 'box4': 0.04,
        'box5': 0.06, 'cube': 0.06, 'cubo_verde': 0.06
    }

    object_offsets_z = {
        'box1_1': 0.063, 'box1_6': 0.063, 'box1_3': 0.063, 'box1_2' : 0.02,
        'box2': 0.024, 'box3': 0.049, 'box4': 0.04,
        'box5': 0.06, 'cube': 0.06, 'cubo_verde': 0.06
    }

    n_box_x = {
        'box1_1': 1, 'box1_6': 3, 'box1_3': 3, 'box1_2' : 4,
    }

    n_box_z = {
        'box1_1': 1, 'box1_6': 2, 'box1_3': 0, 'box1_2' : 4,
    }

    offset_x = object_offsets_x.get(data.name, None)
    offset_z = object_offsets_z.get(data.name, None)

    box_x = int(n_box_x.get(data.name, 0))
    box_z = int(n_box_z.get(data.name, 0))

    if offset_x is None or offset_z is None or box_x is None or box_z is None:
        rospy.logerr(f"{data.name} is not a valid model")
        return

    translation = data.pose.translation
    rotation = data.pose.rotation

    transformation = generate_grasp_pose(translation, rotation, box_x, box_z, offset_x, offset_z, 0, 0)

    tf = rospy.Publisher("/tf", TFMessage, queue_size=10)

    for i in range(transformation.shape[1] // 4):
        matrix = transformation[:, i * 4:(i + 1) * 4]
        translation = matrix[:3, 3]
        rotation = tr.quaternion_from_matrix(matrix)
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "world"
        transform.child_frame_id = f"grasp_pose_{i}"
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]
        tf_message = TFMessage([transform])
        tf.publish(tf_message)


def generate_grasp_pose(translation, rotation, box_x, box_z, offset_x, offset_z, object_name, index):
  # Convert quaternion to Euler angles
  roll, pitch, yaw = tr.euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w], 'sxyz')
  transformation_matrix = tr.euler_matrix(roll, pitch, yaw, 'sxyz')
  transformation_matrix[0, 3] = translation.x
  transformation_matrix[1, 3] = translation.y
  transformation_matrix[2, 3] = translation.z

  transformation_matrix = np.eye(4)
  adjustment_matrix_x = np.array([
    [1, 0.0, 0.0, 0.0],
    [0.0, 1, 0.0, 0.0],
    [0.0, 0.0, 1, 0.0],
    [0.0, 0.0, 0.0, 1]
  ])

  adjustment_matrix_z = np.array([
    [1, 0.0, 0.0, 0.0],
    [0.0, 1, 0.0, 0.0],
    [0.0, 0.0, 1, 0.0],
    [0.0, 0.0, 0.0, 1]
  ])

  result_matrix = np.zeros((4, int(box_x) * int(box_z) * 4))

  for i in range(box_x):
        if i != 0:
            adjustment_matrix_x[0, 3] += offset_x

        for c in range(box_z):
            if c != 0:
                adjustment_matrix_z[2, 3] += offset_z
            result = np.matmul(np.matmul(transformation_matrix, adjustment_matrix_z), adjustment_matrix_x)
            result_matrix[:, (c * 4 + 4*i*box_z ):((c + 1) * 4 + 4 * i* box_z)] = result

            if (c == box_z -1):
                adjustment_matrix_z = np.eye(4)
  
  return result_matrix

if __name__ == '__main__':

    transformation = np.zeros((4, 36))
    try:
        rospy.init_node('result_subscriber', anonymous=True)
        rospy.Subscriber('Result', Result, callback)  # Replace String with the appropriate message type
        rospy.spin()



    except rospy.ROSInterruptException:
        pass