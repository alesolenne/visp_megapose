#!/usr/bin/env python3

import rospy
import numpy as np
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import json
import getpass

def create_bounding_box(seq, T, q, object_name):
    """
    Create a BoundingBox object with the given sequence number, position, and orientation.
    """
    bbox = BoundingBox()
    bbox.header.seq = seq
    bbox.header.stamp = rospy.Time.now()
    bbox.header.frame_id = object_name

    # Set position
    bbox.pose.position.x = T[0]
    bbox.pose.position.y = T[1]
    bbox.pose.position.z = T[2]

    # Set orientation
    bbox.pose.orientation.x = q[0]
    bbox.pose.orientation.y = q[1]
    bbox.pose.orientation.z = q[2]
    bbox.pose.orientation.w = q[3]

    # Set dimensions
    bbox.dimensions.x = 0.06
    bbox.dimensions.y = 0.14
    bbox.dimensions.z = 0.07

    # Set value and label
    bbox.value = 1.0  # Example value
    bbox.label = 1    # Example label

    return bbox

def talker():
    """
    ROS node that publishes a BoundingBoxArray message at a fixed rate.
    """
    pub = rospy.Publisher('/perception/visualiztion/tracked_scene_objects/bboxes3d', BoundingBoxArray, queue_size=10)
    rospy.init_node('talker_bb', anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz

    object_name = "box1_1"

    # Define position (T) and orientation (q)
    # Load T and q from the JSON file

    json_file_path = '/home/' + getpass.getuser() +'/catkin_ws/src/visp_megapose/output/pose/' + object_name + '_pose.json'
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        T = np.array(data['position'])
        q = np.array(data['rotation'])

    seq = 0  # Initialize sequence number

    while not rospy.is_shutdown():
        # Create BoundingBoxArray message
        bb_array = BoundingBoxArray()
        bb_array.header.stamp = rospy.Time.now()
        bb_array.header.frame_id = "map"
        bb_array.header.seq = seq
        bb_array.boxes = []  # Explicitly initialize as an empty list

        # Create and add a single bounding box
        bbox = create_bounding_box(seq, T, q, object_name)
        bb_array.boxes.append(bbox)

        # Publish the message
        pub.publish(bb_array)

        seq += 1  # Increment sequence number
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass