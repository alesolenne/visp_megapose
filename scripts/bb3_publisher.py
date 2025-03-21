#!/usr/bin/env python3

import rospy
import numpy as np
from visp_megapose.msg import BB3D

def talker():
    pub = rospy.Publisher('BB3D', BB3D, queue_size=10)
    rospy.init_node('talker_bb', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    T = np.array([0.1,0.2,0.3,0.4])
    q = np.array([0,0,0,1])

    bb3 = BB3D()
    bb3.pose.translation.x = T[0]
    bb3.pose.translation.y = T[1]
    bb3.pose.translation.z = T[2]
    bb3.pose.rotation.x = q[0]
    bb3.pose.rotation.y = q[1]
    bb3.pose.rotation.z = q[2]
    bb3.pose.rotation.w = q[3]
    bb3.dim_x = 0.2
    bb3.dim_y = 0.3
    bb3.dim_z = 0.4
    
    while not rospy.is_shutdown():

        pub.publish(bb3)
        rate.sleep()

if __name__ == '__main__':
        talker()