#!/usr/bin/env python3

import rospy
import numpy as np
from visp_megapose.msg import BB3D

def talker():
    pub = rospy.Publisher('BB3D', BB3D, queue_size=10)
    rospy.init_node('talker_bb', anonymous=True)
    rate = rospy.Rate(10) # 10hz
                    
    T = np.array([        -0.07546298950910568,
        -0.010649887844920158,
        0.4491899907588959])
    q = np.array([        -0.29764723297929335,
        0.24537875302813547,
        0.5299172092793448,
        0.7552371439301164])



    bb3 = BB3D()
    bb3.pose.translation.x = T[0]
    bb3.pose.translation.y = T[1]
    bb3.pose.translation.z = T[2]
    bb3.pose.rotation.x = q[0]
    bb3.pose.rotation.y = q[1]
    bb3.pose.rotation.z = q[2]
    bb3.pose.rotation.w = q[3]
    bb3.dim_x = 0.06
    bb3.dim_y = 0.14
    bb3.dim_z = 0.07
    
    while not rospy.is_shutdown():

        pub.publish(bb3)
        rate.sleep()

if __name__ == '__main__':
        talker()