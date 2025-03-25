#!/usr/bin/env python3

import rospy
import numpy as np
from visp_megapose.msg import BB3D

def talker():
    pub = rospy.Publisher('BB3D', BB3D, queue_size=10)
    rospy.init_node('talker_bb', anonymous=True)
    rate = rospy.Rate(10) # 10hz
                    
    T = np.array([        0.05470834672451019,
        -0.07370814681053162,
        0.44264769554138184])
    q = np.array([        -0.5669970308206876,
        -0.32195008629041766,
        0.6887277793242483,
        0.3170434591112174])


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