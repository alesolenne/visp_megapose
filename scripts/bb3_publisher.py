#!/usr/bin/env python3

import rospy
import numpy as np
from visp_megapose.msg import BB3D

def talker():
    pub = rospy.Publisher('BB3D', BB3D, queue_size=10)
    rospy.init_node('talker_bb', anonymous=True)
    rate = rospy.Rate(10) # 10hz
                    
    T = np.array([        0.059022847563028336,
        -0.07990124821662903,
        0.4462498724460602])
    q = np.array([        -0.4820561923049879,
        -0.35743020093631367,
        0.7496579330081293,
        0.279067128836455])


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