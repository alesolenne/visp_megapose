#!/usr/bin/env python3

import rospy
import numpy as np
from visp_megapose.msg import BB3D

def talker():
    pub = rospy.Publisher('BB3D', BB3D, queue_size=10)
    rospy.init_node('talker_bb', anonymous=True)
    rate = rospy.Rate(10) # 10hz
                    
    T = np.array([               0.060065142810344696,
        -0.038453008979558945,
        0.41706982254981995])
    q = np.array([               -0.6903986052067901,
        -0.2536955375753227,
        0.5673210717303238,
        0.3703176227424828])


    bb3 = BB3D()
    bb3.pose.translation.x = T[0]
    bb3.pose.translation.y = T[1]
    bb3.pose.translation.z = T[2]
    bb3.pose.rotation.x = q[0]
    bb3.pose.rotation.y = q[1]
    bb3.pose.rotation.z = q[2]
    bb3.pose.rotation.w = q[3]
    bb3.dimensions.x = 0.06
    bb3.dimensions.y = 0.14
    bb3.dimensions.z = 0.07
    
    while not rospy.is_shutdown():

        pub.publish(bb3)
        rate.sleep()

if __name__ == '__main__':
        talker()