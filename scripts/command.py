#!/usr/bin/env python3

import rospy
import numpy as np
from visp_megapose.msg import Object, Result
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose

skip = False

def printa(msg):
    global skip
    global x,q
    x = msg.pose.translation
    q = msg.pose.rotation
    print('running')
    if msg.result == True :
        skip = True
        a = Bool()
        pub2.publish(a)

    

if __name__ == '__main__':

    n_object = rospy.get_param("n_object")

    for i in range(n_object):
        object_name = rospy.get_param("object_name_" + str(i+1))
        print('Oggetto numero '+ str(i+1) +': '+ object_name)

            
    pub = rospy.Publisher('new_obj', Object, queue_size=10)
    response = rospy.Subscriber('result', Result, printa)
    pub2 = rospy.Publisher('ciao', Bool, queue_size=10)


    rospy.init_node('command', anonymous=True)



    while not rospy.is_shutdown():

            for i in range(n_object):
                msg = Object()
                msg.obj_name = 'box1'
                rate = rospy.Rate(1) 
                if (not skip):
                    pub.publish(msg)
                rate.sleep()
