#!/usr/bin/env python3

import rospy
from visp_megapose.msg import Object
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose

def printa(msg):
                print(msg)


if __name__ == '__main__':
    try:

        while not rospy.is_shutdown():
            
            pub = rospy.Publisher('new_obj', Object, queue_size=10)
            rospy.init_node('command', anonymous=True)
            msg = Object()
            msg.obj_name = 'box1'
            msg.new_msg = True
            rate = rospy.Rate(1) 
            pub.publish(msg)
            rate.sleep()

            response = rospy.Subscriber('result', Pose, printa)

    except rospy.ROSInterruptException:
        pass