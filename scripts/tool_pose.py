#!/usr/bin/env python3  

import rospy
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage
import tf.transformations as tr
from tf.transformations import quaternion_matrix
import numpy as np

# Transformazione da megapose al tag

T1 = np.array([0, 0.0, -0.042])
q1 = [1, 0, 0, 0 ] # rotazione per allineare frame a quello della SoftHand
R1 = quaternion_matrix(q1) 
R1[0:3,3] = T1

def mult(data):
    if data.transforms[0].child_frame_id=='cube':
         translation_q = data.transforms[0].transform.translation
         orientation_q = data.transforms[0].transform.rotation
         T2= np.array([translation_q.x, translation_q.y, translation_q.z])
         R2=quaternion_matrix([orientation_q.x, orientation_q.y, 
                                        orientation_q.z, orientation_q.w])
         R2[0:3,3] = T2
         a = np.matmul(R2,R1) 
         q = tr.quaternion_from_matrix(a)
         T = a[0:3,3] 
         t = geometry_msgs.msg.TransformStamped()
         t.header.frame_id = "rgb_camera_link"
         t.header.stamp = rospy.Time.now()
         t.child_frame_id = "cubo_new"
         t.transform.translation.x = T[0]
         t.transform.translation.y = T[1]
         t.transform.translation.z = T[2]
         t.transform.rotation.x = q[0]
         t.transform.rotation.y = q[1]
         t.transform.rotation.z = q[2]
         t.transform.rotation.w = q[3]
         tfm = TFMessage([t])
         pub_tf.publish(tfm)

# Publish these transformations
if __name__ == '__main__':
    rospy.init_node('mega_transf')
    pub_tf = rospy.Publisher("/tf", TFMessage, queue_size=10)
    rospy.Subscriber("/tf",TFMessage, mult)
    rospy.spin()
