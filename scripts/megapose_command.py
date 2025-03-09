#!/usr/bin/env python3

import rospy
import numpy as np
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage
from visp_megapose.msg import ObjectName, PoseResult
import tf.transformations as tr



def callback(msg):

    global i
    global k

    x = msg.pose.translation
    q = msg.pose.rotation
    if msg.skip == True :
        grasp_pose(x, q,  object_list[i], i)

        while (i < (n_object) and k):
              k = False
              i = i + 1



def grasp_pose(x, q, name, i):
    (r,p,y) = tr.euler_from_quaternion([q.x, q.y, q.z, q.w], 'sxyz')
    RT2 = tr.euler_matrix(r, p, y, 'sxyz')
    RT2[0,3] = x.x
    RT2[1,3] = x.y
    RT2[2,3] = x.z

    # Trasformazione per portare frame sulla superficie dell'oggetto
    if name == 'box1':
      a =  0.063
    elif name == 'box2':
      a = 0.024
    elif name == 'box3':
      a = 0.049
    elif name == 'box4':
      a = 0.04
    elif name == 'box5':
      a = 0.06
    elif name == 'cube':
      a = 0.06
    elif name == 'cubo_verde':
      a = 0.06
    else:
      rospy.logerr("%s is not a valid model", name)


    T = np.array([[    1,   0.0,    0.0,   0.0],
                  [   0.0,    1,    0.0,     a],
                  [   0.0,  0.0,      1,   0.0],
                  [   0.0,  0.0,    0.0,     1]])

    T12 = np.matmul(RT2, T) 

    q12 = tr.quaternion_from_matrix(T12)
    T12 = T12[0:3,3]

    s[i, 0:3] = T12
    s[i, 3:] = q12



if __name__ == '__main__':

    n_object = rospy.get_param("n_object")
    print('Numero di oggetti:' + str(n_object))

    object_list = []
    i = 0
    v = 0
    s = np.zeros((n_object, 7))
    header_frame = 'robot_arm_link0'


    for c in range(n_object):
        object_name = rospy.get_param("object_name_" + str(c+1))
        object_list.append(object_name)
        print('Oggetto numero '+ str(c+1) +': '+ object_name)

    pub = rospy.Publisher('ObjectList', ObjectName, queue_size=10)
    response = rospy.Subscriber('PoseResult', PoseResult, callback)
    pub_tf = rospy.Publisher("/tf", TFMessage, queue_size=10)


    rospy.init_node('command', anonymous=True)



    while not rospy.is_shutdown():
        if (i<n_object):
            k = True
            msg = ObjectName()
            msg.obj_name = object_list[i]
            msg.number = n_object
            rate = rospy.Rate(10) 
            pub.publish(msg)
        
        for v in range(i):
            
            child_frame = object_list[v]
            t1 = geometry_msgs.msg.TransformStamped()
            t1.header.frame_id = header_frame
            t1.header.stamp = rospy.Time.now()
            t1.child_frame_id = "object_" + str(v)
            t1.transform.translation.x = s[v][0]
            t1.transform.translation.y = s[v][1]
            t1.transform.translation.z = s[v][2]
            t1.transform.rotation.x = s[v][3]
            t1.transform.rotation.y = s[v][4]
            t1.transform.rotation.z = s[v][5]
            t1.transform.rotation.w = s[v][6]

            tfm1 = TFMessage([t1])
            pub_tf.publish(tfm1)

        rate.sleep()




