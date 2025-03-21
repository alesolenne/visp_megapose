#!/usr/bin/env python3
import rospy
from visp_megapose.msg import BB3D
import numpy as np
from pytransform3d.rotations import matrix_from_quaternion

def callback(data):
    rospy.loginfo("Bounding box ottenuta")

    dim_x = data.dim_x
    dim_y = data.dim_y
    dim_z = data.dim_z

    p1_x = dim_x / 2
    p2_x = - dim_x / 2

    p1_y = dim_y / 2
    p2_y = - dim_y / 2

    p1_z = dim_z / 2
    p2_z = - dim_z / 2

    p1 = np.array([p1_x, p2_y, p2_z, 1])
    p2 = np.array([p1_x, p1_y, p2_z, 1])
    p3 = np.array([p2_x, p1_y, p2_z, 1])
    p4 = np.array([p1_x, p1_y, p1_z, 1])
    p5 = np.array([p1_x, p2_y, p1_z, 1])
    p6 = np.array([p2_x, p2_y, p2_z, 1])
    p7 = np.array([p2_x, p1_y, p1_z, 1])
    p8 = np.array([p2_x, p2_y, p1_z, 1])

    T = np.identity(4)
    T[:3,3] = [data.pose.translation.x, data.pose.translation.y, data.pose.translation.z]
    x = data.pose.rotation.x
    y = data.pose.rotation.y
    z = data.pose.rotation.z
    w = data.pose.rotation.w
    q_new = [w,x,y,z]   # different order of elements in the representation of the quaternion
    R = matrix_from_quaternion(q_new)
    T[0:3,0:3] = R

    p1_c = np.matmul(T,p1)
    p2_c = np.matmul(T,p2)
    p3_c = np.matmul(T,p3)
    p4_c = np.matmul(T,p4)
    p5_c = np.matmul(T,p5)
    p6_c = np.matmul(T,p6)
    p7_c = np.matmul(T,p7)
    p8_c = np.matmul(T,p8)

    f_x = 0.1 
    f_y = 0.2
    c_x = 0.1
    c_y = 0.02
    
    u1 = f_x * p1_c[0] / p1_c[2] + c_x
    u2 = f_x * p2_c[0] / p1_c[2] + c_x
    u3 = f_x * p3_c[0] / p1_c[2] + c_x
    u4 = f_x * p4_c[0] / p1_c[2] + c_x
    u5 = f_x * p5_c[0] / p1_c[2] + c_x
    u6 = f_x * p6_c[0] / p1_c[2] + c_x
    u7 = f_x * p7_c[0] / p1_c[2] + c_x
    u8 = f_x * p8_c[0] / p1_c[2] + c_x

    v1 = f_y * p1_c[1] / p1_c[2] + c_y
    v2 = f_y * p2_c[1] / p1_c[2] + c_y
    v3 = f_y * p3_c[1] / p1_c[2] + c_y
    v4 = f_y * p4_c[1] / p1_c[2] + c_y
    v5 = f_y * p5_c[1] / p1_c[2] + c_y
    v6 = f_y * p6_c[1] / p1_c[2] + c_y
    v7 = f_y * p7_c[1] / p1_c[2] + c_y
    v8 = f_y * p8_c[1] / p1_c[2] + c_y

    u_min = min(u1, u2, u3, u4, u5 ,u6 ,u7, u8)
    v_min = min(v1, v2, v3, v4, v5 ,v6 ,v7, v8)
    u_max = max(u1, u2, u3, u4, u5 ,u6 ,u7, u8)
    v_max = max(v1, v2, v3, v4, v5 ,v6 ,v7, v8)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('bb2_maker', anonymous=True)

    rospy.Subscriber("BB3D", BB3D, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()