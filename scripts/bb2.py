#!/usr/bin/env python3
import rospy
from visp_megapose.msg import BB3D
import numpy as np
from pytransform3d.rotations import matrix_from_quaternion
import cv2

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

    r_vec = np.array([0.0, 0.0, 0.0])
    t_vec = np.array([0.0, 0.0, 0.0])

    coords = np.array([p1_c[0:3], p2_c[0:3], p3_c[0:3], p4_c[0:3], p5_c[0:3], p6_c[0:3], p7_c[0:3], p8_c[0:3],])

    cam_matrix = np.array([[385.76629638671875, 0, 313.4191589355469], [0, 385.35888671875, 244.85601806640625], [0, 0, 1]])
    distortion = np.array([-0.05377520993351936, 0.06127079576253891, -0.00161056499928236, 0.0007866412051953375, -0.019219011068344116])

    points_2d = cv2.projectPoints(coords, r_vec, t_vec, cam_matrix, distortion)[0]

    u1_p = points_2d[0][0][0]
    u2_p = points_2d[1][0][0]
    u3_p = points_2d[2][0][0]
    u4_p = points_2d[3][0][0]
    u5_p = points_2d[4][0][0]
    u6_p = points_2d[5][0][0]
    u7_p = points_2d[6][0][0]
    u8_p = points_2d[7][0][0]

    v1_p = points_2d[0][0][1]
    v2_p = points_2d[1][0][1]
    v3_p = points_2d[2][0][1]
    v4_p = points_2d[3][0][1]
    v5_p = points_2d[4][0][1]
    v6_p = points_2d[5][0][1]
    v7_p = points_2d[6][0][1]
    v8_p = points_2d[7][0][1]

    u_p_min =  min(u1_p, u2_p, u3_p, u4_p, u5_p ,u6_p ,u7_p, u8_p)
    v_p_min =  min(v1_p, v2_p, v3_p, v4_p, v5_p ,v6_p ,v7_p, v8_p)

    u_p_max =  max(u1_p, u2_p, u3_p, u4_p, u5_p ,u6_p ,u7_p, u8_p)
    v_p_max =  max(v1_p, v2_p, v3_p, v4_p, v5_p ,v6_p ,v7_p, v8_p)

    print(v_p_min, u_p_min)
    print(v_p_max, u_p_max)

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