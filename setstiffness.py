#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header

if __name__ == '__main__':
    rospy.init_node('set_stiffness')
    pub_tf = rospy.Publisher("/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command", JointTrajectory, queue_size=10)
    x=0.0
    f = 10.0      # Frequenza di spin del nodo
    rate = rospy.Rate(f)

    rospy.loginfo("Inizio della fase di grasp del tool")



    # Inizio della fase di visual servoing
    while not rospy.is_shutdown():

        # Conversione dei dati per controllore di posizione
        q_tolist = [0.2, 0.3]
        dq_tolist = [0.1,0]
        ddq_tolist = [0,0]

        # Pubblica sul topic del controllore il comando
        joints_str = JointTrajectory()
        joints_str.header = Header()
        joints_str.header.stamp = rospy.Duration(0)
        joints_str.joint_names = ['qbmove2_shaft_joint', 'qbmove2_stiffness_preset_virtual_joint']
        point = JointTrajectoryPoint()
        point.positions = q_tolist
        point.velocities = dq_tolist
        point.accelerations = ddq_tolist
        x=x+1/f
        point.time_from_start = rospy.Duration(x)
        joints_str.points.append(point)
        
        pub_tf.publish(joints_str)      # Comando al controllore del robot

        rate.sleep()
