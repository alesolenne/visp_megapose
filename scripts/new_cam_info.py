#!/usr/bin/env python3
# license removed for brevity
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo

def talker():
    pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30) # 10hz
    msg = CameraInfo()
    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "camera"
    msg.height = 720
    msg.width = 1280
    msg.distortion_model = "rational_polynomial"
    msg.D = [-2.290143, 0.155983, -0.012132, 0.001048, 1.655226, -2.183502, -0.164187, 1.900890]
    msg.K = [616.96516,   0.     , 628.09156,
             0.     , 616.40858, 352.8349 ,
             0.     ,   0.     ,   1.     ]
    msg.R = [1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.]
    msg.P = [580.65076,   0.     , 629.65363,   0.     ,
              0.     , 595.84741, 343.43344,   0.     ,
              0.     ,   0.     ,   1.     ,   0.     ]
    msg.binning_x = 0
    msg.binning_y = 0
    msg.roi.x_offset = 0
    msg.roi.y_offset = 0
    msg.roi.height = 0
    msg.roi.width = 0
    msg.roi.do_rectify = False

    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass