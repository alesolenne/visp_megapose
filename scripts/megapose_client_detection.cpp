#include <filesystem>
#include <fstream>
#include <iostream>

// visp includes
#include <visp3/core/vpTime.h>
#include <visp3/gui/vpDisplayX.h>

// OpenCV/visp bridge includes
#include <visp_bridge/3dpose.h>
#include <visp_bridge/camera.h>
#include <visp_bridge/image.h>

// ROS includes
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

// ROS message filter includes
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

// ROS visp_megapose Server includes
#include <visp_megapose/Init.h>
#include <visp_megapose/Track.h>

using namespace std;

bool fileExists(const string &path)
{
  filesystem::path p{path};
  return filesystem::exists(p) && filesystem::is_regular_file(p);
}

class MegaPoseClient 
{
  private:
  ros::NodeHandle nh_;
  
  // ROS parameters
  string image_topic;
  string camera_info_topic;
  string camera_tf;
  string object_name;
  double reinitThreshold;
  bool reset_bb;
  string label;

  string object_name_1;
  string object_name_2;
  string object_name_3;
  string object_name_4;
  string object_name_5;
  string object_name_6;
  string object_name_7;
  string object_name_8;


  int n;

  int i;
  int j;

  void frameCallback(const sensor_msgs::Image::ConstPtr &image,
                     const sensor_msgs::CameraInfo::ConstPtr &cam_info);
  bool got_image;
  vpCameraParameters vpcam_info;
  sensor_msgs::CameraInfo::ConstPtr roscam_info;
  unsigned width, height;

  vpImage<vpRGBa> vpI;               // Image used for debug display
  optional<vpRect> detection;

  sensor_msgs::Image::ConstPtr rosI; // Image received from ROS

  double confidence;

  void waitForImage();
  bool initialized;
  bool init_request_done;
  bool show_bb;

  void broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id,
                          const string &camera_tf);
  geometry_msgs::Transform transform;

  vpColor interpolate(const vpColor &low, const vpColor &high, const float f);
  void displayScore(float);
  optional<vpRect> detectObjectForInitMegaposeClick(bool &reset_bb, const string &object_name);

public:
MegaPoseClient(ros::NodeHandle *nh) 
{ 
 //Parameters:

 //     image_topic(string) : Name of the image topic
 //     camera_info_topic(string) : Name of the camera info topic
 //     camera_tf(string) : Name of the camera frame
 //     object_name(string) : Name of the object model
 //     reinitThreshold(int) : Reinit threshold for init and track service 
 //     reset_bb(bool) : Whether to reset the bounding box saved

  this->nh_ = *nh;
  initialized = false;
  init_request_done = true;
  got_image = false;
  i = 0;
  j = 1;
  
  ros::param::get("image_topic", image_topic);
  ros::param::get("camera_info_topic", camera_info_topic);
  ros::param::get("camera_tf", camera_tf);
  ros::param::get("reinitThreshold",  reinitThreshold);
  ros::param::get("reset_bb",  reset_bb);
  ros::param::get("object_name_1", object_name_1);
  ros::param::get("object_name_2", object_name_2);
  ros::param::get("object_name_3", object_name_3);
  ros::param::get("object_name_4", object_name_4);
  ros::param::get("object_name_5", object_name_5);
  ros::param::get("object_name_6", object_name_6);
  ros::param::get("object_name_7", object_name_7);
  ros::param::get("object_name_8", object_name_8);
  ros::param::get("n_object", n);

};

~MegaPoseClient()
{
  ROS_INFO("Shutting down MegaPoseClient");
  ros::shutdown();
}

void spin();
};


void MegaPoseClient::waitForImage()
{
  ros::Rate loop_rate(10);
  ROS_INFO("Waiting for a rectified image...");
  while (ros::ok()) {
    if (got_image) {
      ROS_INFO("Got image!");
      return;
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
}
void MegaPoseClient::frameCallback(const sensor_msgs::Image::ConstPtr &image,
                                   const sensor_msgs::CameraInfo::ConstPtr &cam_info)
{
  rosI = image;
  roscam_info = cam_info;
  vpI = visp_bridge::toVispImageRGBa(*image);
  vpcam_info = visp_bridge::toVispCameraParameters(*cam_info);
  width = image->width;
  height = image->height;
  got_image = true;
}
void MegaPoseClient::broadcastTransform(const geometry_msgs::Transform &transform,
                                        const string &child_frame_id, const string &camera_tf)
{
  static tf2_ros::TransformBroadcaster br;  
  static geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = camera_tf;
  transformStamped.child_frame_id = child_frame_id;
  transformStamped.transform = transform;
  br.sendTransform(transformStamped);
}
void MegaPoseClient::spin()
{
  // Get parameters
  ROS_INFO("Subscribing to image topic: %s", image_topic.c_str());
  ROS_INFO("Subscribing to camera info topic: %s", camera_info_topic.c_str());
  string object_list[]= {object_name_1, object_name_2, object_name_3, object_name_4, object_name_5, object_name_6, object_name_7, object_name_8};

  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh_, image_topic, 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh_, camera_info_topic, 1);
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_sub, info_sub, 10);
  sync.registerCallback(boost::bind(&MegaPoseClient::frameCallback, this, _1, _2));
  waitForImage();

  vpDisplayX *d = NULL;
  d = new vpDisplayX();
  ros::spinOnce();
  d->init(vpI); // also init display
  vpDisplay::setTitle(vpI, "MegaPoseClient display");

  ros::ServiceClient init_pose_client = nh_.serviceClient<visp_megapose::Init>("init_pose");
  ros::ServiceClient track_pose_client = nh_.serviceClient<visp_megapose::Track>("track_pose");

  while (!init_pose_client.waitForExistence(ros::Duration(10)) && !track_pose_client.waitForExistence(ros::Duration(10))) {
    if (!ros::ok()) {
      ROS_ERROR("Interrupted while waiting for the service. Exiting.");
      return;
    }
    ROS_INFO("Service not available, waiting again...");
  }

  ros::Rate loop_rate(10);
  bool flag_track = false; //flag for tracking
  bool show_bb = false; //show initial bb

  while (ros::ok()) {
    vpDisplay::display(vpI);
    ros::spinOnce();
    optional<vpRect> detection = nullopt;

    if (i > (n-1))     // suppongo di avere n oggetti
        break;    // esce se tutti trovati
    
    object_name = object_list[i];
    label = to_string(i);

    if (!initialized) {

       detection = detectObjectForInitMegaposeClick(reset_bb, object_name);

      if (detection && init_request_done) {
        visp_megapose::Init init_pose;
        init_pose.request.object_name = object_name;
        init_pose.request.topleft_i = detection->getTopLeft().get_i();
        init_pose.request.topleft_j = detection->getTopLeft().get_j();
        init_pose.request.bottomright_i = detection->getBottomRight().get_i();
        init_pose.request.bottomright_j = detection->getBottomRight().get_j();

        init_pose.request.image = *rosI;


        if (init_pose_client.call(init_pose)) {
            transform = init_pose.response.pose;
            confidence = init_pose.response.confidence;
            ROS_INFO("Bounding box generated, checking the confidence");

            if (confidence < reinitThreshold) {
                ROS_INFO("Initial pose not reliable, reinitializing...");
                reset_bb = true;
            }
            else {
                initialized = true;
                init_request_done = false;
                flag_track = false;
                ROS_INFO("Initialized successfully!");
            }
        } else {
            ROS_WARN("Init server down, start server node and reinitialize...");
            init_pose_client.waitForExistence();

          }
      }
    } else if (initialized) {
        visp_megapose::Track track_pose;

        track_pose.request.object_name = object_name;
        track_pose.request.init_pose = transform;
        track_pose.request.image = *rosI;
        if (track_pose_client.call(track_pose)) {
            transform = track_pose.response.pose;
            confidence = track_pose.response.confidence;

            if (confidence < reinitThreshold) {
                initialized = false;
                init_request_done = true;
                reset_bb = true;
                flag_track = false;
                ROS_INFO("Tracking lost, reinitializing...");
            } else { if(!flag_track){
                        ROS_INFO("Object tracked successfully!");
                        flag_track = true; //print only once
                      }
                      j++;
                      if (j == 100) {            //dopo aver trakkato 100 volte passa oggetto successivo
                            ofstream fd;
                            fd.open("/home/nuvo/catkin_ws/src/visp_megapose/output/box" + label + "_pose.txt", ios::out);   
                            fd <<setprecision(10)<<transform.translation.x<<endl<<transform.translation.y<<endl<<transform.translation.z<<endl /* scrittura dati traslazione*/
                            <<transform.rotation.x<<endl<<transform.rotation.y<<endl<<transform.rotation.z<<endl<<transform.rotation.w; /* scrittura dati rotazione*/
                            fd.close();  
                            ROS_INFO("Pose saved on box%s_pose.txt file!", label.c_str());
                            }
                      string keyboardEvent3;
                      const bool keyPressed3 = vpDisplay::getKeyboardEvent(vpI, keyboardEvent3, false);
                      if (keyPressed3) {
                          if (keyboardEvent3 == "c") {
                            initialized = false;
                            init_request_done = true;
                            i++;                   // passo all'oggetto successivo
                            j = 1;                 // reinizializzo contatore
                          }
                          if (keyboardEvent3 == "b") {
                             show_bb=true;
                          }
                          if (keyboardEvent3 == "t") {
                                      initialized = false;
                                      init_request_done = true;
                                      reset_bb=true;
                                      ROS_INFO("Reinitialize...");
                                  }

                          if (keyboardEvent3 == "q") {
                                      break;
                                  }
                  
                      }
                            if (show_bb) {
                                  vpImagePoint topLeft(detection->getTopLeft().get_i(), detection->getTopLeft().get_j());
                                  vpImagePoint bottomRight(detection->getBottomRight().get_i(),detection->getBottomRight().get_j());
                                  vpDisplay::displayRectangle(vpI,topLeft, bottomRight, vpColor::red); 
                                  if (keyboardEvent3 == "n") {
                                      show_bb=!show_bb;
          }
      }
            }
        } else {
            ROS_WARN("Tracking server down, start server node and reinitialize...");
            vpDisplay::flush(vpI);
            track_pose_client.waitForExistence();
            initialized = false;
            init_request_done = true;
            reset_bb = true;
        }
      
      vpDisplay::displayText(vpI, 30, 20, "t: Reinitialize", vpColor::red);
      vpDisplay::displayText(vpI, 40, 20, "b: Display Bounding box / n: Clear Bounding box ", vpColor::red);


      static vpHomogeneousMatrix M;
      M = visp_bridge::toVispHomogeneousMatrix(transform);
      vpDisplay::displayFrame(vpI, M, vpcam_info, 0.05, vpColor::none, 3);
      displayScore(confidence);
      // string name = "object"+label;
      // broadcastTransform(transform, name, camera_tf);
    }

  vpDisplay::displayText(vpI, 20, 20, "q: Quit", vpColor::red);
  vpDisplay::flush(vpI);
  string keyboardEvent2;
  const bool keyPressed2 = vpDisplay::getKeyboardEvent(vpI, keyboardEvent2, false);
      if (keyPressed2) {
          if (keyboardEvent2 == "q") {
              break;
              }
      }
  loop_rate.sleep();

  }
  delete d;
}

void MegaPoseClient::displayScore(float confidence)
{
  const unsigned top = static_cast<unsigned>(vpI.getHeight() * 0.85f);
  const unsigned height = static_cast<unsigned>(vpI.getHeight() * 0.1f);
  const unsigned left = static_cast<unsigned>(vpI.getWidth() * 0.05f);
  const unsigned width = static_cast<unsigned>(vpI.getWidth() * 0.5f);
  vpRect full(left, top, width, height);
  vpRect scoreRect(left, top, width * confidence, height);
  const vpColor low = vpColor::red;
  const vpColor high = vpColor::green;
  const vpColor c = interpolate(low, high, confidence);

  vpDisplay::displayRectangle(vpI, full, c, false, 5);
  vpDisplay::displayRectangle(vpI, scoreRect, c, true, 1);
}
vpColor MegaPoseClient::interpolate(const vpColor &low, const vpColor &high, const float f)
{
  const float r = ((float)high.R - (float)low.R) * f;
  const float g = ((float)high.G - (float)low.G) * f;
  const float b = ((float)high.B - (float)low.B) * f;
  return vpColor((unsigned char)r, (unsigned char)g, (unsigned char)b);
}

optional<vpRect> MegaPoseClient::detectObjectForInitMegaposeClick(bool &reset_bb, const string &object_name)
{ 
  vpImagePoint topLeft, bottomRight;

  if (reset_bb){
    const bool startLabelling = vpDisplay::getClick(vpI, false);

    const vpImagePoint textPosition(10.0, 20.0);

     if (startLabelling) {
       vpDisplay::displayText(vpI, textPosition, "Click the upper left corner of the bounding box", vpColor::red);
       vpDisplay::flush(vpI);
       vpDisplay::getClick(vpI, topLeft, true);
       vpDisplay::display(vpI);
       vpDisplay::displayCross(vpI, topLeft, 5, vpColor::red, 2);
       vpDisplay::displayText(vpI, textPosition, "Click the bottom right corner of the bounding box", vpColor::red);
       vpDisplay::flush(vpI);
       vpDisplay::getClick(vpI, bottomRight, true);
       int a = topLeft.get_i();
       int b = topLeft.get_j();
       int c = bottomRight.get_i();
       int d = bottomRight.get_j();

       ofstream fd;
       fd.open("/home/nuvo/catkin_ws/src/visp_megapose/output/bb/multi/box" + label + "_bb.txt", ios::out);   
       fd <<a<<endl<<b<<endl<<c<<endl<<d; /* scrittura dati */
       fd.close();

       vpRect bb(topLeft, bottomRight);
       return bb;

     } else {
        vpDisplay::display(vpI);
        vpDisplay::displayText(vpI, textPosition, "Click when the object is visible and static to start reinitializing megapose.", vpColor::red);
        vpDisplay::flush(vpI);
        return nullopt;
     }
  } else {
     int bb_r[4];
     ifstream fd;
     fd.open("/home/nuvo/catkin_ws/src/visp_megapose/output/bb/multi/box" + label + "_bb.txt", ios::in);
     for(int i=0;i<4;i++) {
        fd >> bb_r[i];
        }
     fd.close();
     topLeft= vpImagePoint(bb_r[0], bb_r[1]);
     bottomRight=vpImagePoint(bb_r[2], bb_r[3]);
     vpRect bb(topLeft, bottomRight);
     return bb;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv,"megapose_client");
  ros::NodeHandle nh;
  MegaPoseClient nc = MegaPoseClient(&nh);
  nc.spin();
  return 0;
}
