#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

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
#include <sensor_msgs/Image.h>

// ROS visp_megapose Server includes
#include <visp_megapose/Init.h>
#include <visp_megapose/Track.h>
#include <visp_megapose/ObjectName.h>
#include <visp_megapose/PoseResult.h>

#include <deque>

using namespace std;
using namespace nlohmann;

class MegaPoseClient 
{
  private:
  ros::NodeHandle nh_;
  
  string user;
  string megapose_directory;

  // ROS parameters
  string image_topic;
  string camera_tf;
  string object_name;
  double reinitThreshold;

  // Variables
  
  bool initialized;
  bool init_request_done;
  bool got_image;
  bool got_name;
  double confidence;
  unsigned width, height;
  int n_object;
  int object_found;

  vpImage<vpRGBa> vpI;               // Image used for debug display
  optional<vpRect> detection;

  geometry_msgs::Transform transform;

  sensor_msgs::CameraInfo roscam_info;
  sensor_msgs::Image::ConstPtr rosI; // Image received from ROS

  json info;

  // Functions

  void waitForImage();
  void frameCallback(const sensor_msgs::Image::ConstPtr &image);

  void init_service_response_callback(const visp_megapose::Init::Response& future);
  void track_service_response_callback(const visp_megapose::Track::Response& future);

  optional<vpRect> detectObjectForInitMegaposeClick(const string &object_name);

  void waitForName();
  void frameObject(const visp_megapose::ObjectName &command);

public:
MegaPoseClient(ros::NodeHandle *nh) 
{ 
 //Parameters:

 //     image_topic(string) : Name of the image topic
 //     camera_tf(string) : Name of the camera frame
 //     object_name(string) : Name of the object model

  this->nh_ = *nh;
  user = "vispci";
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose";

  got_name = false;
  initialized = false;
  init_request_done = true;
  got_image = false;
  reinitThreshold = 0.9;     // Reinit threshold for init and track service
  object_found = 0;

  ros::param::get("image_topic", image_topic);
  ros::param::get("camera_tf", camera_tf);

  // Load camera parameters from file
  ifstream camera_file(megapose_directory + "/params/camera.json", ifstream::in);
  json info;
  camera_file >> info;
  roscam_info.K = {info["K"][0][0], 0, info["K"][0][2], 0, info["K"][1][1], info["K"][1][2], 0, 0, info["K"][2][2]};
  camera_file.close();

};

~MegaPoseClient() = default;

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

void MegaPoseClient::frameCallback(const sensor_msgs::Image::ConstPtr &image)
{
  rosI = image;
  vpI = visp_bridge::toVispImageRGBa(*image);
  width = image->width;
  height = image->height;
  got_image = true;
}

optional<vpRect> MegaPoseClient::detectObjectForInitMegaposeClick(const string &object_name)
{ 
  vpImagePoint topLeft, bottomRight;

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

    vpRect bb(topLeft, bottomRight);
    return bb;

  } else {
    vpDisplay::display(vpI);
    vpDisplay::displayText(vpI, textPosition, "Click when the object is visible and static to start reinitializing megapose.", vpColor::red);
    vpDisplay::flush(vpI);
    return nullopt;
  }

}

void MegaPoseClient::init_service_response_callback(const visp_megapose::Init::Response& future)
{
  transform = future.pose;
  confidence = future.confidence;
  ROS_INFO("Bounding box generated, checking the confidence");


  if (confidence < reinitThreshold) {
      ROS_INFO("Initial pose not reliable, reinitializing...");
  }
  else {
      initialized = true;
      init_request_done = false;
      ROS_INFO("Initialized successfully!");
  }
}

void MegaPoseClient::track_service_response_callback(const visp_megapose::Track::Response& future)
{
  static ros::Publisher pub_pose_ = nh_.advertise<visp_megapose::PoseResult>("PoseResult", 1, true);

  transform = future.pose;
  confidence = future.confidence;
  initialized = false;
  init_request_done = true;

  if (confidence < reinitThreshold) {


      ROS_INFO("Tracking lost, reinitializing...");

  } else {        
              
              visp_megapose::PoseResult res;
              res.pose = transform;
              res.skip = true;
              pub_pose_.publish(res);
              object_found = object_found + 1;
              ROS_INFO("Object %s found! Pose: [%f, %f, %f, %f, %f, %f, %f] ", object_name.c_str(), transform.translation.x, transform.translation.y, transform.translation.z, transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
              ROS_INFO("Pose confidence: %f: ", confidence);

              got_name = false;

         }

}

void MegaPoseClient::frameObject(const visp_megapose::ObjectName &command)
{
  got_name = true;
  object_name = command.obj_name;
  n_object = command.number;
}

void MegaPoseClient::waitForName()
{
  ros::Rate loop_rate(10);
  while (ros::ok()) {
    if (got_name) {
      return;
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
}

void MegaPoseClient::spin()
{
  // Get parameters
  ROS_INFO("Subscribing to image topic: %s", image_topic.c_str());
  ROS_INFO("Camera info loaded from camera.json file");

  ros::Subscriber sub = nh_.subscribe(image_topic, 1000, &MegaPoseClient::frameCallback, this);
  ros::Subscriber sub2 = nh_.subscribe("ObjectList", 1000, &MegaPoseClient::frameObject, this);

  waitForImage();


  vpDisplayX *d = NULL;
  d = new vpDisplayX();

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

  while (ros::ok()) {
    vpDisplay::display(vpI);
    ros::spinOnce();
    optional<vpRect> detection = nullopt;

    if (!initialized) {
  
       waitForName();

       detection = detectObjectForInitMegaposeClick(object_name);
       
      if (detection && init_request_done) {

        visp_megapose::Init init_pose;
        init_pose.request.object_name = object_name;
        init_pose.request.topleft_i = detection->getTopLeft().get_i();
        init_pose.request.topleft_j = detection->getTopLeft().get_j();
        init_pose.request.bottomright_i = detection->getBottomRight().get_i();
        init_pose.request.bottomright_j = detection->getBottomRight().get_j();
        init_pose.request.image = *rosI;

        if (init_pose_client.call(init_pose)) {
           init_service_response_callback(init_pose.response);
           } else {
                   ROS_WARN("Init server down, exiting...");
                   ros::shutdown();
           }
           }
    } else if (initialized) {

        visp_megapose::Track track_pose;
        track_pose.request.object_name = object_name;
        track_pose.request.init_pose = transform;
        track_pose.request.image = *rosI;

        if (track_pose_client.call(track_pose)) {
            track_service_response_callback(track_pose.response);
            } else {
                    ROS_WARN("Tracking server down, exiting...");
                    ros::shutdown();
            }

    }

  vpDisplay::flush(vpI);
  
  if (object_found == n_object)
  {
    ROS_INFO("All object in the list found!");
    break;
  }
  
  }
  
  delete d;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv,"megapose_client");
  ros::NodeHandle nh;
  MegaPoseClient nc = MegaPoseClient(&nh);
  nc.spin();
  return 0;
}