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
  bool reset_bb;

  // Variables
  
  bool initialized;
  bool init_request_done;
  bool got_image;
  double confidence;
  bool object_request;
  unsigned width, height;

  vpImage<vpRGBa> overlay_img;
  vpImage<vpRGBa> vpI;               // Image used for debug display
  vpCameraParameters vpcam_info;
  optional<vpRect> detection;

  sensor_msgs::CameraInfo roscam_info;
  sensor_msgs::Image::ConstPtr rosI; // Image received from ROS

  geometry_msgs::Transform transform;

  json info;

  // Functions

  void waitForImage();
  void frameCallback(const sensor_msgs::Image::ConstPtr &image);
  void broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id, const string &camera_tf);
  
  void init_service_response_callback(const visp_megapose::Init::Response& future);
  void track_service_response_callback(const visp_megapose::Track::Response& future);

  optional<vpRect> detectObjectForInitMegaposeClick(bool &reset_bb, const string &object_name);

public:
MegaPoseClient(ros::NodeHandle *nh) 
{ 
 //Parameters:

 //     image_topic(string) : Name of the image topic
 //     camera_tf(string) : Name of the camera frame
 //     object_name(string) : Name of the object model
 //     reset_bb(bool) : Whether to reset the bounding box saved

  this->nh_ = *nh;
  user = "vispci";
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose";

  initialized = false;
  init_request_done = true;
  bool object_request = false;
  got_image = false;
  reinitThreshold = 0.1;     // Reinit threshold for init and track service

  ros::param::get("image_topic", image_topic);
  ros::param::get("camera_tf", camera_tf);
  ros::param::get("reset_bb",  reset_bb);

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

void MegaPoseClient::broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id, const string &camera_tf)
{
  static tf2_ros::TransformBroadcaster br;  
  static geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = camera_tf;
  transformStamped.child_frame_id = child_frame_id;
  transformStamped.transform = transform;
  br.sendTransform(transformStamped);
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

       ofstream bb_file(megapose_directory + "/output/bb/" + object_name + "_bb.json", ios::out);
       json bb_out;

       int point1 [1][2] = {a, b};
       int point2 [1][2] = {c, d};

       bb_out["object_name"] = object_name;
       bb_out["point1"] = point1[0];
       bb_out["point2"] = point2[0];

       bb_file << bb_out.dump(4);
       bb_file.close();

       vpRect bb(topLeft, bottomRight);
       return bb;

     } else {
        vpDisplay::display(vpI);
        vpDisplay::displayText(vpI, textPosition, "Click when the object is visible and static to start reinitializing megapose.", vpColor::red);
        vpDisplay::flush(vpI);
        return nullopt;
     }
  } else {
     ifstream bb_file(megapose_directory + "/output/bb/" + object_name + "_bb.json", ios::in);
     json bb_in;
     bb_file >> bb_in;

     topLeft= vpImagePoint(bb_in["point1"][0], bb_in["point1"][1]);
     bottomRight=vpImagePoint(bb_in["point2"][0], bb_in["point2"][0]);
     vpRect bb(topLeft, bottomRight);
     bb_file.close();
     return bb;
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
  transform = future.pose;
  confidence = future.confidence;

  if (confidence < reinitThreshold) {
      initialized = false;
      init_request_done = true;
      reset_bb = true;
      
      ROS_INFO("Tracking lost, reinitializing...");
  } else {
              ofstream output_file(megapose_directory + "/output/pose/" + object_name + "_pose.json", ios::out);
              json outJson;

              double translation [1][3] = {transform.translation.x, transform.translation.y, transform.translation.z};
              double rotation [1][4] = {transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w};

              outJson["object_name"] = object_name;
              outJson["position"] = translation[0];
              outJson["rotation"] = rotation[0];

              output_file << outJson.dump(4);
              output_file.close();
              ROS_INFO("Pose saved on %s.json file!", object_name.c_str());
              ROS_INFO("Object tracked successfully!");
              cout<<"Position: "<<transform.translation.x<<" "<<transform.translation.y<<" "<<transform.translation.z<<endl;
              cout<<"Rotation: "<<transform.rotation.x<<" "<<transform.rotation.y<<" "<<transform.rotation.z<<" "<<transform.rotation.w<<endl;
              cout<<"Confidence: "<<confidence<<endl;
              initialized = false;
              init_request_done = true;
              vpDisplay::flush(vpI);
              ROS_INFO("Wanna other objects? yes: y, no: n");
              char c;
              cin>>c;
              if(c == 'n'){ros::shutdown();}
              ros::shutdown();
            }
}

void MegaPoseClient::spin()
{
  // Get parameters
  ROS_INFO("Subscribing to image topic: %s", image_topic.c_str());
  ROS_INFO("Object name: %s", object_name.c_str());

  vector<string> labels = {object_name};

  vpDisplayX *d = NULL;
  d = new vpDisplayX();
  vpDisplay::setTitle(vpI, "MegaPoseClient display");

  ros::Subscriber sub = nh_.subscribe(image_topic, 1000, &MegaPoseClient::frameCallback, this);
  waitForImage();


  ros::spinOnce();
  d->init(vpI); // also init display

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
      if (!object_request)
      {
        ROS_INFO("Which object?");
        cin >> object_name;
        object_request = true;
      }
      

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
           init_service_response_callback(init_pose.response);
           } else {
                   ROS_WARN("Init server down, exiting...");
                   ros::shutdown();
           }
           }

    } 
    else if (initialized) {

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