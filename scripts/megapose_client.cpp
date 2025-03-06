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
#include <visp_megapose/Render.h>

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

  vpImage<vpRGBa> overlay_img;

  void frameCallback(const sensor_msgs::Image::ConstPtr &image);
  bool got_image;
  vpCameraParameters vpcam_info;
  sensor_msgs::CameraInfo roscam_info;
  unsigned width, height;

  vpImage<vpRGBa> vpI;               // Image used for debug display
  optional<vpRect> detection;

  sensor_msgs::Image::ConstPtr rosI; // Image received from ROS

  double confidence;

  void waitForImage();
  bool initialized;
  bool init_request_done;
  bool show_bb;
  bool flag_track;
  bool flag_render;
  json info;

  void broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id,
                          const string &camera_tf);
  geometry_msgs::Transform transform;

  vpColor interpolate(const vpColor &low, const vpColor &high, const float f);
  void displayScore(float);
  optional<vpRect> detectObjectForInitMegaposeClick(bool &reset_bb, const string &object_name);

  void init_service_response_callback(const visp_megapose::Init::Response& future);
  void track_service_response_callback(const visp_megapose::Track::Response& future);
  void render_service_response_callback(const visp_megapose::Render::Response& future);

  bool overlayModel;
  void overlayRender(const vpImage<vpRGBa> &overlay);
  
  void displayEvent(const optional<vpRect> &detection);

public:
MegaPoseClient(ros::NodeHandle *nh) 
{ 
 //Parameters:

 //     image_topic(string) : Name of the image topic
 //     camera_tf(string) : Name of the camera frame
 //     object_name(string) : Name of the object model
 //     reinitThreshold(int) : Reinit threshold for init and track service 
 //     reset_bb(bool) : Whether to reset the bounding box saved

  this->nh_ = *nh;
  user = "vispci";
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose";

  initialized = false;
  init_request_done = true;
  got_image = false;
  overlayModel = false;


  ros::param::get("image_topic", image_topic);
  ros::param::get("camera_tf", camera_tf);
  ros::param::get("object_name", object_name);
  ros::param::get("reinitThreshold",  reinitThreshold);
  ros::param::get("reset_bb",  reset_bb);

  std::ifstream camera_file(megapose_directory + "/params/camera.json", std::ifstream::binary);
  json info;
  camera_file >> info;
  roscam_info.K = {info["K"][0][0], 0, info["K"][0][2], 0, info["K"][1][1], info["K"][1][2], 0, 0, info["K"][2][2]};

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
       fd.open(megapose_directory + "/output/bb/" + object_name + "_bb.txt", ios::out);   
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
     fd.open(megapose_directory + "/output/bb/" + object_name + "_bb.txt", ios::in);
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

void MegaPoseClient::init_service_response_callback(const visp_megapose::Init::Response& future)
{
  transform = future.pose;
  confidence = future.confidence;
  ROS_INFO("Bounding box generated, checking the confidence");

  if (confidence < reinitThreshold) {
      ROS_INFO("Initial pose not reliable, reinitializing...");
      reset_bb = true;
  }
  else {
      initialized = true;
      init_request_done = false;
      flag_track = false;
      flag_render = false;
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
      flag_track = false;
      flag_render = false;
      ROS_INFO("Tracking lost, reinitializing...");
  } else { if(!flag_track){
              ROS_INFO("Object tracked successfully!");
              flag_track = true; //print only once
            }
  }
}

void MegaPoseClient::render_service_response_callback(const visp_megapose::Render::Response& future)
{
  overlay_img = visp_bridge::toVispImageRGBa(future.image);
  if (overlay_img.getSize() > 0){
    overlayRender(overlay_img);
  if(!flag_render){
      ROS_INFO("Model rendered successfully!");
      flag_render = true; //print only once
    }
  }
}

void MegaPoseClient::overlayRender(const vpImage<vpRGBa> &overlay)
{
  vpRGBa black(0, 0, 0);
  for (unsigned int i = 0; i < height; ++i)
  {
    for (unsigned int j = 0; j < width; ++j)
    {
      if (const_cast<vpRGBa&>(overlay[i][j]) != black)
      {
        vpI[i][j] = overlay[i][j];
      }
    }
  }
}

void MegaPoseClient::displayEvent(const optional<vpRect> &detection) 
{
  vpDisplay::displayText(vpI, 20, 20, "q: Quit", vpColor::red);
  vpDisplay::displayText(vpI, 30, 20, "t: Reinitialize", vpColor::red);
  vpDisplay::displayText(vpI, 40, 20, "b: Display Bounding box / n: Clear Bounding box ", vpColor::red);
  vpDisplay::displayText(vpI, 50, 20, "s: Save current pose", vpColor::red);
  vpDisplay::displayText(vpI, 60, 20, "r: Render model", vpColor::red);

  string keyboardEvent;
  const bool keyPressed = vpDisplay::getKeyboardEvent(vpI, keyboardEvent, false);

  if (keyPressed) {

    if (keyboardEvent == "q") {
        ros::shutdown();
    }

    if (keyboardEvent == "t") {
        initialized = false;
        init_request_done = true;
        reset_bb = true;
        ROS_INFO("Reinitialize...");
    }

    if (keyboardEvent == "b") {
        show_bb = true;
    }

    if (keyboardEvent == "s") {
        ofstream fd;
        fd.open(megapose_directory + "/output/pose/" + object_name + "_pose.txt", ios::out);   
        fd <<setprecision(10)<<transform.translation.x<<endl<<transform.translation.y<<endl<<transform.translation.z<<endl /* scrittura dati traslazione*/
            <<transform.rotation.x<<endl<<transform.rotation.y<<endl<<transform.rotation.z<<endl<<transform.rotation.w; /* scrittura dati rotazione*/
        fd.close();  
        ROS_INFO("Pose saved on %s.txt file!", object_name.c_str());
    }

    if (keyboardEvent == "r") {
        overlayModel = !overlayModel;
        flag_render = false;
    }

    }

  if (show_bb) {
    vpImagePoint topLeft(detection->getTopLeft().get_i(), detection->getTopLeft().get_j());
    cout << topLeft.get_i() << " " << topLeft.get_j() << endl;
    vpImagePoint bottomRight(detection->getBottomRight().get_i(),detection->getBottomRight().get_j());
    cout << bottomRight.get_i() << " " << bottomRight.get_j() << endl;
    vpDisplay::displayRectangle(vpI, topLeft, bottomRight, vpColor::red); 
    
    if (keyboardEvent == "n") {
        show_bb=!show_bb;
    }

  }

  static vpHomogeneousMatrix M;
  M = visp_bridge::toVispHomogeneousMatrix(transform);
  vpcam_info = visp_bridge::toVispCameraParameters(roscam_info);
  vpDisplay::displayFrame(vpI, M, vpcam_info, 0.05, vpColor::none, 3);
  displayScore(confidence);
  broadcastTransform(transform, object_name, camera_tf);

}

void MegaPoseClient::spin()
{
  // Get parameters
  ROS_INFO("Subscribing to image topic: %s", image_topic.c_str());
  ROS_INFO("Object name: %s", object_name.c_str());

  vector<string> labels = {object_name};

  ros::Subscriber sub = nh_.subscribe(image_topic, 1000, &MegaPoseClient::frameCallback, this);
  waitForImage();

  vpDisplayX *d = NULL;
  d = new vpDisplayX();
  ros::spinOnce();
  d->init(vpI); // also init display
  vpDisplay::setTitle(vpI, "MegaPoseClient display");

  ros::ServiceClient init_pose_client = nh_.serviceClient<visp_megapose::Init>("init_pose");
  ros::ServiceClient track_pose_client = nh_.serviceClient<visp_megapose::Track>("track_pose");
  ros::ServiceClient render_client = nh_.serviceClient<visp_megapose::Render>("render_object");

  while (!init_pose_client.waitForExistence(ros::Duration(10)) && !track_pose_client.waitForExistence(ros::Duration(10)) && !render_client.waitForExistence(ros::Duration(10))) {
    if (!ros::ok()) {
      ROS_ERROR("Interrupted while waiting for the service. Exiting.");
      return;
    }
    ROS_INFO("Service not available, waiting again...");
  }

  ros::Rate loop_rate(100);
  bool flag_track = false; //flag for tracking
  bool flag_render = false; //flag for tracking
  bool show_bb = false; //show initial bb

  while (ros::ok()) {
    vpDisplay::display(vpI);
    ros::spinOnce();
    optional<vpRect> detection = nullopt;

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
      
      if (overlayModel)
      {
        visp_megapose::Render render_request;
        render_request.request.object_name = object_name;
        render_request.request.pose = transform;
        if (render_client.call(render_request)) 
        {
          render_service_response_callback(render_request.response);
        } 
        else 
        {
          ROS_ERROR("Render server down, exiting...");
          ros::shutdown();
        }
      }

      displayEvent(detection);

    }

  vpDisplay::flush(vpI);

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