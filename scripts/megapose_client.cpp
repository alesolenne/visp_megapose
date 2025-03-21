#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdio.h>

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

// ROS message filter includes
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

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
  string camera_info_topic;
  string camera_tf;
  string object_name;
  string depth_topic;
  bool depth_enable;
  bool reset_bb;
  double reinitThreshold;
  double refilterThreshold;
  int buffer_size;

  // Variables
  bool initialized;
  bool init_request_done;
  bool show_init_bb;
  bool show_track_bb;
  bool flag_track;
  bool flag_render;
  bool overlayModel;
  bool got_image;
  bool got_depth;
  double confidence;
  unsigned width, height, widthD, heightD;

  json info;
  float bb[4];

  vpImage<vpRGBa> overlay_img;
  vpImage<vpRGBa> vpI;        
  vpCameraParameters vpcam_info;
  optional<vpRect> detection;
  deque<double> buffer_x, buffer_y, buffer_z,buffer_qw, buffer_qx, buffer_qy, buffer_qz;

  // ROS variables
  sensor_msgs::CameraInfoConstPtr roscam_info;

  boost::shared_ptr<const sensor_msgs::Image> rosI; 
  boost::shared_ptr<const sensor_msgs::Image> rosD; 

  geometry_msgs::Transform transform; 
  geometry_msgs::Transform filter_transform;

  message_filters::Subscriber<sensor_msgs::Image> image_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub;

  message_filters::Subscriber<sensor_msgs::Image> depth_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;
 
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicyRGB;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image> SyncPolicyRGBD;

  boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGB> > sync_rgb;
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGBD> > sync_rgbd;

  // Functions

  void waitForImage();
  void waitForDepth();
  void frameCallback_rgb(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camera_info);
  void frameCallback_rgbd(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info, const sensor_msgs::ImageConstPtr &depth);
  void broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id, const string &camera_tf);
  void broadcastTransform_filter(const geometry_msgs::Transform &origpose, const string &child_frame_id, const string &camera_tf);
  double calculateMovingAverage(const deque<double>& buffer);
  void displayScore(float);
  void overlayRender(const vpImage<vpRGBa> &overlay);
  void displayEvent(const optional<vpRect> &detection);
  vpColor interpolate(const vpColor &low, const vpColor &high, const float f);
  optional<vpRect> detectObjectForInitMegaposeClick(bool &reset_bb, const string &object_name);

  void init_service_response_callback(const visp_megapose::Init::Response& future);
  void track_service_response_callback(const visp_megapose::Track::Response& future);
  void render_service_response_callback(const visp_megapose::Render::Response& future);

public:
MegaPoseClient(ros::NodeHandle *nh) 
{ 
 //Parameters:

 //     image_topic(string) : Name of the image topic
 //     camera_info_topic(string) : Name of the camera info topic
 //     camera_tf(string) : Name of the camera frame
 //     object_name(string) : Name of the object model
 //     reset_bb(bool) : Whether to reset the bounding box saved
 //     depth_enable(bool) : Whether to use depth image
 //     depth_topic(string) : Name of the depth image topic
 //     reinitThreshold(double) : Reinit threshold for init and track service
 //     refilterThreshold(double) : Filter threshold for filter poses
 //     buffer_size(int) : Buffer size for filter poses

  this->nh_ = *nh;
  char username[32];
  cuserid(username);
  std::string user(username);
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose"; //change this path to your catkin workspace path

  initialized = false;
  init_request_done = true;
  got_image = false;
  overlayModel = false;
  show_init_bb = false;
  show_track_bb = false;

  flag_track = false; //flag for tracking
  flag_render = false; //flag for tracking

  ros::param::get("image_topic", image_topic);
  ros::param::get("camera_info_topic", camera_info_topic);
  ros::param::get("camera_tf", camera_tf);
  ros::param::get("object_name", object_name);
  ros::param::get("reset_bb",  reset_bb);
  ros::param::get("depth_enable", depth_enable);
  ros::param::get("depth_topic", depth_topic);
  ros::param::get("reinitThreshold", reinitThreshold);
  ros::param::get("refilterThreshold", refilterThreshold);
  ros::param::get("buffer_size", buffer_size);

  image_sub.subscribe(*nh, image_topic, 1);
  camera_info_sub.subscribe(*nh, camera_info_topic, 1);

  if (!depth_enable)
  {
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGB> > sync1(new message_filters::Synchronizer<SyncPolicyRGB>(SyncPolicyRGB(1), image_sub, camera_info_sub));
    sync_rgb = sync1;
    sync_rgb->registerCallback(boost::bind(&MegaPoseClient::frameCallback_rgb, this, _1, _2));
  }
  else
  {
    depth_sub.subscribe(*nh, depth_topic, 1);
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGBD> > sync2(new message_filters::Synchronizer<SyncPolicyRGBD>(SyncPolicyRGBD(1), image_sub, camera_info_sub, depth_sub));
    sync_rgbd = sync2;
    sync_rgbd->registerCallback(boost::bind(&MegaPoseClient::frameCallback_rgbd, this, _1, _2, _3));
  }
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
      ROS_INFO("Got RGB image!");
      return;
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
}

void MegaPoseClient::waitForDepth()
 {
   ros::Rate loop_rate(10);
   ROS_INFO("Waiting for a rectified depth...");
   while (ros::ok())
   {
     if (got_depth)
     {
       ROS_INFO("Got Depth image!");
       return;
     }
     ros::spinOnce();
     loop_rate.sleep();
   }
 }

void MegaPoseClient::frameCallback_rgb(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info)
{
rosI = image;
roscam_info = cam_info;
width = image->width;
height = image->height;

vpI = visp_bridge::toVispImageRGBa(*image);
vpcam_info = visp_bridge::toVispCameraParameters(*cam_info);

got_image = true;
}

void MegaPoseClient::frameCallback_rgbd(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info, const sensor_msgs::ImageConstPtr &depth)
{
  rosI = image;
  rosD = depth;
  roscam_info = cam_info;
  width = image->width;
  height = image->height;
  widthD = depth->width;
  heightD = depth->height;

  vpI = visp_bridge::toVispImageRGBa(*image);
  vpcam_info = visp_bridge::toVispCameraParameters(*cam_info);

  got_image = true;
  got_depth = true;
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

void MegaPoseClient::broadcastTransform_filter(const geometry_msgs::Transform &origpose, const string &child_frame_id, const string &camera_tf)
{
  if(confidence > refilterThreshold)
  {
    if (boost::numeric_cast<int>(buffer_x.size()) >= buffer_size)
    {
      buffer_x.pop_front();
      buffer_y.pop_front();
      buffer_z.pop_front();
      buffer_qw.pop_front();
      buffer_qx.pop_front();
      buffer_qy.pop_front();
      buffer_qz.pop_front();
    }
    buffer_x.push_back(origpose.translation.x);
    buffer_y.push_back(origpose.translation.y);
    buffer_z.push_back(origpose.translation.z);
    buffer_qw.push_back(origpose.rotation.w);
    buffer_qx.push_back(origpose.rotation.x);
    buffer_qy.push_back(origpose.rotation.y);
    buffer_qz.push_back(origpose.rotation.z);
    
    filter_transform.translation.x = calculateMovingAverage(buffer_x);
    filter_transform.translation.y = calculateMovingAverage(buffer_y);
    filter_transform.translation.z = calculateMovingAverage(buffer_z);
    filter_transform.rotation.w = calculateMovingAverage(buffer_qw);
    filter_transform.rotation.x = calculateMovingAverage(buffer_qx);
    filter_transform.rotation.y = calculateMovingAverage(buffer_qy);
    filter_transform.rotation.z = calculateMovingAverage(buffer_qz);

    static tf2_ros::TransformBroadcaster br2;  
    static geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = camera_tf;
    transformStamped.child_frame_id = child_frame_id + "_filtered";
    transformStamped.transform = filter_transform;
    br2.sendTransform(transformStamped);
  }
}

double MegaPoseClient::calculateMovingAverage(const deque<double>& buffer)
{
  if (buffer.size() < 1) return 0.0;  // Avoid division by zero
  return accumulate(buffer.begin(), buffer.end(), 0.0) / buffer.size();
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
  vpDisplay::displayText(vpI, 40, 20, "b: Enable/Disable Current BBox / n: Enable/Disable Initial BBox ", vpColor::red);
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

    if (keyboardEvent == "s") {
        
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
    }

    if (keyboardEvent == "r") {
        overlayModel = !overlayModel;
        flag_render = false;
    }

  }

  if (show_init_bb) {

    vpImagePoint init_topLeft(detection->getTopLeft().get_i(), detection->getTopLeft().get_j());
    vpImagePoint init_bottomRight(detection->getBottomRight().get_i(),detection->getBottomRight().get_j());
    vpDisplay::displayRectangle(vpI, init_topLeft, init_bottomRight, vpColor::blue, false, 2);
    
    if (keyboardEvent == "n") {
        show_init_bb=!show_init_bb;
    }
  }

  else{

    if (keyboardEvent == "n") {
        show_init_bb=!show_init_bb;
    }

  }

  if (show_track_bb) {

    vpImagePoint topLeft(bb[1], bb[0]);
    vpImagePoint bottomRight(bb[3], bb[2]);
    vpDisplay::displayRectangle(vpI, topLeft, bottomRight, vpColor::red, false, 2);
    
    if (keyboardEvent == "b") {
        show_track_bb=!show_track_bb;
    }

  }

  else{

    if (keyboardEvent == "b") {
        show_track_bb=!show_track_bb;
    }

  }

  static vpHomogeneousMatrix M;
  displayScore(confidence);
  broadcastTransform(transform, object_name, camera_tf);
  broadcastTransform_filter(transform, object_name, camera_tf);
  M = visp_bridge::toVispHomogeneousMatrix(filter_transform);
  vpDisplay::displayFrame(vpI, M, vpcam_info, 0.05, vpColor::none, 3);

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

  if (confidence < refilterThreshold)
  {
    buffer_x.clear();
    buffer_y.clear();
    buffer_z.clear();
    buffer_qw.clear();
    buffer_qx.clear();
    buffer_qy.clear();
    buffer_qz.clear();
  }

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
  bb[0] = future.bb[0];
  bb[1] = future.bb[1];
  bb[2] = future.bb[2];
  bb[3] = future.bb[3];

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

void MegaPoseClient::spin()
{
  ROS_INFO("Object name: %s", object_name.c_str());
  ROS_INFO("Subscribing to image topic: %s", image_topic.c_str());
  ROS_INFO("Subscribing to camera info topic: %s", camera_info_topic.c_str());  
  
  waitForImage();
  if (depth_enable)
   {
     ROS_INFO("Subscribing to depth topic: %s", depth_topic.c_str());
     waitForDepth();
   }

  vpDisplayX *d = NULL;
  d = new vpDisplayX();

  d->init(vpI); // init display
  vpDisplay::setTitle(vpI, "Display");

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

  while (ros::ok()) {
    vpDisplay::display(vpI);
    ros::spinOnce();
    optional<vpRect> detection = nullopt;

    if (!initialized) {

       buffer_x.clear();
       buffer_y.clear();
       buffer_z.clear();
       buffer_qw.clear();
       buffer_qx.clear();
       buffer_qy.clear();
       buffer_qz.clear();
       
       detection = detectObjectForInitMegaposeClick(reset_bb, object_name);
       
      if (detection && init_request_done) {

        visp_megapose::Init init_pose;
        init_pose.request.object_name = object_name;
        init_pose.request.topleft_i = detection->getTopLeft().get_i();
        init_pose.request.topleft_j = detection->getTopLeft().get_j();
        init_pose.request.bottomright_i = detection->getBottomRight().get_i();
        init_pose.request.bottomright_j = detection->getBottomRight().get_j();
        init_pose.request.image = *rosI;
        init_pose.request.camera_info = *roscam_info;
        init_pose.request.depth_enable = depth_enable;

        if (depth_enable) { 

          init_pose.request.depth = *rosD;

        }
        else {

           init_pose.request.depth = sensor_msgs::Image();

        }

        if (init_pose_client.call(init_pose)) {

           init_service_response_callback(init_pose.response);

        } 
        else {

              ROS_WARN("Init server down, exiting...");
              ros::shutdown();

           }
      }
    } else if (initialized) {

        visp_megapose::Track track_pose;
        track_pose.request.object_name = object_name;
        track_pose.request.init_pose = transform;
        track_pose.request.image = *rosI;

        if (depth_enable) {

          track_pose.request.depth = *rosD;

        }
        else {

          track_pose.request.depth = sensor_msgs::Image();

        }

        if (track_pose_client.call(track_pose)) {

            track_service_response_callback(track_pose.response);

        } 
        else {
              ROS_WARN("Tracking server down, exiting...");
              ros::shutdown();
        }
      
        if (overlayModel) {

          visp_megapose::Render render_request;
          render_request.request.object_name = object_name;
          render_request.request.pose = filter_transform;

          if (render_client.call(render_request)) {

            render_service_response_callback(render_request.response);

          } 
          else {

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