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
#include <visp_megapose/ObjectName.h>
#include <visp_megapose/PoseResult.h>

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
  string object_name;
  string depth_topic;
  bool depth_enable;
  double reinitThreshold;

  // Variables
  bool initialized;
  bool init_request_done;
  bool overlayModel;
  bool got_image;
  bool got_depth;
  bool got_name;
  double confidence;
  unsigned width, height, widthD, heightD;
  int n_object;
  int object_found;

  json info;

  vpImage<vpRGBa> overlay_img;
  vpImage<vpRGBa> vpI;
  vpCameraParameters vpcam_info;
  optional<vpRect> detection;

  // ROS variables
  sensor_msgs::CameraInfoConstPtr roscam_info;

  boost::shared_ptr<const sensor_msgs::Image> rosI;
  boost::shared_ptr<const sensor_msgs::Image> rosD; 

  geometry_msgs::Transform transform;

  ros::Subscriber obj_sub;
  ros::Publisher pub_pose;

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
  
  void init_service_response_callback(const visp_megapose::Init::Response& future);
  void track_service_response_callback(const visp_megapose::Track::Response& future);
  void render_service_response_callback(const visp_megapose::Render::Response& future);

  void overlayRender(const vpImage<vpRGBa> &overlay);
  vpColor interpolate(const vpColor &low, const vpColor &high, const float f);
  optional<vpRect> detectObjectForInitMegaposeClick(const string &object_name);

  void waitForName();
  void frameObject(const visp_megapose::ObjectName &command);

public:
MegaPoseClient(ros::NodeHandle *nh) 
{ 
 //Parameters:

 //     image_topic(string) : Name of the image topic
 //     camera_info_topic(string) : Name of the camera info topic
 //     object_name(string) : Name of the object model
 //     depth_enable(bool) : Whether to use depth image
 //     depth_topic(string) : Name of the depth image topic
 //     reinitThreshold(double) : Reinit threshold for init and track service

  this->nh_ = *nh;
  char username[32];
  cuserid(username);
  std::string user(username);
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose"; //change this path to your catkin workspace path

  initialized = false;
  init_request_done = true;
  got_image = false;
  overlayModel = false;
  got_name = false;
  object_found = 0;

  ros::param::get("image_topic", image_topic);
  ros::param::get("camera_info_topic", camera_info_topic);
  ros::param::get("depth_enable", depth_enable);
  ros::param::get("depth_topic", depth_topic);
  ros::param::get("reinitThreshold", reinitThreshold);

  image_sub.subscribe(*nh, image_topic, 1);
  camera_info_sub.subscribe(*nh, camera_info_topic, 1);
  obj_sub = nh->subscribe("ObjectList", 1, &MegaPoseClient::frameObject, this);
  pub_pose = nh->advertise<visp_megapose::PoseResult>("PoseResult", 1, true);

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
      ROS_INFO("Got image!");
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

vpColor MegaPoseClient::interpolate(const vpColor &low, const vpColor &high, const float f)
{
  const float r = ((float)high.R - (float)low.R) * f;
  const float g = ((float)high.G - (float)low.G) * f;
  const float b = ((float)high.B - (float)low.B) * f;
  return vpColor((unsigned char)r, (unsigned char)g, (unsigned char)b);
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
              pub_pose.publish(res);
              object_found = object_found + 1;
              ROS_INFO("Object %s found! \n Pose: [%f, %f, %f, %f, %f, %f, %f] ", object_name.c_str(), transform.translation.x, transform.translation.y, transform.translation.z, transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
              ROS_INFO("Pose confidence: %f: ", confidence);

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

              got_name = false;

         }

}

void MegaPoseClient::render_service_response_callback(const visp_megapose::Render::Response& future)
{
  overlay_img = visp_bridge::toVispImageRGBa(future.image);
  if (overlay_img.getSize() > 0){
    overlayRender(overlay_img);
    ROS_INFO("Model rendered successfully!");
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

  d->init(vpI);
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
          render_request.request.pose = transform;

          if (render_client.call(render_request)) {

            render_service_response_callback(render_request.response);

          } 
          else {

            ROS_ERROR("Render server down, exiting...");
            ros::shutdown();

          }
        }

    }

  vpDisplay::flush(vpI);
  
  if (object_found == n_object)
  {
    ROS_WARN("All object in the list found!");
    ros::shutdown();
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