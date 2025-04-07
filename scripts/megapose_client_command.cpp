// Standard includes
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

// Include Eigen library for matrix and vector operations
#include <Eigen/Dense>

// ViSP and OpenCV includes
#include <visp3/core/vpTime.h>
#include <visp3/gui/vpDisplayX.h>
#include <opencv2/opencv.hpp>

// ViSP bridge includes
#include <visp_bridge/3dpose.h>
#include <visp_bridge/camera.h>
#include <visp_bridge/image.h>

// ROS includes
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/Image.h>

// ROS visp_megapose service and message includes
#include <visp_megapose/Init.h>
#include <visp_megapose/Track.h>
#include <visp_megapose/Render.h>
#include <visp_megapose/ObjectName.h>
#include <visp_megapose/PoseResult.h>

// Include JSK recognition messages for handling bounding boxes
#include <jsk_recognition_msgs/BoundingBox.h>       
#include <jsk_recognition_msgs/BoundingBoxArray.h>

// ROS message filter includes
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Include deque for buffer management
#include <deque>

using namespace std;
using namespace nlohmann;

enum DetectionMethod
{
  UNKNOWN,
  CLICK,
  LOAD,
  BB3D
};

std::map<std::string, DetectionMethod> stringToDetectionMethod = {
  {"UNKNOWN", UNKNOWN},
  {"CLICK", CLICK},
  {"LOAD", LOAD},
  {"BB3D", BB3D}};

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
  string bb3d_topic;
  string detector_method;
  bool depth_enable;
  double reinitThreshold;

  // Variables
  bool initialized;
  bool init_request_done;
  bool got_image;
  bool got_depth;
  bool got_name;
  bool got_bb3d;
  double confidence;
  unsigned width, height, widthD, heightD;
  int n_object;
  int object_found;

  json info;

  vpImage<vpRGBa> vpI;
  vpCameraParameters vpcam_info;
  optional<vpRect> detection;

  // ROS variables

  sensor_msgs::CameraInfoConstPtr roscam_info;
  jsk_recognition_msgs::BoundingBox bb3d_msg;

  boost::shared_ptr<const sensor_msgs::Image> rosI;
  boost::shared_ptr<const sensor_msgs::Image> rosD; 

  geometry_msgs::Transform transform;

  ros::Subscriber obj_sub;
  ros::Publisher pub_pose;

  // Message filters for synchronization
  message_filters::Subscriber<sensor_msgs::Image> image_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub;
 
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicyRGB;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image> SyncPolicyRGBD;

  boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGB> > sync_rgb;
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGBD> > sync_rgbd;

  // ROS subscriber for 3D bounding box messages
  ros::Subscriber bb3d_sub;

  // Functions

  void waitForData(const string &data_type);
  void frameCallback_rgb(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camera_info);
  void frameCallback_rgbd(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info, const sensor_msgs::ImageConstPtr &depth);
  optional<vpRect> detectObjectForInitMegaposeClick(const string &object_name);
  optional<vpRect> detectObjectForInitMegaposeLoad(const string &object_name);
  optional<vpRect> detectObjectForInitMegaposeBB3D(const jsk_recognition_msgs::BoundingBox &bb_msg);
  DetectionMethod getDetectionMethodFromString(const std::string &str);

  void init_service_response_callback(const visp_megapose::Init::Response &future);
  void track_service_response_callback(const visp_megapose::Track::Response &future);

  void frameObject(const visp_megapose::ObjectName &command);
  void BB3DCallback(const jsk_recognition_msgs::BoundingBoxArray &bb_array);

public:
MegaPoseClient(ros::NodeHandle *nh)
  : nh_(*nh),
    initialized(false),
    init_request_done(true),
    got_image(false),
    got_depth(false),
    got_bb3d(false),
    object_found(0)
{ 
  // Retrieve the username for constructing the megapose directory path
  char username[32];
  cuserid(username);
  std::string user(username);
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose"; // Adjust this path as needed

  // Load parameters from the ROS parameter server

  ros::param::get("image_topic", image_topic);                     // image_topic(string): Name of the image topic
  ros::param::get("camera_info_topic", camera_info_topic);         // camera_info_topic(string): Name of the camera info topic
  ros::param::get("depth_enable", depth_enable);                   // depth_enable(bool): Whether to use depth image
  ros::param::get("depth_topic", depth_topic);                     // depth_topic(string): Name of the depth image topic
  ros::param::get("reinitThreshold", reinitThreshold);             // reinitThreshold(double): Reinit threshold for init and track service
  ros::param::get("detector_method", detector_method);             // Detection method to use
  ros::param::get("bb3d_topic", bb3d_topic);                       // Name of the 3D bounding box topic

  // Subscribe to image and camera info topics
  image_sub.subscribe(*nh, image_topic, 1);
  camera_info_sub.subscribe(*nh, camera_info_topic, 1);

  // Subscribe to the "ObjectList" topic to receive object names and numbers
  obj_sub = nh->subscribe("ObjectList", 1, &MegaPoseClient::frameObject, this);

  // Advertise the "PoseResult" topic to publish the pose results of detected objects
  pub_pose = nh->advertise<visp_megapose::PoseResult>("PoseResult", 1, true);

  // Subscribe to BB3D topic if the detection method is BB3D
  if (getDetectionMethodFromString(detector_method) == BB3D)
  {
    bb3d_sub = nh_.subscribe(bb3d_topic, 1, &MegaPoseClient::BB3DCallback, this);
  }
  
  // Set up synchronization for RGB or RGBD topics
  if (!depth_enable)
  {
    sync_rgb = boost::make_shared<message_filters::Synchronizer<SyncPolicyRGB>>(SyncPolicyRGB(1), image_sub, camera_info_sub);
    sync_rgb->registerCallback(boost::bind(&MegaPoseClient::frameCallback_rgb, this, _1, _2));
  }
  else
  {
    depth_sub.subscribe(nh_, depth_topic, 1);
    sync_rgbd = boost::make_shared<message_filters::Synchronizer<SyncPolicyRGBD>>(SyncPolicyRGBD(1), image_sub, camera_info_sub, depth_sub);
    sync_rgbd->registerCallback(boost::bind(&MegaPoseClient::frameCallback_rgbd, this, _1, _2, _3));
  }

};

~MegaPoseClient() = default;

void spin();
};

void MegaPoseClient::waitForData(const string &data_type)
{
  ros::Rate loop_rate(10);
  if (data_type == "image" || data_type == "depth" || data_type == "BB3D") {
    ROS_INFO("Waiting for %s...", data_type.c_str());
  }

  while (ros::ok())
  {
    if ((data_type == "image" && got_image) ||
        (data_type == "depth" && got_depth) ||
        (data_type == "BB3D" && got_bb3d))
    {
      ROS_INFO("Got %s!", data_type.c_str());
      return;
    }
    if (data_type == "name" && got_name){
      return;
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
}

void MegaPoseClient::frameCallback_rgb(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info)
{

  // Store the received image and camera info
  rosI = image;
  roscam_info = cam_info;

  // Extract image dimensions
  width = image->width;
  height = image->height;

  // Convert ROS image and camera info to ViSP format
  vpI = visp_bridge::toVispImageRGBa(*image);
  vpcam_info = visp_bridge::toVispCameraParameters(*cam_info);

  // Update flag to indicate image availability
  got_image = true;

}

void MegaPoseClient::frameCallback_rgbd(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info, const sensor_msgs::ImageConstPtr &depth)
{

  // Store the received image, depth, and camera info
  rosI = image;
  rosD = depth;
  roscam_info = cam_info;

  // Extract dimensions for RGB and depth images
  width = image->width;
  height = image->height;
  widthD = depth->width;
  heightD = depth->height;

  // Convert ROS image and camera info to ViSP format
  vpI = visp_bridge::toVispImageRGBa(*image);
  vpcam_info = visp_bridge::toVispCameraParameters(*cam_info);

  // Update flags to indicate data availability
  got_image = true;
  got_depth = true;

}

optional<vpRect> MegaPoseClient::detectObjectForInitMegaposeClick(const string &object_name)
{
  vpImagePoint topLeft, bottomRight;
  const vpImagePoint textPosition(10.0, 20.0);

  // Wait for the user to start labeling
  if (vpDisplay::getClick(vpI, false)) {
    // Prompt user to click the upper left corner
    vpDisplay::displayText(vpI, textPosition, "Click the upper left corner of the bounding box", vpColor::red);
    vpDisplay::flush(vpI);
    vpDisplay::getClick(vpI, topLeft, true);

    // Display the selected point
    vpDisplay::display(vpI);
    vpDisplay::displayCross(vpI, topLeft, 5, vpColor::red, 2);

    // Prompt user to click the bottom right corner
    vpDisplay::displayText(vpI, textPosition, "Click the bottom right corner of the bounding box", vpColor::red);
    vpDisplay::flush(vpI);
    vpDisplay::getClick(vpI, bottomRight, true);

    // Save bounding box coordinates to a JSON file
    ofstream bb_file(megapose_directory + "/output/bb/" + object_name + "_bb.json", ios::out);
    if (bb_file.is_open()) {
      json bb_out;
      bb_out["object_name"] = object_name;
      bb_out["point1"] = {topLeft.get_i(), topLeft.get_j()};
      bb_out["point2"] = {bottomRight.get_i(), bottomRight.get_j()};
      bb_file << bb_out.dump(4);
      bb_file.close();
    } else {
      ROS_WARN("Failed to open bounding box file for writing: %s", object_name.c_str());
    }

    // Return the bounding box
    return vpRect(topLeft, bottomRight);
  } else {
    // Display a message prompting the user to click when ready
    vpDisplay::display(vpI);
    vpDisplay::displayText(vpI, textPosition, "Click when the object is visible and static to start reinitializing megapose.", vpColor::red);
    vpDisplay::flush(vpI);
    return nullopt;
  }
}

optional<vpRect> MegaPoseClient::detectObjectForInitMegaposeLoad(const string &object_name)
{

    ifstream bb_file(megapose_directory + "/output/bb/" + object_name + "_bb.json", ios::in);
    if (!bb_file.is_open()) {
      ROS_WARN("Failed to open bounding box file for object: %s", object_name.c_str());
      return nullopt;
    }
    
    json bb_in;
    bb_file >> bb_in;

    vpImagePoint topLeft(bb_in["point1"][0], bb_in["point1"][1]);
    vpImagePoint bottomRight(bb_in["point2"][0], bb_in["point2"][1]);
    vpRect bb(topLeft, bottomRight);
    
    bb_file.close();
    return vpRect(topLeft, bottomRight);

}

optional<vpRect> MegaPoseClient::detectObjectForInitMegaposeBB3D(const jsk_recognition_msgs::BoundingBox &bb_msg)
{ 
    double dim_x = bb_msg.dimensions.x;
    double dim_y = bb_msg.dimensions.y;
    double dim_z = bb_msg.dimensions.z;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf q(bb_msg.pose.orientation.w, bb_msg.pose.orientation.x, bb_msg.pose.orientation.y, bb_msg.pose.orientation.z);
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T.block<3,1>(0,3) << bb_msg.pose.position.x, bb_msg.pose.position.y, bb_msg.pose.position.z;

    Eigen::Vector4f p1 ( dim_x / 2,  dim_y / 2,  dim_z / 2, 1);
    Eigen::Vector4f p2 ( dim_x / 2,  dim_y / 2, -dim_z / 2, 1);
    Eigen::Vector4f p3 ( dim_x / 2, -dim_y / 2,  dim_z / 2, 1);
    Eigen::Vector4f p4 ( dim_x / 2, -dim_y / 2, -dim_z / 2, 1);
    Eigen::Vector4f p5 (-dim_x / 2,  dim_y / 2,  dim_z / 2, 1);
    Eigen::Vector4f p6 (-dim_x / 2,  dim_y / 2, -dim_z / 2, 1);
    Eigen::Vector4f p7 (-dim_x / 2, -dim_y / 2,  dim_z / 2, 1);
    Eigen::Vector4f p8 (-dim_x / 2, -dim_y / 2, -dim_z / 2, 1);

    // Transform 3D points to camera coordinates
    std::vector<Eigen::Vector4f> points = {p1, p2, p3, p4, p5, p6, p7, p8};
    std::vector<cv::Point3f> object_points(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      Eigen::Vector4f transformed_point = T * points[i];
      object_points[i] = cv::Point3f(transformed_point(0), transformed_point(1), transformed_point(2));
    }

    // Camera matrix and distortion coefficients
    cv::Mat cam_matrix = (cv::Mat_<double>(3, 3) << roscam_info->K[0], roscam_info->K[1], roscam_info->K[2], 
                roscam_info->K[3], roscam_info->K[4], roscam_info->K[5], 
                roscam_info->K[6], roscam_info->K[7], roscam_info->K[8]);
    cv::Mat distortion;
    if (roscam_info->distortion_model == "plumb_bob") {
      distortion = (cv::Mat_<double>(1, 5) << roscam_info->D[0], roscam_info->D[1], roscam_info->D[2], roscam_info->D[3], roscam_info->D[4]);
    } else if (roscam_info->distortion_model == "rational_polynomial") {
      distortion = (cv::Mat_<double>(1, 8) << roscam_info->D[0], roscam_info->D[1], roscam_info->D[2], roscam_info->D[3], roscam_info->D[4], roscam_info->D[5], roscam_info->D[6], roscam_info->D[7]);
    } else {
      ROS_WARN("Unknown distortion model: %s", roscam_info->distortion_model.c_str());
      distortion = cv::Mat::zeros(1, 5, CV_64F); // Default to zero distortion if unknown model
    }

    // Project 3D points to 2D image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(object_points, cv::Vec3d::zeros(), cv::Vec3d::zeros(), cam_matrix, distortion, image_points);

    // Initialize bounding box coordinates
    float u_p_min = std::numeric_limits<float>::max();
    float v_p_min = std::numeric_limits<float>::max();
    float u_p_max = std::numeric_limits<float>::lowest();
    float v_p_max = std::numeric_limits<float>::lowest();

    // Calculate bounding box coordinates
    for (const auto& point : image_points) {
      u_p_min = std::min(u_p_min, point.x);
      v_p_min = std::min(v_p_min, point.y);
      u_p_max = std::max(u_p_max, point.x);
      v_p_max = std::max(v_p_max, point.y);
    }

    // Output bounding box coordinates
    ROS_INFO("2D Bounding box coordinates convertion: (%f, %f) to (%f, %f)", v_p_min, u_p_min, v_p_max, u_p_max);

    vpImagePoint topLeft, bottomRight;

    topLeft = vpImagePoint(v_p_min, u_p_min);
    bottomRight = vpImagePoint(v_p_max, u_p_max);
    
    // Save bounding box coordinates to a JSON file
    ofstream bb_file(megapose_directory + "/output/bb/" + object_name + "_bb.json", ios::out);
    if (bb_file.is_open()) {
      json bb_out;
      bb_out["object_name"] = object_name;
      bb_out["point1"] = {v_p_min, u_p_min};
      bb_out["point2"] = {v_p_max, u_p_max};
      bb_file << bb_out.dump(4);
      bb_file.close();
    } else {
      ROS_WARN("Failed to open bounding box file for writing: %s", object_name.c_str());
    }

    return vpRect(topLeft, bottomRight);

}

DetectionMethod MegaPoseClient::getDetectionMethodFromString(const std::string &str)
{
  if (stringToDetectionMethod.find(str) != stringToDetectionMethod.end())
  {
    return stringToDetectionMethod[str];
  }
  return UNKNOWN;
};

void MegaPoseClient::init_service_response_callback(const visp_megapose::Init::Response &future)
{
// Update the transform and confidence from the response
transform = future.pose;
confidence = future.confidence;
ROS_INFO("Bounding box generated, checking the confidence");

// Handle reinitialization or successful initialization based on confidence
if (confidence < reinitThreshold) {
    ROS_WARN("Initial pose not reliable (%.2f < %.2f). Reinitializing...", confidence, reinitThreshold);
} else {
    initialized = true;
    init_request_done = false;
    ROS_INFO("Initialization successful with confidence: %.2f", confidence);
}
}

void MegaPoseClient::track_service_response_callback(const visp_megapose::Track::Response &future)
{
  // Update the transform and confidence from the response
  transform = future.pose;
  confidence = future.confidence;
  initialized = false;
  init_request_done = true;

  if (confidence < reinitThreshold) {

    ROS_WARN("Tracking lost. Confidence below threshold (%.2f < %.2f). Reinitializing...", confidence, reinitThreshold);
  } else {                      
              visp_megapose::PoseResult res;
              res.pose = transform;
              res.skip = true;
              pub_pose.publish(res);
              object_found = object_found + 1;
              ROS_WARN("Object %s found! \n Pose: [%f, %f, %f, %f, %f, %f, %f] ", object_name.c_str(), transform.translation.x, transform.translation.y, transform.translation.z, transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
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
              got_bb3d = false;

         }

}

void MegaPoseClient::frameObject(const visp_megapose::ObjectName &command)
{
  got_name = true;
  object_name = command.obj_name;
  n_object = command.number;
}

void MegaPoseClient::BB3DCallback(const jsk_recognition_msgs::BoundingBoxArray &bb_array)
{

  // Check if the bounding box array contains the boxes wanted
  if (bb_array.boxes[0].header.frame_id == object_name)
  {
      // Process the first bounding box in the array
      const auto &bb3d = bb_array.boxes[0];

      // Log the 3D bounding box pose and dimensions
      ROS_INFO("3D Bounding box pose: Translation (%f, %f, %f), Rotation (%f, %f, %f, %f), Dimensions: (%f, %f, %f)", 
              bb3d.pose.position.x, bb3d.pose.position.y, bb3d.pose.position.z, 
              bb3d.pose.orientation.x, bb3d.pose.orientation.y, bb3d.pose.orientation.z, bb3d.pose.orientation.w,
              bb3d.dimensions.x, bb3d.dimensions.y, bb3d.dimensions.z);

      // Save the BB3D message
      bb3d_msg = bb3d;

      // Update the flag to indicate that a bounding box has been received
      got_bb3d = true;
  }
  else{
    return;
  }

}

void MegaPoseClient::spin()
{
  ROS_INFO("Subscribing to image topic: %s", image_topic.c_str());
  ROS_INFO("Subscribing to camera info topic: %s", camera_info_topic.c_str());  

  waitForData("image");
  if (depth_enable)
   {
     ROS_INFO("Subscribing to depth topic: %s", depth_topic.c_str());
     waitForData("depth");
    }
  if (getDetectionMethodFromString(detector_method) == BB3D)
  {
    ROS_INFO("Subscribing to BB3D topic: %s", bb3d_topic.c_str());
  }

  if (getDetectionMethodFromString(detector_method) == CLICK) {

    vpDisplayX *d = NULL;
    d = new vpDisplayX();
  
    d->init(vpI);
    vpDisplay::setTitle(vpI, "Display");

  }

  ros::ServiceClient init_pose_client = nh_.serviceClient<visp_megapose::Init>("init_pose");
  ros::ServiceClient track_pose_client = nh_.serviceClient<visp_megapose::Track>("track_pose");

  // Wait for all required services to become available
  while (ros::ok()) {
    if (init_pose_client.waitForExistence(ros::Duration(10)) &&
        track_pose_client.waitForExistence(ros::Duration(10))) {
          ROS_INFO("All required services are available.");
          break;
    }
    ROS_WARN("Some services are still unavailable. Retrying...");
  }

  // Main processing loop
  while (ros::ok()) {
    vpDisplay::display(vpI);
    ros::spinOnce();

    if (!initialized) {

       optional<vpRect> detection = nullopt;
  
       waitForData("name");

       DetectionMethod method = getDetectionMethodFromString(detector_method);

       switch (method)
       {
         case BB3D:
           detection = detectObjectForInitMegaposeBB3D(bb3d_msg);
           waitForData("BB3D");
           break;
 
         case CLICK:
           detection = detectObjectForInitMegaposeClick(object_name);
           break;
 
         case LOAD:
           detection = detectObjectForInitMegaposeLoad(object_name);
           break;
 
         default:
           ROS_WARN("Unsupported detection method: %s", detector_method.c_str());
           ros::shutdown();
       }
       
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
            visp_megapose::Render render_request;
            render_request.request.object_name = object_name;
            render_request.request.pose = transform;
    
        } 
        else {
              ROS_WARN("Tracking server down, exiting...");
              ros::shutdown();
        }

    }

  if (getDetectionMethodFromString(detector_method) == CLICK) {
    vpDisplay::flush(vpI);
  }
  
  if (object_found == n_object)
  {
    ROS_WARN("All object in the list found!");
    ros::shutdown();
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