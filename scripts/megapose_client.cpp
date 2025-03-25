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

// ROS core includes
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/Image.h>

// ROS visp_megapose service and message includes
#include <visp_megapose/Init.h>
#include <visp_megapose/Track.h>
#include <visp_megapose/Render.h>
#include <visp_megapose/BB3D.h>

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
  BB3D,
  LOAD
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
  string camera_tf;
  string object_name;
  string depth_topic;
  string bb3d_topic;
  string detector_method;
  bool depth_enable;
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
  bool got_bb3d;
  double confidence;
  unsigned width, height, widthD, heightD;

  json info;
  float bb[4];

  vpImage<vpRGBa> overlay_img;
  vpImage<vpRGBa> vpI;        
  vpCameraParameters vpcam_info;
  optional<vpRect> detection;
  deque<double> buffer_x, buffer_y, buffer_z, buffer_qw, buffer_qx, buffer_qy, buffer_qz;
  deque<double> buffer_bb1, buffer_bb2, buffer_bb3, buffer_bb4;

  // ROS variables

  sensor_msgs::CameraInfoConstPtr roscam_info;
  visp_megapose::BB3D bb3d_msg;

  boost::shared_ptr<const sensor_msgs::Image> rosI; 
  boost::shared_ptr<const sensor_msgs::Image> rosD; 

  geometry_msgs::Transform transform; 
  geometry_msgs::Transform filter_transform;

  // Message filters for synchronization
  message_filters::Subscriber<sensor_msgs::Image> image_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicyRGB;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image> SyncPolicyRGBD;

  boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGB>> sync_rgb;
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicyRGBD>> sync_rgbd;

  // ROS subscriber for 3D bounding box messages
  ros::Subscriber bb3d_sub;

  // Functions

  void waitForData(const string &data_type);
  void frameCallback_rgb(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &camera_info);
  void frameCallback_rgbd(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::CameraInfoConstPtr &cam_info, const sensor_msgs::ImageConstPtr &depth);
  void broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id, const string &camera_tf);
  void broadcastTransform_filter(const geometry_msgs::Transform &origpose, const string &child_frame_id, const string &camera_tf);
  void boundingbox_filter(const float (&bb)[4]);
  double calculateMovingAverage(const deque<double>& buffer);
  void displayScore(float);
  void overlayRender(const vpImage<vpRGBa> &overlay);
  void displayEvent(const optional<vpRect> &detection);
  void savePose();
  void displayInitBoundingBox(); 
  void displayTrackingBoundingBox(); 
  vpColor interpolate(const vpColor &low, const vpColor &high, const float f);
  optional<vpRect> detectObjectForInitMegaposeClick(const string &object_name);
  optional<vpRect> detectObjectForInitMegaposeLoad(const string &object_name);
  optional<vpRect> detectObjectForInitMegaposeBB3D(const visp_megapose::BB3D &bb_msg);
  DetectionMethod getDetectionMethodFromString(const std::string &str);

  void init_service_response_callback(const visp_megapose::Init::Response &future);
  void track_service_response_callback(const visp_megapose::Track::Response &future);
  void render_service_response_callback(const visp_megapose::Render::Response &future);

  void BB3DCallback(const visp_megapose::BB3D &bb3d);

public:
MegaPoseClient(ros::NodeHandle *nh) 
  : nh_(*nh),
    initialized(false),
    init_request_done(true),
    got_image(false),
    got_depth(false),
    got_bb3d(false),
    overlayModel(false),
    show_init_bb(false),
    show_track_bb(false),
    flag_track(false),
    flag_render(false)
{
  // Retrieve the username for constructing the megapose directory path
  char username[32];
  cuserid(username);
  user = std::string(username);
  megapose_directory = "/home/" + user + "/catkin_ws/src/visp_megapose"; // Adjust this path as needed

  // Load parameters from the ROS parameter server
 
  ros::param::get("image_topic", image_topic);             // Name of the image topic
  ros::param::get("camera_info_topic", camera_info_topic); // Name of the camera info topic
  ros::param::get("camera_tf", camera_tf);                 // Name of the camera frame
  ros::param::get("object_name", object_name);             // Name of the object model
  ros::param::get("depth_enable", depth_enable);           // Whether to use depth image
  ros::param::get("depth_topic", depth_topic);             // Name of the depth image topic
  ros::param::get("reinitThreshold", reinitThreshold);     // Reinit threshold for init and track service
  ros::param::get("refilterThreshold", refilterThreshold); // Filter threshold for filter poses
  ros::param::get("buffer_size", buffer_size);             // Buffer size for filter poses
  ros::param::get("detector_method", detector_method);     // Detection method to use
  ros::param::get("bb3d_topic", bb3d_topic);               // Name of the 3D bounding box topic

  // Subscribe to image and camera info topics
  image_sub.subscribe(nh_, image_topic, 1);
  camera_info_sub.subscribe(nh_, camera_info_topic, 1);

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
  ROS_INFO("Waiting for %s...", data_type.c_str());

  while (ros::ok())
  {
    if ((data_type == "image" && got_image) ||
        (data_type == "depth" && got_depth) ||
        (data_type == "BB3D" && got_bb3d))
    {
      ROS_INFO("Got %s!", data_type.c_str());
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

void MegaPoseClient::broadcastTransform(const geometry_msgs::Transform &transform, const string &child_frame_id, const string &camera_tf)
{
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  // Populate the transformStamped message
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = camera_tf;
  transformStamped.child_frame_id = child_frame_id;
  transformStamped.transform = transform;

  // Broadcast the transform
  br.sendTransform(transformStamped);
}

void MegaPoseClient::broadcastTransform_filter(const geometry_msgs::Transform &origpose, const string &child_frame_id, const string &camera_tf)
{
  if (confidence <= refilterThreshold)
    return;

  // Maintain buffer size by removing oldest elements
  if (buffer_x.size() >= buffer_size) {
    for (auto &buffer : {&buffer_x, &buffer_y, &buffer_z, &buffer_qw, &buffer_qx, &buffer_qy, &buffer_qz}) {
      buffer->pop_front();
    }
  }

  // Add new values to buffers
  buffer_x.push_back(origpose.translation.x);
  buffer_y.push_back(origpose.translation.y);
  buffer_z.push_back(origpose.translation.z);
  buffer_qw.push_back(origpose.rotation.w);
  buffer_qx.push_back(origpose.rotation.x);
  buffer_qy.push_back(origpose.rotation.y);
  buffer_qz.push_back(origpose.rotation.z);

  // Calculate moving averages for filtered transform
  filter_transform.translation.x = calculateMovingAverage(buffer_x);
  filter_transform.translation.y = calculateMovingAverage(buffer_y);
  filter_transform.translation.z = calculateMovingAverage(buffer_z);
  filter_transform.rotation.w = calculateMovingAverage(buffer_qw);
  filter_transform.rotation.x = calculateMovingAverage(buffer_qx);
  filter_transform.rotation.y = calculateMovingAverage(buffer_qy);
  filter_transform.rotation.z = calculateMovingAverage(buffer_qz);

  // Broadcast the filtered transform
  static tf2_ros::TransformBroadcaster br2;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = camera_tf;
  transformStamped.child_frame_id = child_frame_id + "_filtered";
  transformStamped.transform = filter_transform;
  br2.sendTransform(transformStamped);
}

void MegaPoseClient::boundingbox_filter(const float (&bb)[4])
{
  if (confidence <= refilterThreshold)
    return;

  // Maintain buffer size by removing oldest elements
  if (buffer_bb1.size() >= buffer_size)
  {
    for (auto &buffer : {&buffer_bb1, &buffer_bb2, &buffer_bb3, &buffer_bb4})
    {
      buffer->pop_front();
    }
  }

  // Add new bounding box values to buffers
  buffer_bb1.push_back(bb[0]);
  buffer_bb2.push_back(bb[1]);
  buffer_bb3.push_back(bb[2]);
  buffer_bb4.push_back(bb[3]);

  // Update the filtered bounding box using moving averages
  this->bb[0] = calculateMovingAverage(buffer_bb1);
  this->bb[1] = calculateMovingAverage(buffer_bb2);
  this->bb[2] = calculateMovingAverage(buffer_bb3);
  this->bb[3] = calculateMovingAverage(buffer_bb4);
}

double MegaPoseClient::calculateMovingAverage(const deque<double>& buffer)
{
  if (buffer.empty()) {
    ROS_WARN("Buffer is empty, returning 0.0 as moving average.");
    return 0.0;  // Avoid division by zero
  }
  return std::accumulate(buffer.begin(), buffer.end(), 0.0) / static_cast<double>(buffer.size());
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
  vpDisplay::displayText(vpI, 20, 20, "q: Quit | t: Reinitialize | b/n: Toggle BBox | s: Save Pose | r: Render", vpColor::red);

  string keyboardEvent;
  const bool keyPressed = vpDisplay::getKeyboardEvent(vpI, keyboardEvent, false);

  if (keyPressed) {
    if (keyboardEvent == "q") {
      ros::shutdown();
    } else if (keyboardEvent == "t") {
      initialized = false;
      init_request_done = true;
      detector_method = "CLICK";
      ROS_INFO("Reinitialize...");
    } else if (keyboardEvent == "s") {
      savePose();
    } else if (keyboardEvent == "r") {
      overlayModel = !overlayModel;
      flag_render = false;
    } else if (keyboardEvent == "n") {
      show_init_bb = !show_init_bb;
    } else if (keyboardEvent == "b") {
      show_track_bb = !show_track_bb;
    }
  }

  if (show_init_bb) {
    displayInitBoundingBox();
  }

  if (show_track_bb) {
    displayTrackingBoundingBox();
  }

  static vpHomogeneousMatrix M;
  displayScore(confidence);
  broadcastTransform(transform, object_name, camera_tf);
  broadcastTransform_filter(transform, object_name, camera_tf);
  M = visp_bridge::toVispHomogeneousMatrix(filter_transform);
  vpDisplay::displayFrame(vpI, M, vpcam_info, 0.05, vpColor::none, 3);
}

void MegaPoseClient::savePose() 
{
  try {
    // Construct the output file path
    string output_file_path = megapose_directory + "/output/pose/" + object_name + "_pose.json";

    // Open the file for writing
    ofstream output_file(output_file_path, ios::out);
    if (!output_file.is_open()) {
      ROS_ERROR("Failed to open file for writing: %s", output_file_path.c_str());
      return;
    }

    // Prepare the JSON object
    json outJson;
    outJson["object_name"] = object_name;
    outJson["position"] = {transform.translation.x, transform.translation.y, transform.translation.z};
    outJson["rotation"] = {transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w};

    // Write the JSON object to the file
    output_file << outJson.dump(4);
    output_file.close();

    ROS_INFO("Pose saved successfully to %s", output_file_path.c_str());
  } catch (const std::exception &e) {
    ROS_ERROR("An error occurred while saving the pose: %s", e.what());
  }
}

void MegaPoseClient::displayInitBoundingBox() 
{
  try {
    // Construct the file path for the bounding box JSON
    string bb_file_path = megapose_directory + "/output/bb/" + object_name + "_bb.json";

    // Open the file for reading
    ifstream bb_file(bb_file_path, ios::in);
    if (!bb_file.is_open()) {
      ROS_WARN("Failed to open bounding box file for object: %s", object_name.c_str());
      return;
    }

    // Parse the JSON content
    json bb_in;
    bb_file >> bb_in;
    bb_file.close();

    // Extract bounding box coordinates
    vpImagePoint init_topLeft(bb_in["point1"][0], bb_in["point1"][1]);
    vpImagePoint init_bottomRight(bb_in["point2"][0], bb_in["point2"][1]);

    // Display the bounding box on the image
    vpDisplay::displayRectangle(vpI, init_topLeft, init_bottomRight, vpColor::blue, false, 2);
  } catch (const std::exception &e) {
    ROS_ERROR("Error displaying initial bounding box: %s", e.what());
  }
}

void MegaPoseClient::displayTrackingBoundingBox() 
{
  // Apply bounding box filtering
  boundingbox_filter(bb);

  // Define top-left and bottom-right points for the bounding box
  const vpImagePoint topLeft(bb[1], bb[0]);
  const vpImagePoint bottomRight(bb[3], bb[2]);

  // Display the bounding box on the image
  vpDisplay::displayRectangle(vpI, topLeft, bottomRight, vpColor::red, false, 2);

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

optional<vpRect> MegaPoseClient::detectObjectForInitMegaposeBB3D(const visp_megapose::BB3D &bb_msg)
{ 
    double dim_x = bb3d_msg.dim_x;
    double dim_y = bb3d_msg.dim_y;
    double dim_z = bb3d_msg.dim_z;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf q(bb3d_msg.pose.rotation.w, bb3d_msg.pose.rotation.x, bb3d_msg.pose.rotation.y, bb3d_msg.pose.rotation.z);
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T.block<3,1>(0,3) << bb3d_msg.pose.translation.x, bb3d_msg.pose.translation.y, bb3d_msg.pose.translation.z;

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
  ROS_INFO("Bounding box generated. Checking confidence...");

  // Clear buffers if confidence is below the refilter threshold
  if (confidence < refilterThreshold) {
    for (auto &buffer : {&buffer_x, &buffer_y, &buffer_z, &buffer_qw, &buffer_qx, &buffer_qy, &buffer_qz, 
                         &buffer_bb1, &buffer_bb2, &buffer_bb3, &buffer_bb4}) {
      buffer->clear();
    }
  }

  // Handle reinitialization or successful initialization based on confidence
  if (confidence < reinitThreshold) {
    ROS_WARN("Initial pose not reliable (%.2f < %.2f). Reinitializing...", confidence, reinitThreshold);
  } else {
    initialized = true;
    init_request_done = false;
    flag_track = false;
    flag_render = false;
    ROS_INFO("Initialization successful with confidence: %.2f", confidence);
  }
}

void MegaPoseClient::track_service_response_callback(const visp_megapose::Track::Response &future)
{
  // Update the transform and confidence from the response
  transform = future.pose;
  confidence = future.confidence;

  // Update the bounding box coordinates from the response
  std::copy(std::begin(future.bb), std::end(future.bb), std::begin(bb));

  // Handle tracking status based on confidence
  if (confidence < reinitThreshold) {
    initialized = false;
    init_request_done = true;
    flag_track = false;
    flag_render = false;

    ROS_WARN("Tracking lost. Confidence below threshold (%.2f < %.2f). Reinitializing...", confidence, reinitThreshold);
  } else {
    if (!flag_track) {
      ROS_INFO("Object tracked successfully with confidence: %.2f", confidence);
      flag_track = true; // Log success only once
    }
  }
}

void MegaPoseClient::render_service_response_callback(const visp_megapose::Render::Response &future)
{
    // Convert the received image to ViSP format
    overlay_img = visp_bridge::toVispImageRGBa(future.image);

    // Check if the overlay image has valid size
    if (overlay_img.getSize() > 0) {
      overlayRender(overlay_img);

      // Log success message only once
      if (!flag_render) {
        ROS_INFO("Model rendered successfully!");
        flag_render = true;
      }
    } else {
      ROS_WARN("Received an empty overlay image. Skipping rendering.");
    }
}

void MegaPoseClient::BB3DCallback(const visp_megapose::BB3D &bb3d)
{

  // if (condition)
  // {
  //   /* code */
  // }

  // Convert 3D bounding box to 2D bounding box

  ROS_INFO("3D Bounding box pose: Translation (%f, %f, %f), Rotation (%f, %f, %f, %f), Dimensions: (%f, %f, %f)", 
           bb3d.pose.translation.x, bb3d.pose.translation.y, bb3d.pose.translation.z, 
           bb3d.pose.rotation.x, bb3d.pose.rotation.y, bb3d.pose.rotation.z, bb3d.pose.rotation.w,
           bb3d.dim_x, bb3d.dim_y, bb3d.dim_z);
  
  // Save BB3D message
  bb3d_msg = bb3d;

  got_bb3d = true;

}

void MegaPoseClient::spin()
{
  ROS_INFO("Object name: %s", object_name.c_str());
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
    waitForData("BB3D");
    bb3d_sub.shutdown();
  }

  vpDisplayX *d = NULL;
  d = new vpDisplayX();

  d->init(vpI);
  vpDisplay::setTitle(vpI, "Display");

  ros::ServiceClient init_pose_client = nh_.serviceClient<visp_megapose::Init>("init_pose");
  ros::ServiceClient track_pose_client = nh_.serviceClient<visp_megapose::Track>("track_pose");
  ros::ServiceClient render_client = nh_.serviceClient<visp_megapose::Render>("render_object");

  // Wait for all required services to become available
  while (ros::ok()) {
    if (init_pose_client.waitForExistence(ros::Duration(10)) &&
        track_pose_client.waitForExistence(ros::Duration(10)) &&
        render_client.waitForExistence(ros::Duration(10))) {
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

      for (auto &buffer : {&buffer_x, &buffer_y, &buffer_z, &buffer_qw, &buffer_qx, &buffer_qy, &buffer_qz, 
                           &buffer_bb1, &buffer_bb2, &buffer_bb3, &buffer_bb4}) {
        buffer->clear();
      }

      DetectionMethod method = getDetectionMethodFromString(detector_method);

      switch (method)
      {
        case BB3D:
          detection = detectObjectForInitMegaposeBB3D(bb3d_msg);
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