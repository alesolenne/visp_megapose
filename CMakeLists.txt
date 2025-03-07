cmake_minimum_required(VERSION 3.0.2)
project(visp_megapose)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  roscpp
  sensor_msgs
  tf2
  visp_bridge
  message_generation
  std_msgs
  message_filters
  tf2_ros
)
find_package(VISP REQUIRED)
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  Object.msg
)

add_service_files(
  FILES
    Init.srv
    Track.srv
    Render.srv
  )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   geometry_msgs sensor_msgs std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES visp_megapose_pkg
  CATKIN_DEPENDS geometry_msgs rospy sensor_msgs tf2 visp_bridge message_runtime message_filters tf2_ros 
  DEPENDS
   VISP) 


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(megapose_client scripts/megapose_client.cpp)
add_executable(megapose_client_detection scripts/megapose_client_detection.cpp)
add_executable(client2 scripts/client2.cpp)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/visp_megapose.cpp
# )


## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(megapose_client ${${PROJECT_NAME}_EXPORTED_TARGETS}  ${catkin_EXPORTED_TARGETS})
add_dependencies(megapose_client_detection ${${PROJECT_NAME}_EXPORTED_TARGETS}  ${catkin_EXPORTED_TARGETS})
add_dependencies(client2 ${${PROJECT_NAME}_EXPORTED_TARGETS}  ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(megapose_client
   ${roscpp_LIBRARIES}
   ${catkin_LIBRARIES}
   ${VISP_LIBS}
 )
 target_link_libraries(megapose_client_detection
 ${roscpp_LIBRARIES}
 ${catkin_LIBRARIES}
 ${VISP_LIBS}
)
target_link_libraries(client2
${roscpp_LIBRARIES}
${catkin_LIBRARIES}
${VISP_LIBS}
)