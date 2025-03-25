#!/usr/bin/env python3

import os
import getpass
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import rospy
from torch import nn
import torch
from visp_megapose.srv import Init, Track, Render, InitResponse, TrackResponse, RenderResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import transforms3d
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.lib3d.transform import Transform
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer

# Constants
NUM_WORKERS = 4
IMAGE_BATCH_SIZE = 256
DEFAULT_Z_NEAR = 0.001
DEFAULT_Z_FAR = 100000

class MegaPoseServer:
    def __init__(self):
        self._initialize_paths()
        self._load_camera_data()
        self._initialize_model()
        self._initialize_renderer()
        self._setup_ros_services()

    def _initialize_paths(self):
        user = getpass.getuser()
        self.main_folder = Path(f"/home/{user}/visp-ws/visp/script")
        self.megapose_folder = Path(f"/home/{user}/catkin_ws/src/visp_megapose")
        self.mesh_dir = self.megapose_folder / "data/models"

        assert self.mesh_dir.exists(), "Mesh directory does not exist, cannot start server"
        assert self.main_folder.exists(), "Main folder directory does not exist, cannot start server"

        os.environ['MEGAPOSE_DATA_DIR'] = str(self.main_folder / "megapose_server/megapose6d/data")

    def _load_camera_data(self):
        camera_json_path = self.megapose_folder / "params/camera.json"
        with open(camera_json_path, 'r') as f:
            data = json.load(f)

        self.camera_data = {
            'K': np.array([
                [data['K'][0][0], 0.0, data['K'][0][2]],
                [0.0, data['K'][1][1], data['K'][1][2]],
                [0.0, 0.0, 1.0]
            ]),
            'h': data['h'],
            'w': data['w']
        }

    def _initialize_model(self):
        megapose_models = {
            'RGB': ('megapose-1.0-RGB', False),
            'RGBD': ('megapose-1.0-RGBD', True),
            'RGB-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis', False),
            'RGBD-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis-icp', True)
        }

        model = rospy.get_param("~megapose_model")
        self.model_name, self.model_use_depth = megapose_models[model]

        self.object_dataset = self._make_object_dataset(self.mesh_dir)
        self.model_info, self.model = self._load_model(self.model_name)
        self.model.eval()
        self.model.bsz_images = IMAGE_BATCH_SIZE
        self.camera_data = self._make_camera_data(self.camera_data)

        rospy.loginfo(f"Model selected: {model}")
        rospy.loginfo(f"Camera resolution: {self.camera_data.resolution}")

    def _initialize_renderer(self):
        self.renderer = Panda3dSceneRenderer(self.object_dataset)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def _setup_ros_services(self):
        rospy.Service("init_pose", Init, self.InitPoseCallback)
        rospy.Service("track_pose", Track, self.TrackPoseCallback)
        rospy.Service("render_object", Render, self.RenderObjectCallback)
        rospy.loginfo("Waiting for requests...")

    def _load_model(self, model_name):
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=NUM_WORKERS).cuda()

    def _make_object_dataset(self, meshes_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        for object_dir in meshes_dir.iterdir():
            label = object_dir.name
            mesh_path = next((fn for fn in object_dir.glob("*") if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}), None)
            assert mesh_path, f"Couldn't find the mesh for {label}"
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units="m"))
        return RigidObjectDataset(rigid_objects)

    def _make_camera_data(self, camera_data: Dict) -> CameraData:
        c = CameraData()
        c.K = camera_data['K']
        c.resolution = (camera_data['h'], camera_data['w'])
        c.z_near = DEFAULT_Z_NEAR
        c.z_far = DEFAULT_Z_FAR
        return c

    def InitPoseCallback(self, request):
        # Callback for initializing the pose of an object
        # request: contains object_name, bounding box coordinates, image, camera_info, and depth_enable flag
        # response: returns the pose and confidence score

        bridge = CvBridge()

        # Extract camera calibration data from the request
        camera_data = {
            'K': np.asarray([
                [request.camera_info.K[0], request.camera_info.K[1], request.camera_info.K[2]],
                [request.camera_info.K[3], request.camera_info.K[4], request.camera_info.K[5]],
                [request.camera_info.K[6], request.camera_info.K[7], request.camera_info.K[8]]
            ]),
            'h': request.camera_info.height,
            'w': request.camera_info.width
        }
        self.camera_data = self._make_camera_data(camera_data)
        self.h = self.camera_data.resolution[0]
        self.w = self.camera_data.resolution[1]

        # Convert the image data from the request to a NumPy array
        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(self.h, self.w, 3)

        # Handle depth data if depth_enable is set
        if request.depth_enable:
            if not self.model_info['requires_depth']:
                rospy.loginfo('Trying to use depth with a model that cannot handle it')
                return
            else:
                depth_uint16 = bridge.imgmsg_to_cv2(request.depth, desired_encoding="passthrough")
                depth = depth_uint16.astype(np.float32) / 1000.0
        else:
            if not self.model_info['requires_depth']:
                depth = None
            else:
                rospy.loginfo('Trying to use a model that requires depth without providing it')
                return

        # Prepare object name and bounding box detections
        object_name = [request.object_name]
        detections = [[request.topleft_j, request.topleft_i, request.bottomright_j, request.bottomright_i]]
        detections = self._make_detections(object_name, detections).cuda()

        # Create an observation tensor from the image and depth data
        observation = self._make_observation_tensor(img, depth).cuda()

        # Run the inference pipeline to estimate the pose
        inference_params = self.model_info['inference_parameters'].copy()
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=None
        )
        poses = output.poses.cpu().numpy().reshape(len(output.poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()

        # Prepare the response with the estimated pose and confidence score
        response = InitResponse()
        response.pose.translation.x = float(poses[0][0, 3])
        response.pose.translation.y = float(poses[0][1, 3])
        response.pose.translation.z = float(poses[0][2, 3])
        rotation = transforms3d.quaternions.mat2quat(poses[0][0:3, 0:3])
        response.pose.rotation.x = rotation[1]
        response.pose.rotation.y = rotation[2]
        response.pose.rotation.z = rotation[3]
        response.pose.rotation.w = rotation[0]
        response.confidence = float(confidence[0])
        return response 

    def TrackPoseCallback(self, request):
        # Callback for tracking the pose of an object
        # request: contains object_name, initial pose, refiner iterations, and image
        # response: returns the updated pose, confidence score, and bounding box

        bridge = CvBridge()

        # Convert the image data from the request to a NumPy array
        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(self.h, self.w, 3)

        # Handle depth data if the model uses depth
        if self.model_use_depth:
            depth_uint16 = bridge.imgmsg_to_cv2(request.depth, desired_encoding="passthrough")
            depth = depth_uint16.astype(np.float32) / 1000.0
        else:
            depth = None

        # Prepare object name and initial pose
        object_name = [request.object_name]
        cTos_np = np.eye(4)
        cTos_np[0:3, 3] = [request.init_pose.translation.x, request.init_pose.translation.y, request.init_pose.translation.z]
        cTos_np[0:3, 0:3] = transforms3d.quaternions.quat2mat([request.init_pose.rotation.w, request.init_pose.rotation.x, request.init_pose.rotation.y, request.init_pose.rotation.z])
        cTos_np = cTos_np.reshape(1, 4, 4)
        tensor = torch.from_numpy(cTos_np).float().cuda()
        infos = pd.DataFrame.from_dict({
            'label': object_name,
            'batch_im_id': [0 for _ in range(len(cTos_np))],
            'instance_id': [i for i in range(len(cTos_np))]
        })
        coarse_estimates = PoseEstimatesType(infos, poses=tensor)

        # Run the inference pipeline to refine the pose
        detections = None
        observation = self._make_observation_tensor(img, depth).cuda()
        inference_params = self.model_info['inference_parameters'].copy()
        inference_params['n_refiner_iterations'] = 1
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=coarse_estimates
        )
        poses = output.poses.cpu().numpy().reshape(len(output.poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()
        bounding_boxes = extra_data['scoring']['preds'].tensors['boxes_rend'].cpu().numpy().reshape(-1, 4).tolist()

        # Prepare the response with the updated pose, confidence score, and bounding box
        response = TrackResponse()
        response.pose.translation.x = float(poses[0][0, 3])
        response.pose.translation.y = float(poses[0][1, 3])
        response.pose.translation.z = float(poses[0][2, 3])
        rotation = transforms3d.quaternions.mat2quat(poses[0][0:3, 0:3])
        response.pose.rotation.x = rotation[1]
        response.pose.rotation.y = rotation[2]
        response.pose.rotation.z = rotation[3]
        response.pose.rotation.w = rotation[0]
        response.confidence = float(confidence[0])
        response.bb = bounding_boxes[0]
        return response

    def RenderObjectCallback(self, request):
        # Callback for rendering an object in a scene
        # request: contains object_name and pose
        # response: returns the rendered image

        # Prepare object labels and poses
        labels = [request.object_name]
        poses = np.eye(4)
        poses[0:3, 3] = [request.pose.translation.x, request.pose.translation.y, request.pose.translation.z]
        poses[0:3, 0:3] = transforms3d.quaternions.quat2mat([request.pose.rotation.w, request.pose.rotation.x, request.pose.rotation.y, request.pose.rotation.z])
        poses = poses.reshape(1, 4, 4)

        # Configure camera data for rendering
        camera_data = CameraData()
        camera_data.K = self.camera_data.K
        camera_data.resolution = self.camera_data.resolution
        camera_data.TWC = Transform(np.eye(4))

        # Prepare object data for rendering
        object_datas = []
        for label, pose in zip(labels, poses):
            object_datas.append(ObjectData(label=label, TWO=Transform(pose)))
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

        # Configure lighting for rendering
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]

        # Render the scene using the Panda3D renderer
        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        # Convert the rendered image to a format suitable for the response
        img = renderings.rgb
        img = np.uint8(img).reshape(1, -1).tolist()[0]

        # Prepare the response with the rendered image
        response = RenderResponse()
        response.image.header.stamp = rospy.Time.now()
        response.image.height = renderings.rgb.shape[0]
        response.image.width = renderings.rgb.shape[1]
        response.image.encoding = 'rgb8'
        response.image.is_bigendian = 0
        response.image.step = 3 * renderings.rgb.shape[1]
        response.image.data = img
        return response

if __name__ == '__main__':
    rospy.init_node('MegaPoseServer')
    MegaPoseServer()
    rospy.spin()