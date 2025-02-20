#!/usr/bin/env python3
# ROS
import sys
import os
import json
import rospy
import transforms3d
from visp_megapose.srv import Init, Track, InitResponse, TrackResponse

sys.path.insert(1,'/root/visp-ws/visp/script')
import megapose_server
megapose_server_install_dir = os.path.dirname(megapose_server.__file__)
variables_file = os.path.join(megapose_server_install_dir, 'megapose_variables_final.json')
with open(variables_file, 'r') as f:
    json_vars = json.load(f)
    # print('Loaded megapose variables', json_vars)
    os.environ['MEGAPOSE_DIR'] = json_vars['megapose_dir']
    os.environ['MEGAPOSE_DATA_DIR'] = json_vars['megapose_data_dir']

if 'HOME' not in os.environ: # Home is always required by megapose but is not always set
    if os.name == 'nt':
      if 'HOMEPATH' in os.environ:
        os.environ['HOME'] = os.environ['HOMEPATH']
      elif 'HOMEDIR' in os.environ:
        os.environ['HOME'] = os.environ['HOMEDIR']
      else:
        os.environ['HOME'] = '.'
    else:
      os.environ['HOME'] = '.'

# 3rd party
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.experimental.optimization import fuse
import time
# MegaPose
sys.path.insert(1,'/root/visp-ws/visp/script/megapose_server/megapose6d/src/megapose/datasets')
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model

# Megapose server
from megapose_server.network_utils import *

# data = np.loadtxt('/home/ws/src/visp_stuffs/visp_megapose/params/camera.txt').astype(np.float64) #Load resized camera calibration parameters
# print('Camera calibration parameter loaded from camera.txt file')

megapose_models = {
        'RGB': ('megapose-1.0-RGB', False)
    }
camera_data = {
        'K': np.asarray([
            [201.30212783813477, 0.0,196.57247543334964],
            [0.0, 250.24478236494178,  143.14741759891876],
            [0.0, 0.0, 1.0]
        ]),
        'h': 320,
        'w': 480
}

def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "m"
    object_dirs = meshes_dir.iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}:
                assert not mesh_path, f"there are multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find the mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

# Server ROS1

class MegaPoseServer:
    def __init__(self):
        # Parameters:
        #     mesh_dir (string): The path to the directory containing the 3D models
        #                        Each model is stored in a subfolder, where the subfolder name gives the name of the object
        #                        A 3D model can be in .obj or .ply format. Units are assumed to be in meters
        #     optimize (bool): Whether to optimize the deep network models for faster inference
        #                      Still very experimental, and may result in a loss of accuracy with no performance gain!
        #     warmup (bool): Whether to perform model warmup in order to avoid a slow first inference pass
        #     num_workers (int): Number of workers for rendering
        #     image_batch_size (int): Image batch dimension
        
        model_name = megapose_models['RGB'][0]
        self.model_name = model_name

        mesh_dir = rospy.get_param("mesh_dir")
        mesh_dir = Path(mesh_dir).absolute()
        assert mesh_dir.exists(), 'Mesh directory does not exist, cannot start server' 
        self.mesh_dir = mesh_dir

        num_workers = rospy.get_param("num_workers")
        self.num_workers = num_workers

        optimize = rospy.get_param("optimize")
        self.optimize = optimize

        warmup = rospy.get_param("warmup")
        self.warmup = warmup

        image_batch_size = rospy.get_param("image_batch_size")
        self.image_batch_size = image_batch_size

        
        self.object_dataset: RigidObjectDataset = make_object_dataset(mesh_dir)
        model_tuple = self._load_model(model_name)
        self.model_info = model_tuple[0]
        self.model = model_tuple[1]
        self.model.eval()

        self.model.bsz_images = image_batch_size
        self.camera_data = self._make_camera_data(camera_data)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if self.optimize:
            print('Optimizing Pytorch models...')
            class Optimized(nn.Module):
                def __init__(self, m: nn.Module, inp):
                    super().__init__()
                    self.m = m.eval()
                    self.m = fuse(self.m, inplace=False)
                    self.m = torch.jit.trace(self.m, torch.rand(inp).cuda())
                    self.m = torch.jit.freeze(self.m)

                def forward(self, x):
                    return self.m(x).float()

            h, w = self.camera_data.resolution
            self.model.coarse_model.backbone = Optimized(self.model.coarse_model.backbone, (1, 9, h, w))
            self.model.refiner_model.backbone = Optimized(self.model.refiner_model.backbone, (1, 32 if self.model_info['requires_depth'] else 27, h, w))

        if self.warmup:
            print('Warming up models...')
            h, w = self.camera_data.resolution
            labels = self.object_dataset.label_to_objects.keys()
            observation = self._make_observation_tensor(np.random.randint(0, 255, (h, w, 4), dtype=np.uint8),
                                                        np.random.rand(h, w).astype(np.float32) if self.model_info['requires_depth'] else None).cuda()
            detections = self._make_detections(labels, np.asarray([[0, 0, w//2, h//2] for _ in range(len(labels))], dtype=np.float32)).cuda()
            self.model.run_inference_pipeline(observation, detections, **self.model_info['inference_parameters'])
        print('Waiting for request...')

        self.srv_init_pose = rospy.Service("init_pose",Init, self.InitPoseCallback)
        self.srv_track_pose = rospy.Service("track_pose",Track, self.TrackPoseCallback)

    def _load_model(self, model_name):
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=self.num_workers).cuda()
    
    def _make_camera_data(self, camera_data: Dict) -> CameraData:
        '''
        Create a camera representation that is understandable by megapose.
        camera_data: A dict containing the keys K, h, w
        K is the 3x3 intrinsics matrix
        h and w are the input image resolution.

        Returns a CameraData object, to be given to megapose.
        '''
        c = CameraData()
        c.K = camera_data['K']
        c.resolution = (camera_data['h'], camera_data['w'])
        # print(c)
        c.z_near = 0.001
        c.z_far = 100000
        return c
    
    def InitPoseCallback(self, request):
        # request: object_name, topleft_i, topleft_j, bottomright_i, bottomright_j, image, camera_info
        # response: pose, scores, bounding_boxes
        depth = None
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

        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(request.camera_info.height, request.camera_info.width, 3)
        object_name = [request.object_name]
        detections = [[request.topleft_j,request.topleft_i , request.bottomright_j, request.bottomright_i ]]
        detections = self._make_detections(object_name, detections).cuda()
        observation = self._make_observation_tensor(img, depth).cuda()
        inference_params = self.model_info['inference_parameters'].copy()
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=None
        )
        poses = output.poses.cpu().numpy()
        poses = poses.reshape(len(poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()

        bounding_boxes = extra_data['scoring']['preds'].tensors['boxes_rend'].cpu().numpy().reshape(-1, 4)
        bounding_boxes = bounding_boxes.tolist()
        
        response= InitResponse()
        response.pose.translation.x = float(poses[0][0, 3])
        response.pose.translation.y = float(poses[0][1, 3])
        response.pose.translation.z = float(poses[0][2, 3])
        rotation = poses[0][0:3, 0:3]
        rotation = transforms3d.quaternions.mat2quat(rotation)
        response.pose.rotation.x = rotation[1]
        response.pose.rotation.y = rotation[2]
        response.pose.rotation.z = rotation[3]
        response.pose.rotation.w = rotation[0]
        response.confidence = float(confidence[0])
        return response 
    
    def TrackPoseCallback(self, request):
        # request: object_name, init_pose, refiner_iterations, image, camera_info
        # response: pose confidence
        t = time.time()
        depth = None
        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(request.camera_info.height, request.camera_info.width, 3)
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

        detections = None
        observation = self._make_observation_tensor(img, depth).cuda()
        inference_params = self.model_info['inference_parameters'].copy()
        inference_params['n_refiner_iterations'] = request.refiner_iterations
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=coarse_estimates
        )
        poses = output.poses.cpu().numpy()
        poses = poses.reshape(len(poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()
        bounding_boxes = extra_data['scoring']['preds'].tensors['boxes_rend'].cpu().numpy().reshape(-1, 4)
        bounding_boxes = bounding_boxes.tolist()

        response = TrackResponse()
        response.pose.translation.x = float(poses[0][0, 3])
        response.pose.translation.y = float(poses[0][1, 3])
        response.pose.translation.z = float(poses[0][2, 3])
        rotation = poses[0][0:3, 0:3]
        rotation = transforms3d.quaternions.mat2quat(rotation)
        response.pose.rotation.x = rotation[1]
        response.pose.rotation.y = rotation[2]
        response.pose.rotation.z = rotation[3]
        response.pose.rotation.w = rotation[0]
        response.confidence = float(confidence[0])

        return response
        
    def _make_detections(self, labels, detections):
        result = []
        for label, detection in zip(labels, detections):
            o = ObjectData(label)
            o.bbox_modal = detection
            result.append(o)

        return make_detections_from_object_data(result)
    
    def _make_observation_tensor(self, image: np.ndarray, depth: Optional[np.ndarray]=None) -> ObservationTensor:
        '''
        Create an observation tensor from an image and a potential depth image
        '''
        return ObservationTensor.from_numpy(image, depth, self.camera_data.K)


if __name__ == '__main__':
    rospy.init_node('MegaPoseServer')
    MegaPoseServer()
    rospy.spin()