#!/usr/bin/env python3

# Import necessary libraries and modules
from visp_megapose.megapose_depend import *  # 3rd-party libraries

# Define paths to the dataset folder and other directories
user = getpass.getuser()
main_folder = "/home/" + user + "/visp-ws/visp/script"              # Path to the main directory of MegaPose
megapose_folder = "/home/" + user + "/catkin_ws/src/visp_megapose"  # Path to the visp_megapose directory
mesh_dir = megapose_folder + "/data/models"                         # Path to the directory containing 3D models

# Ensure the specified paths exist
mesh_dir = Path(mesh_dir).absolute()
assert mesh_dir.exists(), 'Mesh directory does not exist, cannot start server'
assert Path(main_folder).absolute().exists(), 'Main_folder directory does not exist, cannot start server'

# Set the environment variable for the MegaPose dataset
os.environ['MEGAPOSE_DATA_DIR'] = main_folder + "/megapose_server/megapose6d/data"

# Import MegaPose-specific modules
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset       # Object dataset handling
from megapose.datasets.scene_dataset import CameraData, ObjectData                 # Scene dataset handling
from megapose.inference.types import ObservationTensor, PoseEstimatesType          # Inference-related types
from megapose.inference.utils import make_detections_from_object_data              # Utility for detections
from megapose.utils.load_model import NAMED_MODELS, load_named_model               # Model loading utilities
from megapose.lib3d.transform import Transform                                     # 3D transformations
from megapose.utils.conversion import convert_scene_observation_to_panda3d         # Scene conversion utility
from megapose.panda3d_renderer import Panda3dLightData                             # Panda3D light data
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer  # Panda3D scene renderer

# Load camera calibration parameters from a JSON file
data = json.loads(open(megapose_folder + '/params/camera.json').read())

# Import ROS1 service and message definitions
from visp_megapose.srv import Init, Track, Render, InitResponse, TrackResponse, RenderResponse
from sensor_msgs.msg import Image

# Define the MegaPoseServer class
class MegaPoseServer:
    def __init__(self):
        # Define available MegaPose models and their configurations
        megapose_models = {
            'RGB': ('megapose-1.0-RGB', False),
            'RGBD': ('megapose-1.0-RGBD', True),
            'RGB-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis', False),
            'RGBD-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis-icp', True)
        }

        # Load camera data from the JSON file
        camera_data = {
            'K': np.asarray([
                [data['K'][0][0], 0.0, data['K'][0][2]],
                [0.0, data['K'][1][1], data['K'][1][2]],
                [0.0, 0.0, 1.0]
            ]),
            'h': data['h'],
            'w': data['w']
        }

        # Retrieve model parameters from ROS parameter server
        model = rospy.get_param("~megapose_model")
        self.model_name = megapose_models[model][0]
        self.model_use_depth = megapose_models[model][1]

        # Initialize server parameters
        self.mesh_dir = mesh_dir
        self.num_workers = 4
        self.optimize = False
        self.warmup = True
        self.image_batch_size = 256

        # Load object dataset and model
        self.object_dataset: RigidObjectDataset = self.make_object_dataset(self.mesh_dir)
        model_tuple = self._load_model(self.model_name)
        self.model_info = model_tuple[0]
        self.model = model_tuple[1]
        self.model.eval()

        # Set model batch size and camera data
        self.model.bsz_images = self.image_batch_size
        self.camera_data = self._make_camera_data(camera_data)
        h, w = self.camera_data.resolution
        self.h = h
        self.w = w

        # Initialize the Panda3D renderer
        self.renderer = Panda3dSceneRenderer(self.object_dataset)

        # Configure PyTorch settings for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Log camera and model information
        rospy.loginfo('Camera calibration parameter loaded from camera.json file')
        rospy.loginfo('Model selected: %s', model)
        rospy.loginfo('Camera resolution: %s, %s', self.w, self.h)

        # Optimize models if enabled
        if self.optimize:
            rospy.loginfo('Optimizing Pytorch models...')
            class Optimized(nn.Module):
                def __init__(self, m: nn.Module, inp):
                    super().__init__()
                    self.m = m.eval()
                    self.m = fuse(self.m, inplace=False)
                    self.m = torch.jit.trace(self.m, torch.rand(inp).cuda())
                    self.m = torch.jit.freeze(self.m)

                def forward(self, x):
                    return self.m(x).float()

            self.model.coarse_model.backbone = Optimized(self.model.coarse_model.backbone, (1, 9, self.h, self.w))
            self.model.refiner_model.backbone = Optimized(self.model.refiner_model.backbone, (1, 32 if self.model_info['requires_depth'] else 27, self.h, self.w))

        # Perform model warmup if enabled
        if self.warmup:
            labels = self.object_dataset.label_to_objects.keys()
            rospy.loginfo('Models are: %s', list(labels))
            rospy.loginfo('Warming up models...')
            observation = self._make_observation_tensor(np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8),
                                                        np.random.rand(self.h, self.w).astype(np.float32) if self.model_info['requires_depth'] else None).cuda()
            detections = self._make_detections(labels, np.asarray([[0, 0, self.w//2, self.h//2] for _ in range(len(labels))], dtype=np.float32)).cuda()
            self.model.run_inference_pipeline(observation, detections, **self.model_info['inference_parameters'])
        rospy.loginfo('Waiting for request...')

        # Initialize ROS services
        self.srv_init_pose = rospy.Service("init_pose", Init, self.InitPoseCallback)
        self.srv_track_pose = rospy.Service("track_pose", Track, self.TrackPoseCallback)
        self.render_service = rospy.Service('render_object', Render, self.RenderObjectCallback)

    def _load_model(self, model_name):
        # Load the specified model and return its information and instance
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=self.num_workers).cuda()

    def make_object_dataset(self, meshes_dir: Path) -> RigidObjectDataset:
        # Create a dataset of rigid objects from the specified directory
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

    def _make_camera_data(self, camera_data: Dict) -> CameraData:
        '''
        Create a camera representation that is understandable by MegaPose.
        camera_data: A dict containing the keys K, h, w
        K is the 3x3 intrinsics matrix
        h and w are the input image resolution.

        Returns a CameraData object, to be given to MegaPose.
        '''
        c = CameraData()
        c.K = camera_data['K']
        c.resolution = (camera_data['h'], camera_data['w'])
        c.z_near = 0.001
        c.z_far = 100000
        return c

    def _make_detections(self, labels, detections):
        # Create detections from object data
        result = []
        for label, detection in zip(labels, detections):
            o = ObjectData(label)
            o.bbox_modal = detection
            result.append(o)

        return make_detections_from_object_data(result)

    def _make_observation_tensor(self, image: np.ndarray, depth: Optional[np.ndarray] = None) -> ObservationTensor:
        '''
        Create an observation tensor from an image and a potential depth image
        '''
        return ObservationTensor.from_numpy(image, depth, self.camera_data.K)

    def InitPoseCallback(self, request):
        # request: object_name, topleft_i, topleft_j, bottomright_i, bottomright_j, image, camera_info, depth_enable
        # response: pose, scores

        bridge = CvBridge()

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

        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(self.h, self.w, 3)

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

        object_name = [request.object_name]
        detections = [[request.topleft_j, request.topleft_i , request.bottomright_j, request.bottomright_i ]]
        detections = self._make_detections(object_name, detections).cuda()
        observation = self._make_observation_tensor(img, depth).cuda()
        inference_params = self.model_info['inference_parameters'].copy()
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=None
        )
        poses = output.poses.cpu().numpy()
        poses = poses.reshape(len(poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()

        # bounding_boxes = extra_data['scoring']['preds'].tensors['boxes_rend'].cpu().numpy().reshape(-1, 4)
        # bounding_boxes = bounding_boxes.tolist()
        
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
        # request: object_name, init_pose, refiner_iterations, image
        # response: pose, confidence, bounding_box

        bridge = CvBridge()

        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(self.h, self.w, 3)

        if self.model_use_depth:

            depth_uint16 = bridge.imgmsg_to_cv2(request.depth, desired_encoding="passthrough")
            depth = depth_uint16.astype(np.float32) / 1000.0

        else:

            depth = None

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
        inference_params['n_refiner_iterations'] = 1
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
        response.bb = bounding_boxes[0]
        return response
    
    def RenderObjectCallback(self, request):
        # request: object_name, pose
        # response: image

        labels = [request.object_name]
        poses = np.eye(4)
        poses[0:3, 3] = [request.pose.translation.x, request.pose.translation.y, request.pose.translation.z]
        poses[0:3, 0:3] = transforms3d.quaternions.quat2mat([request.pose.rotation.w, request.pose.rotation.x, request.pose.rotation.y, request.pose.rotation.z])
        poses = poses.reshape(1, 4, 4)

        camera_data = CameraData()
        camera_data.K = self.camera_data.K
        camera_data.resolution = self.camera_data.resolution
        camera_data.TWC = Transform(np.eye(4))

        object_datas = []
        for label, pose in zip(labels, poses):
            object_datas.append(ObjectData(label=label, TWO=Transform(pose)))
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
             light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        img = renderings.rgb
        img = np.uint8(img).reshape(1, -1).tolist()[0]

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