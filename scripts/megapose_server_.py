#!/usr/bin/env python3

from visp_megapose.megapose_depend import *               # 3rdparty libraries

# Path to the dataset folder, change to your own path
user = getpass.getuser()
main_folder = "/home/" + user + "/visp-ws/visp/script"               # The path to the directory of MegaPose
megapose_folder = "/home/" + user + "/catkin_ws/src/visp_megapose"   # The path to the directory of visp_megapose
mesh_dir = megapose_folder + "/data/models"                          # The path to the directory containing the 3D models    

# Check for path existence
mesh_dir = Path(mesh_dir).absolute()
assert mesh_dir.exists(), 'Mesh directory does not exist, cannot start server'
assert Path(main_folder).absolute().exists(), 'Main_folder directory does not exist, cannot start server' 

# Path for the MegaPose dataset
os.environ['MEGAPOSE_DATA_DIR'] = main_folder + "/megapose_server/megapose6d/data"

# MegaPose
# Import necessary modules from the MegaPose library
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset          # For handling 3D object datasets
from megapose.datasets.scene_dataset import CameraData, ObjectData                    # For representing camera and object data
from megapose.inference.types import ObservationTensor, PoseEstimatesType             # For handling observations and pose estimates
from megapose.inference.utils import make_detections_from_object_data                 # Utility for creating detections from object data
from megapose.utils.load_model import NAMED_MODELS, load_named_model                  # For loading pre-trained MegaPose models
from megapose.lib3d.transform import Transform                                        # For handling 3D transformations
from megapose.utils.conversion import convert_scene_observation_to_panda3d            # Conversion utility for Panda3D rendering
from megapose.panda3d_renderer import Panda3dLightData                                # For defining light data in Panda3D
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer     # For rendering 3D scenes using Panda3D

# Load camera information, can be generated with ros_imresize node
data = json.loads(open(megapose_folder + '/params/camera.json').read())

# Server ROS1
from visp_megapose.srv import Init, Track, Render, InitResponse, TrackResponse, RenderResponse

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
        #     megaposse_model (string): The name of the model to use. Options are:
        #                              'RGB': RGB only model
        #                              'RGBD': RGBD model
        #                              'RGB-multi-hypothesis': RGB only model with multi-hypothesis
        #                              'RGBD-multi-hypothesis': RGBD model with multi-hypothesis

        # Define available MegaPose models and their configurations
        megapose_models = {
            'RGB': ('megapose-1.0-RGB', False),
            'RGBD': ('megapose-1.0-RGBD', True),
            'RGB-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis', False),
            'RGBD-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis-icp', True)
            }

        # Load camera intrinsic parameters from the provided data
        print(data['K'][0])
        camera_data = {
                 'K': np.asarray([
                     [data['K'][0], 0.0, data['K'][2]],
                     [0.0, data['K'][4], data['K'][5]],
                     [0.0, 0.0, 1.0]
                 ]),
                 'h': data['h'],
                 'w': data['w']
         }

        # Retrieve the selected MegaPose model from ROS parameters
        model = rospy.get_param("~megapose_model")
        self.model_name = megapose_models[model][0]
        self.model_use_depth = megapose_models[model][1]

        # Initialize various parameters for the MegaPose server
        self.mesh_dir = mesh_dir
        self.num_workers = 4
        self.optimize = False
        self.warmup = True
        self.image_batch_size = 256

        # Create the object dataset from the 3D model directory
        self.object_dataset: RigidObjectDataset = self.make_object_dataset(self.mesh_dir)

        # Load the selected MegaPose model
        model_tuple = self._load_model(self.model_name)
        self.model_info = model_tuple[0]
        self.model = model_tuple[1]
        self.model.eval()  # Set the model to evaluation mode

        # Set batch size for image processing
        self.model.bsz_images = self.image_batch_size

        # Create camera data representation for MegaPose
        self.camera_data = self._make_camera_data(camera_data)
        h, w = self.camera_data.resolution
        self.h = h
        self.w = w

        # Initialize the Panda3D renderer for rendering 3D scenes
        self.renderer = Panda3dSceneRenderer(self.object_dataset)

        # Enable CUDA optimizations for PyTorch
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Log camera calibration and model information
        rospy.loginfo('Camera calibration parameter loaded from camera.json file')
        rospy.loginfo('Model selected: %s', model)
        rospy.loginfo('Camera resolution: %s, %s', self.w, self.h)
        
        # Optimize the model for faster inference if enabled
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

        # Perform model warmup to avoid slow initial inference if enabled
        if self.warmup:
            labels = self.object_dataset.label_to_objects.keys()
            rospy.loginfo('Models are: %s', list(labels))
            rospy.loginfo('Warming up models...')
            observation = self._make_observation_tensor(np.random.randint(0, 255, (self.h, self.w, 3), dtype=np.uint8),
                                np.random.rand(self.h, self.w).astype(np.float32) if self.model_info['requires_depth'] else None).cuda()
            detections = self._make_detections(labels, np.asarray([[0, 0, self.w//2, self.h//2] for _ in range(len(labels))], dtype=np.float32)).cuda()
            self.model.run_inference_pipeline(observation, detections, **self.model_info['inference_parameters'])

        # Log that the server is ready and waiting for requests
        rospy.loginfo('Waiting for request...')

        # Initialize ROS services for pose initialization, tracking, and rendering
        self.srv_init_pose = rospy.Service("init_pose", Init, self.InitPoseCallback)
        self.srv_track_pose = rospy.Service("track_pose", Track, self.TrackPoseCallback)
        self.render_service = rospy.Service('render_object', Render, self.RenderObjectCallback)

    # Helper function to load a MegaPose model
    def _load_model(self, model_name):
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=self.num_workers).cuda()

    # Helper function to create a dataset of rigid objects from 3D model files
    def make_object_dataset(self, meshes_dir: Path) -> RigidObjectDataset:
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
    
    # Helper function to create camera data for MegaPose
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
        c.z_near = 0.001
        c.z_far = 100000
        return c
        
    # Helper function to create detections from object labels and bounding boxes
    def _make_detections(self, labels, detections):
        result = []
        for label, detection in zip(labels, detections):
            o = ObjectData(label)
            o.bbox_modal = detection
            result.append(o)

        return make_detections_from_object_data(result)
    
    # Helper function to create an observation tensor from an image and optional depth data
    def _make_observation_tensor(self, image: np.ndarray, depth: Optional[np.ndarray]=None) -> ObservationTensor:
        '''
        Create an observation tensor from an image and a potential depth image
        '''
        return ObservationTensor.from_numpy(image, depth, self.camera_data.K)

    def InitPoseCallback(self, request):
        # request: object_name, topleft_i, topleft_j, bottomright_i, bottomright_j, image, camera_info, depth_enable
        # response: pose, scores

        # Initialize a CvBridge instance for converting ROS image messages to OpenCV format
        bridge = CvBridge()

        # Extract camera intrinsic parameters from the request and update the server's camera data
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

        # Convert the input image from the request to a NumPy array
        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(self.h, self.w, 3)

        # Handle depth data if depth is enabled in the request
        if request.depth_enable:
            # Check if the selected model supports depth
            if not self.model_info['requires_depth']:
                rospy.loginfo('Trying to use depth with a model that cannot handle it')
                return
            else:
            # Convert the depth image from the request to a NumPy array and scale it to meters
                depth_uint16 = bridge.imgmsg_to_cv2(request.depth, desired_encoding="passthrough")
                depth = depth_uint16.astype(np.float32) / 1000.0
        else:
            # Handle cases where depth is not enabled
            if not self.model_info['requires_depth']:
                depth = None
            else:
                rospy.loginfo('Trying to use a model that requires depth without providing it')
                return

        # Extract the object name and bounding box coordinates from the request
        object_name = [request.object_name]
        detections = [[request.topleft_j, request.topleft_i, request.bottomright_j, request.bottomright_i]]
        detections = self._make_detections(object_name, detections).cuda()

        # Create an observation tensor from the image and depth data
        observation = self._make_observation_tensor(img, depth).cuda()

        # Copy inference parameters and run the inference pipeline
        inference_params = self.model_info['inference_parameters'].copy()
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=None
        )

        # Extract pose estimates and confidence scores from the inference output
        poses = output.poses.cpu().numpy()
        poses = poses.reshape(len(poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()

        # Prepare the response with the estimated pose and confidence score
        response = InitResponse()
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

        # Initialize a CvBridge instance for converting ROS image messages to OpenCV format
        bridge = CvBridge()

        # Convert the input image from the request to a NumPy array
        img = np.array(np.frombuffer(request.image.data, dtype=np.uint8)).reshape(self.h, self.w, 3)

        # Handle depth data if the model requires it
        if self.model_use_depth:
            # Convert the depth image from the request to a NumPy array and scale it to meters
            depth_uint16 = bridge.imgmsg_to_cv2(request.depth, desired_encoding="passthrough")
            depth = depth_uint16.astype(np.float32) / 1000.0
        else:
            # Set depth to None if the model does not use depth
            depth = None

        # Extract the object name from the request
        object_name = [request.object_name]

        # Convert the initial pose from the request to a 4x4 transformation matrix
        cTos_np = np.eye(4)
        cTos_np[0:3, 3] = [request.init_pose.translation.x, request.init_pose.translation.y, request.init_pose.translation.z]
        cTos_np[0:3, 0:3] = transforms3d.quaternions.quat2mat([request.init_pose.rotation.w, request.init_pose.rotation.x, request.init_pose.rotation.y, request.init_pose.rotation.z])
        cTos_np = cTos_np.reshape(1, 4, 4)

        # Convert the transformation matrix to a PyTorch tensor and move it to the GPU
        tensor = torch.from_numpy(cTos_np).float().cuda()

        # Create a DataFrame to store pose information
        infos = pd.DataFrame.from_dict({
            'label': object_name,
            'batch_im_id': [0 for _ in range(len(cTos_np))],
            'instance_id': [i for i in range(len(cTos_np))]
        })

        # Create coarse pose estimates using the provided initial pose
        coarse_estimates = PoseEstimatesType(infos, poses=tensor)

        # Set detections to None as they are not used in this callback
        detections = None

        # Create an observation tensor from the image and depth data
        observation = self._make_observation_tensor(img, depth).cuda()

        # Copy inference parameters and set the number of refiner iterations
        inference_params = self.model_info['inference_parameters'].copy()
        inference_params['n_refiner_iterations'] = 1

        # Run the inference pipeline with the observation and coarse estimates
        output, extra_data = self.model.run_inference_pipeline(
            observation, detections=detections, **inference_params, coarse_estimates=coarse_estimates
        )

        # Extract pose estimates and confidence scores from the inference output
        poses = output.poses.cpu().numpy()
        poses = poses.reshape(len(poses), 4, 4)
        confidence = output.infos['pose_score'].to_numpy()

        # Extract bounding boxes from the extra data
        bounding_boxes = extra_data['scoring']['preds'].tensors['boxes_rend'].cpu().numpy().reshape(-1, 4)
        bounding_boxes = bounding_boxes.tolist()

        # Prepare the response with the estimated pose, confidence score, and bounding box
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

        # Extract the object name and pose from the request
        labels = [request.object_name]
        poses = np.eye(4)
        poses[0:3, 3] = [request.pose.translation.x, request.pose.translation.y, request.pose.translation.z]
        poses[0:3, 0:3] = transforms3d.quaternions.quat2mat([request.pose.rotation.w, request.pose.rotation.x, request.pose.rotation.y, request.pose.rotation.z])
        poses = poses.reshape(1, 4, 4)

        # Create a CameraData object with the current camera parameters
        camera_data = CameraData()
        camera_data.K = self.camera_data.K
        camera_data.resolution = self.camera_data.resolution
        camera_data.TWC = Transform(np.eye(4))  # Set the camera pose to identity

        # Create ObjectData instances for each object to be rendered
        object_datas = []
        for label, pose in zip(labels, poses):
            object_datas.append(ObjectData(label=label, TWO=Transform(pose)))

        # Convert the scene observation to a format compatible with Panda3D
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

        # Define lighting for the scene (ambient light in this case)
        light_datas = [
            Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),  # White ambient light
            ),
        ]

        # Render the scene using the Panda3D renderer
        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,          # Do not render depth
            render_binary_mask=False,    # Do not render binary mask
            render_normals=False,        # Do not render normals
            copy_arrays=True,            # Copy the rendered arrays
        )[0]

        # Extract the rendered RGB image and convert it to a list of uint8 values
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