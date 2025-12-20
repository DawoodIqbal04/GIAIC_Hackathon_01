---
id: chapter-10-isaac-sim-generation
title: "Chapter 10: NVIDIA Isaac Sim: Photorealistic Simulation and Synthetic Data Generation"
sidebar_label: "Chapter 10: NVIDIA Isaac Sim for Data Generation"
description: "Using NVIDIA Isaac Sim for photorealistic simulation and synthetic data generation for humanoid robotics perception systems"
keywords: [nvidia-isaac-sim, simulation, synthetic-data, photorealistic, perception, robotics, training]
tags: [isaac-sim, synthetic-data, simulation]
authors: [book-authors]
difficulty: advanced
estimated_time: "120 minutes"
module: 3
chapter: 10
prerequisites: [python-ai-basics, ros2-foundations, perception-basics, computer-vision, gpu-programming]
learning_objectives:
  - Understand NVIDIA Isaac Sim architecture and capabilities
  - Create photorealistic simulation environments for humanoid robots
  - Generate synthetic datasets for perception system training
  - Implement domain randomization techniques for robust perception
  - Integrate Isaac Sim with ROS 2 for perception pipeline testing
related:
  - next: chapter-11-isaac-ros-navigation
  - previous: chapter-9-advanced-perception
  - see_also: [chapter-9-advanced-perception, chapter-11-isaac-ros-navigation, ../module-2-digital-twins/chapter-7-unity-interaction]
---

# Chapter 10: NVIDIA Isaac Sim: Photorealistic Simulation and Synthetic Data Generation

## Learning Objectives

After completing this chapter, you will be able to:
- Set up and configure NVIDIA Isaac Sim for humanoid robotics simulation
- Create photorealistic environments for perception training
- Generate synthetic datasets with accurate annotations
- Apply domain randomization to improve model robustness
- Integrate Isaac Sim with ROS 2 for perception system validation

## Introduction

NVIDIA Isaac Sim is a high-fidelity simulation environment built on the Omniverse platform, specifically designed for robotics development. It provides photorealistic rendering capabilities, accurate physics simulation, and GPU-accelerated sensor simulation, making it ideal for generating synthetic data to train perception systems for humanoid robots.

Isaac Sim enables the creation of diverse, annotated datasets that would be difficult or expensive to collect in the real world. This synthetic data can be used to train perception models that are more robust and generalizable, particularly important for humanoid robots operating in varied human environments.

## NVIDIA Isaac Sim Architecture

### Core Components

Isaac Sim consists of several key components:

1. **Omniverse Platform**: Provides the underlying real-time 3D simulation and rendering capabilities
2. **PhysX Physics Engine**: Ensures accurate physics simulation for realistic interactions
3. **RTX Ray Tracing**: Enables photorealistic rendering for synthetic data generation
4. **Sensor Simulation**: Provides GPU-accelerated simulation of cameras, LiDAR, IMU, and other sensors
5. **Robot Simulation**: Supports articulated robot models with accurate kinematics and dynamics
6. **ROS 2 Bridge**: Enables seamless integration with ROS 2-based robotics applications

### Isaac Sim vs Traditional Simulation

Compared to traditional robotics simulators like Gazebo:

- **Visual Fidelity**: RTX ray tracing for photorealistic rendering
- **Sensor Accuracy**: Physically-based sensor simulation
- **Domain Randomization**: Advanced tools for synthetic data variation
- **Scalability**: GPU-accelerated simulation for large datasets
- **Integration**: Native Omniverse connectivity for asset sharing

## Setting Up Isaac Sim

### Installation and Prerequisites

To use Isaac Sim effectively, you'll need:

- NVIDIA GPU with RTX capabilities (recommended)
- Isaac Sim installed (part of Isaac ROS Developer Kit)
- Omniverse Create or Isaac Sim application
- CUDA-compatible GPU with sufficient VRAM

### Basic Isaac Sim Python API

```python
import omni
import carb
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np

class IsaacSimEnvironment:
    def __init__(self):
        self.stage = None
        self.world = None
        self.objects = {}
        
    def initialize(self):
        """Initialize Isaac Sim environment."""
        # Create a new USD stage
        self.stage = omni.usd.get_context().get_stage()
        
        # Create world prim
        self.world = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World"))
        
        print("Isaac Sim environment initialized")
    
    def create_object(self, name, prim_type, position=(0, 0, 0)):
        """Create an object in the simulation."""
        path = Sdf.Path(f"/World/{name}")
        
        if prim_type == "Sphere":
            prim = UsdGeom.Sphere.Define(self.stage, path)
            prim.CreateRadiusAttr(0.5)
        elif prim_type == "Cube":
            prim = UsdGeom.Cube.Define(self.stage, path)
            prim.CreateSizeAttr(1.0)
        elif prim_type == "Cylinder":
            prim = UsdGeom.Cylinder.Define(self.stage, path)
            prim.CreateRadiusAttr(0.3)
            prim.CreateHeightAttr(1.0)
        else:
            raise ValueError(f"Unsupported prim type: {prim_type}")
        
        # Set position
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        
        self.objects[name] = prim
        return prim
    
    def set_object_material(self, name, color=(1.0, 1.0, 1.0, 1.0)):
        """Set material properties for an object."""
        # Implementation to set material properties
        pass
    
    def setup_lighting(self):
        """Set up lighting in the scene."""
        # Create dome light
        dome_light_path = Sdf.Path("/World/DomeLight")
        dome_light = UsdGeom.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr(1000)
        
        # Create additional lights as needed
        pass
```

## Creating Photorealistic Environments

### Environment Design Principles

Creating effective environments for synthetic data generation requires attention to:

1. **Visual Complexity**: Include diverse textures, lighting conditions, and object arrangements
2. **Realistic Physics**: Accurate material properties and physical interactions
3. **Variety**: Multiple scenes representing different operational environments
4. **Annotation Readiness**: Clear segmentation and labeling capabilities

### Sample Environment Creation

```python
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

class HumanoidEnvironment(IsaacSimEnvironment):
    def __init__(self):
        super().__init__()
        self.world = World(stage_units_in_meters=1.0)
        
    def create_household_environment(self):
        """Create a household environment for humanoid robot training."""
        # Create ground plane
        self.create_ground_plane()
        
        # Add furniture
        self.add_furniture()
        
        # Add household objects
        self.add_household_objects()
        
        # Set up lighting
        self.setup_household_lighting()
        
        # Add humanoid robot
        self.add_humanoid_robot()
        
    def create_ground_plane(self):
        """Create a ground plane for the environment."""
        from omni.isaac.core.objects import GroundPlane
        self.world.scene.add(
            GroundPlane(
                prim_path="/World/defaultGroundPlane",
                name="default_ground_plane",
                size=1000.0,
                color=np.array([0.2, 0.2, 0.2])
            )
        )
    
    def add_furniture(self):
        """Add furniture to the environment."""
        # Add a table
        from omni.isaac.core.objects import VisualCuboid
        self.world.scene.add(
            VisualCuboid(
                prim_path="/World/table",
                name="table",
                position=np.array([1.0, 0.0, 0.5]),
                size=0.8,
                color=np.array([0.5, 0.3, 0.1])
            )
        )
        
        # Add a chair
        self.world.scene.add(
            VisualCuboid(
                prim_path="/World/chair",
                name="chair",
                position=np.array([1.5, 1.0, 0.3]),
                size=0.6,
                color=np.array([0.3, 0.3, 0.5])
            )
        )
    
    def add_household_objects(self):
        """Add household objects for perception training."""
        # Add various objects that a humanoid robot might encounter
        objects = [
            ("red_cup", [0.5, 0.5, 0.6], [1.0, 0.0, 0.0]),
            ("blue_bowl", [-0.5, -0.5, 0.6], [0.0, 0.0, 1.0]),
            ("green_box", [0.0, 1.0, 0.5], [0.0, 1.0, 0.0]),
        ]
        
        for name, pos, color in objects:
            from omni.isaac.core.objects import VisualCuboid
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/{name}",
                    name=name,
                    position=np.array(pos),
                    size=0.2,
                    color=np.array(color)
                )
            )
    
    def setup_household_lighting(self):
        """Set up realistic household lighting."""
        # Add dome light for ambient lighting
        from omni.isaac.core.utils.prims import create_prim
        create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            position=np.array([0, 0, 0]),
            attributes={"color": np.array([1.0, 1.0, 1.0])}
        )
        
        # Add a few point lights for more realistic lighting
        create_prim(
            prim_path="/World/PointLight1",
            prim_type="SphereLight",
            position=np.array([2.0, 2.0, 3.0]),
            attributes={"inputs:diffuse": 1, "inputs:specular": 1}
        )
    
    def add_humanoid_robot(self):
        """Add a humanoid robot to the environment."""
        # For this example, we'll add a simple articulated robot
        # In practice, you would load a URDF or USD model of your humanoid
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import add_reference_to_stage
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets path")
            return
        
        # Add a simple robot (in practice, use your humanoid model)
        robot_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")
```

## Synthetic Data Generation Pipeline

### Basic Data Generation

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import cv2
import json
import os
from PIL import Image

class SyntheticDataGenerator:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = output_dir
        self.data_counter = 0
        self.world = World(stage_units_in_meters=1.0)
        
        # Create output directories
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/seg", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        
        # Initialize synthetic data helper
        self.sd_helper = SyntheticDataHelper()
        
    def generate_single_sample(self, environment_config):
        """Generate a single synthetic data sample."""
        # Set up environment based on config
        self.setup_environment(environment_config)
        
        # Position camera
        self.position_camera_randomly()
        
        # Capture data
        rgb_image = self.capture_rgb_image()
        depth_image = self.capture_depth_image()
        segmentation = self.capture_segmentation()
        
        # Create annotations
        annotations = self.create_annotations()
        
        # Save data
        sample_id = f"sample_{self.data_counter:06d}"
        self.save_sample(sample_id, rgb_image, depth_image, segmentation, annotations)
        
        self.data_counter += 1
        
    def setup_environment(self, config):
        """Set up environment based on configuration."""
        # Clear previous environment
        self.world.clear()
        
        # Add ground plane
        from omni.isaac.core.objects import GroundPlane
        self.world.scene.add(GroundPlane(prim_path="/World/defaultGroundPlane", name="ground_plane"))
        
        # Add objects based on config
        for obj_config in config.get("objects", []):
            self.add_object(obj_config)
        
        # Set lighting
        self.setup_lighting(config.get("lighting", {}))
    
    def add_object(self, obj_config):
        """Add an object to the environment."""
        from omni.isaac.core.objects import VisualCuboid, VisualSphere, VisualCylinder
        import random
        
        obj_type = obj_config.get("type", "cube")
        position = obj_config.get("position", [random.uniform(-2, 2), random.uniform(-2, 2), 0.5])
        color = obj_config.get("color", [random.random(), random.random(), random.random()])
        
        if obj_type == "cube":
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/obj_{len(self.world.scene.objects)}",
                    name=f"obj_{len(self.world.scene.objects)}",
                    position=np.array(position),
                    size=0.3,
                    color=np.array(color)
                )
            )
        elif obj_type == "sphere":
            self.world.scene.add(
                VisualSphere(
                    prim_path=f"/World/obj_{len(self.world.scene.objects)}",
                    name=f"obj_{len(self.world.scene.objects)}",
                    position=np.array(position),
                    radius=0.2,
                    color=np.array(color)
                )
            )
    
    def position_camera_randomly(self):
        """Position camera randomly in the environment."""
        import random
        
        # Random camera position around the scene
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        z = random.uniform(1, 3)  # Height above ground
        
        # Look at center of scene
        target = [0, 0, 0.5]
        
        set_camera_view(eye=np.array([x, y, z]), target=np.array(target))
    
    def capture_rgb_image(self):
        """Capture RGB image from the current camera view."""
        # Implementation to capture RGB image
        # This would use Isaac Sim's rendering capabilities
        pass
    
    def capture_depth_image(self):
        """Capture depth image."""
        # Implementation to capture depth image
        # This would use Isaac Sim's depth rendering
        pass
    
    def capture_segmentation(self):
        """Capture semantic segmentation."""
        # Implementation to capture segmentation
        # This would use Isaac Sim's segmentation rendering
        pass
    
    def create_annotations(self):
        """Create annotations for the captured data."""
        # Implementation to create bounding box and segmentation annotations
        annotations = {
            "objects": [],
            "camera_pose": {},
            "lighting_conditions": {}
        }
        return annotations
    
    def save_sample(self, sample_id, rgb, depth, segmentation, annotations):
        """Save a complete data sample."""
        # Save RGB image
        rgb_path = f"{self.output_dir}/rgb/{sample_id}.png"
        Image.fromarray(rgb).save(rgb_path)
        
        # Save depth image
        depth_path = f"{self.output_dir}/depth/{sample_id}.png"
        Image.fromarray(depth).save(depth_path)
        
        # Save segmentation
        seg_path = f"{self.output_dir}/seg/{sample_id}.png"
        Image.fromarray(segmentation).save(seg_path)
        
        # Save annotations
        annot_path = f"{self.output_dir}/annotations/{sample_id}.json"
        with open(annot_path, 'w') as f:
            json.dump(annotations, f)
```

### Advanced Data Generation with Domain Randomization

Domain randomization is crucial for creating robust perception models:

```python
import random
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf

class DomainRandomizationGenerator(SyntheticDataGenerator):
    def __init__(self, output_dir="synthetic_data_dr"):
        super().__init__(output_dir)
        
        # Domain randomization parameters
        self.lighting_params = {
            'intensity_range': (100, 1000),
            'color_temperature_range': (3000, 8000),  # Kelvin
            'position_variance': 2.0
        }
        
        self.material_params = {
            'albedo_range': (0.1, 1.0),
            'roughness_range': (0.0, 1.0),
            'metallic_range': (0.0, 1.0)
        }
        
        self.camera_params = {
            'fov_range': (30, 90),  # degrees
            'position_variance': 3.0,
            'orientation_variance': 0.5  # radians
        }
    
    def setup_environment(self, config):
        """Set up environment with domain randomization."""
        # Call parent method to set up basic environment
        super().setup_environment(config)
        
        # Apply domain randomization
        self.randomize_lighting()
        self.randomize_materials()
        self.randomize_camera()
    
    def randomize_lighting(self):
        """Apply randomization to lighting conditions."""
        # Get all lights in the scene
        stage = omni.usd.get_context().get_stage()
        
        # Randomize dome light
        dome_light_path = "/World/DomeLight"
        dome_light_prim = stage.GetPrimAtPath(dome_light_path)
        if dome_light_prim.IsValid():
            # Randomize intensity
            intensity = random.uniform(*self.lighting_params['intensity_range'])
            dome_light_prim.GetAttribute("inputs:intensity").Set(intensity)
            
            # Randomize color temperature (simplified)
            color_temp = random.uniform(*self.lighting_params['color_temperature_range'])
            # Convert to approximate RGB - this is a simplification
            rgb_color = self.color_temperature_to_rgb(color_temp)
            dome_light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*rgb_color))
    
    def color_temperature_to_rgb(self, color_temp):
        """Convert color temperature to RGB (simplified approximation)."""
        temp = color_temp / 100
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        
        # Blue calculation
        if temp >= 66:
            blue = 255
        elif temp <= 19:
            blue = 0
        else:
            blue = temp - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
        
        # Normalize to 0-1 range
        return [max(0, min(255, x)) / 255.0 for x in [red, green, blue]]
    
    def randomize_materials(self):
        """Apply randomization to material properties."""
        # Get all objects in the scene
        stage = omni.usd.get_context().get_stage()
        
        # For each object, randomize material properties
        for prim in stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Sphere):
                # Randomize albedo
                albedo = [random.uniform(*self.material_params['albedo_range']) for _ in range(3)]
                
                # Randomize roughness
                roughness = random.uniform(*self.material_params['roughness_range'])
                
                # Randomize metallic
                metallic = random.uniform(*self.material_params['metallic_range'])
                
                # Apply material properties (this is a simplified example)
                self.apply_material_properties(prim, albedo, roughness, metallic)
    
    def apply_material_properties(self, prim, albedo, roughness, metallic):
        """Apply material properties to a prim."""
        # Implementation to apply material properties
        # This would involve creating and assigning materials in USD
        pass
    
    def randomize_camera(self):
        """Apply randomization to camera parameters."""
        # Randomize field of view
        fov = random.uniform(*self.camera_params['fov_range'])
        
        # Randomize camera position
        pos_variance = self.camera_params['position_variance']
        x_offset = random.uniform(-pos_variance, pos_variance)
        y_offset = random.uniform(-pos_variance, pos_variance)
        z_offset = random.uniform(-pos_variance/2, pos_variance/2)
        
        # Apply camera randomization (this would involve updating camera settings)
        pass
    
    def generate_diverse_scenes(self, num_samples=1000):
        """Generate diverse scenes with domain randomization."""
        for i in range(num_samples):
            # Create random environment configuration
            env_config = self.generate_random_environment_config()
            
            # Generate sample with this configuration
            self.generate_single_sample(env_config)
            
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples")
    
    def generate_random_environment_config(self):
        """Generate a random environment configuration."""
        # Random number of objects
        num_objects = random.randint(3, 10)
        
        objects = []
        for _ in range(num_objects):
            obj_config = {
                "type": random.choice(["cube", "sphere", "cylinder"]),
                "position": [
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(0.2, 2)
                ],
                "color": [random.random(), random.random(), random.random()]
            }
            objects.append(obj_config)
        
        lighting = {
            "intensity": random.uniform(*self.lighting_params['intensity_range']),
            "color_temp": random.uniform(*self.lighting_params['color_temperature_range'])
        }
        
        return {
            "objects": objects,
            "lighting": lighting
        }
```

## Isaac Sim ROS 2 Integration

### Setting up Isaac Sim with ROS 2 Bridge

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.synthetic_utils import plot
import carb

class IsaacSimROS2Bridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros2_bridge')
        
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        
        # Initialize camera and sensors
        self.camera = None
        self.bridge = CvBridge()
        
        # ROS publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)
        
        # Timer for publishing data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz
        
        # Initialize the simulation environment
        self.setup_simulation_environment()
        
        self.get_logger().info('Isaac Sim ROS 2 Bridge initialized')
    
    def setup_simulation_environment(self):
        """Set up the simulation environment."""
        # Add ground plane
        from omni.isaac.core.objects import GroundPlane
        self.world.scene.add(GroundPlane(prim_path="/World/defaultGroundPlane", name="ground_plane"))
        
        # Add a simple robot
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        assets_root_path = get_assets_root_path()
        if assets_root_path is not None:
            robot_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
            add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")
        
        # Add a camera sensor
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Robot/base_link/Camera",
                frequency=10,
                resolution=(640, 480)
            )
        )
        
        # Initialize the world
        self.world.reset()
    
    def publish_sensor_data(self):
        """Publish sensor data from Isaac Sim to ROS 2."""
        # Step the physics simulation
        self.world.step(render=True)
        
        # Get camera data
        if self.camera is not None:
            # Get RGB image
            rgb_data = self.camera.get_rgb()
            if rgb_data is not None:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb_data, encoding="rgba8")
                rgb_msg.header.stamp = self.get_clock().now().to_msg()
                rgb_msg.header.frame_id = "camera_rgb_optical_frame"
                self.rgb_pub.publish(rgb_msg)
            
            # Get depth image
            depth_data = self.camera.get_depth()
            if depth_data is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_data, encoding="32FC1")
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = "camera_depth_optical_frame"
                self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            camera_info_msg = self.create_camera_info_msg()
            camera_info_msg.header.stamp = self.get_clock().now().to_msg()
            camera_info_msg.header.frame_id = "camera_rgb_optical_frame"
            self.camera_info_pub.publish(camera_info_msg)
    
    def create_camera_info_msg(self):
        """Create camera info message."""
        camera_info = CameraInfo()
        
        # Set camera parameters (these should match your Isaac Sim camera)
        camera_info.height = 480
        camera_info.width = 640
        camera_info.distortion_model = 'plumb_bob'
        
        # Example intrinsic parameters (you should get these from Isaac Sim)
        camera_info.k = [616.175, 0.0, 311.175,  # fx, 0, cx
                         0.0, 616.175, 229.5,   # 0, fy, cy
                         0.0, 0.0, 1.0]          # 0, 0, 1
        
        camera_info.r = [1.0, 0.0, 0.0,          # R: row 1
                         0.0, 1.0, 0.0,          # R: row 2
                         0.0, 0.0, 1.0]          # R: row 3
        
        camera_info.p = [616.175, 0.0, 311.175, 0.0,  # P: row 1
                         0.0, 616.175, 229.5, 0.0,   # P: row 2
                         0.0, 0.0, 1.0, 0.0]          # P: row 3
        
        return camera_info

def main(args=None):
    rclpy.init(args=args)
    
    isaac_sim_bridge = IsaacSimROS2Bridge()
    
    try:
        rclpy.spin(isaac_sim_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        isaac_sim_bridge.destroy_node()
        rclpy.shutdown()
```

## Perception System Testing in Isaac Sim

### Testing Perception Models in Simulation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

class PerceptionTesterInSim(Node):
    def __init__(self):
        super().__init__('perception_tester_in_sim')
        
        # Load pre-trained perception model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        
        # Image processing
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/perception/detections',
            10
        )
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        self.get_logger().info('Perception Tester in Simulation initialized')
    
    def image_callback(self, msg):
        """Process image and run perception."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run inference
            start_time = self.get_clock().now()
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(img_rgb)
            
            inference_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            self.total_inference_time += inference_time
            self.frame_count += 1
            
            # Process detections
            detections = self.process_detections(results, cv_image.shape)
            
            # Publish detections
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_msg)
            
            # Log performance
            if self.frame_count % 100 == 0:
                avg_time = self.total_inference_time / self.frame_count
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(f'Average inference time: {avg_time:.4f}s ({fps:.2f} FPS)')
            
        except Exception as e:
            self.get_logger().error(f'Perception testing error: {e}')
    
    def process_detections(self, results, image_shape):
        """Process YOLO detection results."""
        detections = []
        
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if conf > 0.5:  # Confidence threshold
                height, width = image_shape[:2]
                
                detection = {
                    'class_id': int(cls),
                    'confidence': conf,
                    'bbox': {
                        'x': int(xyxy[0]),
                        'y': int(xyxy[1]),
                        'width': int(xyxy[2] - xyxy[0]),
                        'height': int(xyxy[3] - xyxy[1])
                    },
                    'center': {
                        'x': int((xyxy[0] + xyxy[2]) / 2),
                        'y': int((xyxy[1] + xyxy[3]) / 2)
                    }
                }
                detections.append(detection)
        
        return detections
    
    def create_detection_message(self, detections, header):
        """Create ROS detection message."""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        for det in detections:
            detection = Detection2D()
            detection.header = header
            
            # Bounding box
            bbox = detection.bbox
            bbox.size_x = det['bbox']['width']
            bbox.size_y = det['bbox']['height']
            
            # Center
            center = detection.bbox.center
            center.x = det['center']['x']
            center.y = det['center']['y']
            
            # Results
            result = ObjectHypothesisWithPose()
            result.id = det['class_id']
            result.score = det['confidence']
            detection.results.append(result)
            
            detection_array.detections.append(detection)
        
        return detection_array

class PerceptionEvaluatorInSim(PerceptionTesterInSim):
    def __init__(self):
        super().__init__()
        
        # Ground truth subscription (from Isaac Sim)
        self.ground_truth_sub = self.create_subscription(
            Detection2DArray,
            '/ground_truth/detections',
            self.ground_truth_callback,
            10
        )
        
        # Evaluation metrics
        self.eval_results = {
            'detections': [],
            'ground_truths': [],
            'tp': 0,  # True positives
            'fp': 0,  # False positives
            'fn': 0,  # False negatives
            'total_frames': 0
        }
        
        self.get_logger().info('Perception Evaluator in Simulation initialized')
    
    def ground_truth_callback(self, msg):
        """Receive ground truth detections."""
        # Store ground truth for evaluation
        gt_detections = []
        for detection in msg.detections:
            gt_detections.append({
                'class_id': detection.results[0].id if detection.results else -1,
                'confidence': detection.results[0].score if detection.results else 0.0,
                'bbox': {
                    'x': int(detection.bbox.center.x - detection.bbox.size_x/2),
                    'y': int(detection.bbox.center.y - detection.bbox.size_y/2),
                    'width': int(detection.bbox.size_x),
                    'height': int(detection.bbox.size_y)
                }
            })
        
        self.eval_results['ground_truths'].append(gt_detections)
    
    def image_callback(self, msg):
        """Process image and evaluate perception."""
        # Call parent method to get detections
        super().image_callback(msg)
        
        # If we have ground truth for this frame, evaluate
        if len(self.eval_results['ground_truths']) > self.eval_results['total_frames']:
            gt = self.eval_results['ground_truths'][self.eval_results['total_frames']]
            # In a real implementation, we'd also have the detections for this frame
            # and compare them to the ground truth
            self.eval_results['total_frames'] += 1
    
    def calculate_metrics(self):
        """Calculate evaluation metrics."""
        # Implementation to calculate precision, recall, mAP, etc.
        pass
```

## Practical Example: Training Perception with Synthetic Data

Here's a complete example showing how to use Isaac Sim for synthetic data generation to train a perception model:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
import os
from PIL import Image

class SyntheticPerceptionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load annotations
        self.annotations = []
        annot_dir = os.path.join(data_dir, "annotations")
        for filename in os.listdir(annot_dir):
            if filename.endswith(".json"):
                with open(os.path.join(annot_dir, filename), 'r') as f:
                    annotation = json.load(f)
                    annotation['image_id'] = filename.replace('.json', '')
                    self.annotations.append(annotation)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        
        # Load image
        img_path = os.path.join(self.data_dir, "rgb", f"{image_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        # Load segmentation mask
        seg_path = os.path.join(self.data_dir, "seg", f"{image_id}.png")
        segmentation = Image.open(seg_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)
        
        return image, segmentation, annotation

class SyntheticDataTrainer:
    def __init__(self, model, dataset_path, batch_size=8, num_epochs=50):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        self.dataset = SyntheticPerceptionDataset(dataset_path, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, targets, annotations) in enumerate(self.dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return running_loss / len(self.dataloader)
    
    def train(self):
        """Train the model."""
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            
            train_loss = self.train_epoch()
            print(f'Train Loss: {train_loss:.4f}')
            
            # Save model checkpoint
            torch.save(self.model.state_dict(), f'perception_model_epoch_{epoch+1}.pth')

def main_training_pipeline():
    """Main training pipeline using synthetic data."""
    # Initialize model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    
    # Replace classifier for our specific classes
    model.classifier[4] = nn.Conv2d(256, 10, kernel_size=(1, 1))  # 10 classes example
    
    # Initialize trainer
    trainer = SyntheticDataTrainer(
        model=model,
        dataset_path="synthetic_data",
        batch_size=8,
        num_epochs=30
    )
    
    # Train the model
    trainer.train()
    
    print("Training completed!")

# Example usage of Isaac Sim for data generation
def generate_training_data():
    """Generate synthetic training data using Isaac Sim."""
    generator = DomainRandomizationGenerator(output_dir="synthetic_training_data")
    
    # Generate 5000 diverse samples
    generator.generate_diverse_scenes(num_samples=5000)
    
    print("Synthetic training data generation completed!")

if __name__ == "__main__":
    # First, generate synthetic data
    generate_training_data()
    
    # Then, train the model with synthetic data
    main_training_pipeline()
```

## Exercises

1. **Environment Creation Exercise**: Create a household environment in Isaac Sim with multiple rooms, furniture, and household objects. Randomize the positions and appearances of objects.

2. **Synthetic Data Generation Exercise**: Generate a dataset of 1000 images with annotations for training a simple object detector. Apply domain randomization to make the dataset diverse.

3. **Integration Exercise**: Integrate Isaac Sim with your ROS 2 perception pipeline and test the perception system in simulation before deploying to hardware.

## Summary

This chapter covered NVIDIA Isaac Sim for photorealistic simulation and synthetic data generation:

- Isaac Sim architecture and capabilities for robotics
- Creating photorealistic environments for humanoid robots
- Generating synthetic datasets with accurate annotations
- Applying domain randomization for robust perception
- Integrating Isaac Sim with ROS 2 for perception testing

Synthetic data generation is crucial for developing robust perception systems that can generalize to real-world conditions, especially important for humanoid robots operating in diverse human environments.

## Next Steps

In the next chapter, we'll explore Isaac ROS, NVIDIA's GPU-accelerated perception and navigation packages, and learn how to implement hardware-accelerated VSLAM and navigation systems for humanoid robots.