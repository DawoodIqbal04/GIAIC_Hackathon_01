---
id: chapter-9-advanced-perception
title: "Chapter 9: Advanced Perception and Training"
sidebar_label: "Chapter 9: Advanced Perception and Training"
description: "Advanced AI-based perception systems and training methodologies for humanoid robotics"
keywords: [perception, ai, computer-vision, deep-learning, robotics, humanoid, training, neural-networks]
tags: [ai-perception, computer-vision, deep-learning]
authors: [book-authors]
difficulty: advanced
estimated_time: "105 minutes"
module: 3
chapter: 9
prerequisites: [python-ai-basics, ros2-foundations, computer-vision-basics, deep-learning-fundamentals]
learning_objectives:
  - Understand advanced AI perception techniques for robotics
  - Implement deep learning models for object detection and recognition
  - Design training methodologies for robotic perception systems
  - Integrate perception systems with ROS 2
  - Evaluate perception system performance and accuracy
related:
  - next: chapter-10-isaac-sim-generation
  - previous: module-3-intro
  - see_also: [chapter-10-isaac-sim-generation, chapter-11-isaac-ros-navigation, ../module-2-digital-twins/chapter-8-sensor-simulation]
---

# Chapter 9: Advanced Perception and Training

## Learning Objectives

After completing this chapter, you will be able to:
- Implement advanced AI perception systems using deep learning
- Design and train neural networks for robotic perception tasks
- Integrate perception systems with ROS 2 communication patterns
- Evaluate perception system performance and accuracy
- Apply synthetic data generation techniques for perception training

## Introduction

Perception is the foundation of autonomous robotic behavior. For humanoid robots operating in human environments, perception systems must be robust, accurate, and capable of understanding complex scenes. This chapter explores advanced AI-based perception techniques that enable humanoid robots to understand their environment, recognize objects and humans, and make informed decisions based on visual input.

Modern robotic perception relies heavily on deep learning techniques, particularly convolutional neural networks (CNNs) and transformer architectures. These approaches have revolutionized the field by enabling robots to recognize objects, estimate depth, understand scenes, and track humans with unprecedented accuracy.

## Advanced Perception Techniques

### Object Detection and Recognition

Object detection and recognition are fundamental capabilities for humanoid robots. These systems allow robots to identify and locate objects in their environment, which is essential for manipulation tasks and navigation.

#### YOLO (You Only Look Once) for Real-time Detection

YOLO is a popular real-time object detection system that provides a good balance between speed and accuracy:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
from yolov5.models.experimental import attempt_load
import numpy as np

class YOLOPerceptionNode(Node):
    def __init__(self):
        super().__init__('yolo_perception_node')
        
        # Initialize YOLO model
        self.model = attempt_load('yolov5s.pt')  # Load pretrained model
        self.model.eval()
        
        # Image processing
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.detection_pub = self.create_publisher(
            DetectionArray,
            '/object_detections',
            10
        )
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        
        self.get_logger().info('YOLO Perception Node initialized')

    def image_callback(self, msg):
        """Process image and detect objects."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess image
            input_tensor = self.preprocess_image(cv_image)
            
            # Run inference
            with torch.no_grad():
                results = self.model(input_tensor)
            
            # Process detections
            detections = self.process_detections(results, cv_image.shape)
            
            # Publish detections
            detection_msg = self.create_detection_message(detections)
            self.detection_pub.publish(detection_msg)
            
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

    def preprocess_image(self, image):
        """Preprocess image for YOLO model."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        input_tensor = self.transform(image_rgb).unsqueeze(0)
        
        return input_tensor

    def process_detections(self, results, image_shape):
        """Process YOLO detection results."""
        # Apply non-maximum suppression
        detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
        
        processed_detections = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # Convert to relative coordinates
            height, width = image_shape[:2]
            relative_coords = {
                'x_center': (x1 + x2) / 2 / width,
                'y_center': (y1 + y2) / 2 / height,
                'width': (x2 - x1) / width,
                'height': (y2 - y1) / height
            }
            
            processed_detections.append({
                'class_id': int(cls),
                'confidence': float(conf),
                'bbox': relative_coords,
                'absolute_bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            })
        
        return processed_detections

    def create_detection_message(self, detections):
        """Create ROS message from detections."""
        # Implementation to create DetectionArray message
        # This would include creating Detection2D messages for each detection
        pass
```

#### Transformer-based Perception

Vision Transformers (ViTs) and other transformer architectures have shown excellent performance in perception tasks:

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VisionTransformerPerception(nn.Module):
    def __init__(self, num_classes=80, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        # Classification head
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        # Get hidden states from ViT
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        
        # Use [CLS] token for classification
        cls_output = sequence_output[:, 0]  # First token
        logits = self.classifier(cls_output)
        
        return logits

class TransformerPerceptionNode(Node):
    def __init__(self):
        super().__init__('transformer_perception_node')
        
        # Initialize transformer model
        self.model = VisionTransformerPerception(num_classes=80)
        self.model.eval()
        
        # Image processing
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.get_logger().info('Transformer Perception Node initialized')
```

### Semantic Segmentation

Semantic segmentation provides pixel-level understanding of scenes, which is crucial for humanoid robots to navigate complex environments:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation_node')
        
        # Load pre-trained DeepLabV3 model
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.eval()
        
        # Image processing
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.segmentation_pub = self.create_publisher(
            Image,  # Publish segmentation mask
            '/segmentation_mask',
            10
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.get_logger().info('Semantic Segmentation Node initialized')

    def image_callback(self, msg):
        """Process image and generate segmentation."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess image
            input_tensor = self.transform(cv_image).unsqueeze(0)
            
            # Run segmentation
            with torch.no_grad():
                output = self.model(input_tensor)['out']
                predicted_mask = output.argmax(1).squeeze().cpu().numpy()
            
            # Convert mask to ROS image and publish
            mask_image = self.bridge.cv2_to_imgmsg(predicted_mask.astype(np.uint8), encoding="mono8")
            self.segmentation_pub.publish(mask_image)
            
        except Exception as e:
            self.get_logger().error(f'Segmentation error: {e}')
```

### Depth Estimation

Depth estimation is crucial for humanoid robots to understand 3D structure and navigate safely:

```python
import torch
import torchvision.transforms as transforms
from torchvision.models import MiDaS, MiDaSConfig

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        
        # Load MiDaS model for depth estimation
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
        self.model.eval()
        
        # Image processing
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.depth_pub = self.create_publisher(
            Image,  # Publish depth map
            '/depth_map',
            10
        )
        
        # MiDaS transforms
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.default_transform
        
        self.get_logger().info('Depth Estimation Node initialized')

    def image_callback(self, msg):
        """Process image and estimate depth."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Apply transforms
            input_batch = self.transform(cv_image).unsqueeze(0)
            
            # Run depth estimation
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=cv_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                depth_map = prediction.cpu().numpy()
            
            # Normalize depth map to 0-255 for visualization
            depth_normalized = ((depth_map - depth_map.min()) / 
                               (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            # Convert to ROS image and publish
            depth_image = self.bridge.cv2_to_imgmsg(depth_normalized, encoding="mono8")
            self.depth_pub.publish(depth_image)
            
        except Exception as e:
            self.get_logger().error(f'Depth estimation error: {e}')
```

## Training Perception Systems

### Data Collection and Annotation

Training effective perception systems requires high-quality, diverse datasets:

```python
import os
import json
import cv2
import numpy as np
from PIL import Image

class PerceptionDataCollector:
    def __init__(self, dataset_path, annotations_path):
        self.dataset_path = dataset_path
        self.annotations_path = annotations_path
        self.collected_data = []
        
    def collect_sensor_data(self, camera_image, depth_image, lidar_data):
        """Collect synchronized sensor data."""
        # Create unique ID for this sample
        sample_id = f"sample_{len(self.collected_data):06d}"
        
        # Save images
        image_path = os.path.join(self.dataset_path, f"{sample_id}_rgb.jpg")
        depth_path = os.path.join(self.dataset_path, f"{sample_id}_depth.png")
        
        cv2.imwrite(image_path, camera_image)
        cv2.imwrite(depth_path, depth_image)
        
        # Store sample info
        sample_info = {
            'id': sample_id,
            'rgb_path': image_path,
            'depth_path': depth_path,
            'lidar_path': f"{sample_id}_lidar.bin",
            'timestamp': time.time(),
            'annotations': {}  # Will be filled during annotation
        }
        
        self.collected_data.append(sample_info)
        
        # Save LiDAR data
        self.save_lidar_data(lidar_data, sample_id)
        
        return sample_info

    def save_lidar_data(self, lidar_data, sample_id):
        """Save LiDAR point cloud data."""
        lidar_path = os.path.join(self.dataset_path, f"{sample_id}_lidar.bin")
        lidar_data.astype(np.float32).tofile(lidar_path)

    def annotate_sample(self, sample_id, annotations):
        """Add annotations to a collected sample."""
        for sample in self.collected_data:
            if sample['id'] == sample_id:
                sample['annotations'] = annotations
                break

    def export_dataset(self):
        """Export dataset in COCO format."""
        coco_format = {
            "info": {
                "description": "Humanoid Robot Perception Dataset",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories (object classes)
        categories = [
            {"id": 1, "name": "human", "supercategory": "person"},
            {"id": 2, "name": "chair", "supercategory": "furniture"},
            {"id": 3, "name": "table", "supercategory": "furniture"},
            # Add more categories as needed
        ]
        coco_format["categories"] = categories
        
        # Add images and annotations
        for i, sample in enumerate(self.collected_data):
            image_info = {
                "id": i,
                "file_name": os.path.basename(sample['rgb_path']),
                "height": 480,  # Set based on actual image dimensions
                "width": 640,
                "date_captured": sample['timestamp']
            }
            coco_format["images"].append(image_info)
        
        # Export to JSON
        with open(os.path.join(self.annotations_path, "dataset.json"), 'w') as f:
            json.dump(coco_format, f)
```

### Training Pipeline

A complete training pipeline for perception systems:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class PerceptionDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        
        # Load image
        img_path = os.path.join('path_to_images', img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = [ann for ann in self.annotations['annotations'] 
                      if ann['image_id'] == img_info['id']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, annotations

class PerceptionTrainer:
    def __init__(self, model, dataset_path, batch_size=8, num_epochs=50):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        self.train_dataset = PerceptionDataset(
            os.path.join(dataset_path, 'train_annotations.json'),
            transform=self.train_transform
        )
        
        self.val_dataset = PerceptionDataset(
            os.path.join(dataset_path, 'val_annotations.json'),
            transform=self.train_transform  # Same transform for simplicity
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4
        )
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return running_loss / len(self.train_loader)

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return val_loss / len(self.val_loader), accuracy

    def train(self):
        """Train the model."""
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_perception_model.pth')
                print(f'Saved best model with accuracy: {val_acc:.2f}%')
```

## Synthetic Data Generation

### Using NVIDIA Isaac Sim for Data Generation

Synthetic data generation is crucial for training robust perception systems:

```python
# This would be a more complex implementation using Isaac Sim APIs
class SyntheticDataGenerator:
    def __init__(self):
        # Initialize Isaac Sim environment
        self.isaac_env = self.initialize_isaac_sim()
        
    def initialize_isaac_sim(self):
        """Initialize Isaac Sim environment."""
        # This would use Isaac Sim Python API
        # For now, we'll outline the structure
        pass
    
    def generate_varied_scenes(self, num_scenes=1000):
        """Generate varied scenes with different lighting, objects, and configurations."""
        for i in range(num_scenes):
            # Randomly configure scene
            self.configure_scene_randomly()
            
            # Capture multiple viewpoints
            for view_idx in range(5):  # Different camera angles
                self.set_camera_view(view_idx)
                rgb_image, depth_image, segmentation = self.capture_scene()
                
                # Save synthetic data
                self.save_synthetic_sample(rgb_image, depth_image, segmentation, i, view_idx)
    
    def configure_scene_randomly(self):
        """Randomly configure the scene with objects, lighting, etc."""
        # Randomly place objects
        # Randomize lighting conditions
        # Randomize camera positions
        pass
    
    def save_synthetic_sample(self, rgb, depth, segmentation, scene_id, view_id):
        """Save synthetic sample with annotations."""
        # Save images
        cv2.imwrite(f"synthetic_data/rgb_{scene_id:06d}_{view_id}.png", rgb)
        cv2.imwrite(f"synthetic_data/depth_{scene_id:06d}_{view_id}.png", depth)
        cv2.imwrite(f"synthetic_data/seg_{scene_id:06d}_{view_id}.png", segmentation)
        
        # Save annotations
        annotation = {
            'scene_id': scene_id,
            'view_id': view_id,
            'objects': self.get_scene_objects(),
            'camera_pose': self.get_camera_pose()
        }
        
        with open(f"synthetic_data/annotations_{scene_id:06d}_{view_id}.json", 'w') as f:
            json.dump(annotation, f)
```

## ROS 2 Integration

### Perception Pipeline Node

A complete perception pipeline integrated with ROS 2:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2

class PerceptionPipelineNode(Node):
    def __init__(self):
        super().__init__('perception_pipeline_node')
        
        # Initialize perception models
        self.object_detector = self.initialize_object_detector()
        self.segmentation_model = self.initialize_segmentation_model()
        self.depth_estimator = self.initialize_depth_estimator()
        
        # Image processing
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers for perception results
        self.detection_pub = self.create_publisher(DetectionArray, '/detections', 10)
        self.segmentation_pub = self.create_publisher(Image, '/segmentation', 10)
        self.depth_pub = self.create_publisher(Image, '/estimated_depth', 10)
        
        # Store camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        self.get_logger().info('Perception Pipeline Node initialized')

    def initialize_object_detector(self):
        """Initialize object detection model."""
        # Load YOLO or other detection model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model

    def initialize_segmentation_model(self):
        """Initialize segmentation model."""
        # Load DeepLab or other segmentation model
        model = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            'deeplabv3_resnet50', 
            pretrained=True
        )
        model.eval()
        return model

    def initialize_depth_estimator(self):
        """Initialize depth estimation model."""
        # Load MiDaS or other depth model
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
        model.eval()
        return model

    def camera_info_callback(self, msg):
        """Update camera parameters."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process image through full perception pipeline."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run object detection
            detections = self.run_object_detection(cv_image)
            
            # Run semantic segmentation
            segmentation = self.run_segmentation(cv_image)
            
            # Run depth estimation
            depth_map = self.run_depth_estimation(cv_image)
            
            # Publish results
            self.publish_detections(detections, msg.header)
            self.publish_segmentation(segmentation, msg.header)
            self.publish_depth(depth_map, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Perception pipeline error: {e}')

    def run_object_detection(self, image):
        """Run object detection on image."""
        # Convert image for model
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.object_detector(img_rgb)
        
        # Extract detections
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if conf > 0.5:  # Confidence threshold
                detection = {
                    'bbox': [int(x) for x in xyxy],
                    'confidence': conf,
                    'class_id': int(cls)
                }
                detections.append(detection)
        
        return detections

    def run_segmentation(self, image):
        """Run semantic segmentation on image."""
        # Preprocess
        input_tensor = self.preprocess_for_segmentation(image)
        
        # Run segmentation
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out']
            predicted_mask = output.argmax(1).squeeze().cpu().numpy()
        
        return predicted_mask

    def run_depth_estimation(self, image):
        """Run depth estimation on image."""
        # Preprocess for MiDaS
        img_input = self.preprocess_for_depth(image)
        
        # Run depth estimation
        with torch.no_grad():
            prediction = self.depth_estimator(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = prediction.cpu().numpy()
        
        return depth_map

    def preprocess_for_segmentation(self, image):
        """Preprocess image for segmentation model."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_pil).unsqueeze(0)
        return input_tensor

    def preprocess_for_depth(self, image):
        """Preprocess image for depth estimation model."""
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.default_transform
        
        input_batch = transform(image).unsqueeze(0)
        return input_batch

    def publish_detections(self, detections, header):
        """Publish object detections."""
        detection_array = DetectionArray()
        detection_array.header = header
        
        for det in detections:
            detection = Detection2D()
            detection.header = header
            
            # Bounding box
            bbox = BoundingBox2D()
            bbox.size_x = det['bbox'][2] - det['bbox'][0]
            bbox.size_y = det['bbox'][3] - det['bbox'][1]
            
            # Center
            center = Point2D()
            center.x = (det['bbox'][0] + det['bbox'][2]) / 2
            center.y = (det['bbox'][1] + det['bbox'][3]) / 2
            
            detection.bbox = bbox
            detection.center = center
            
            # Results
            result = ObjectHypothesisWithPose()
            result.id = det['class_id']
            result.score = det['confidence']
            detection.results.append(result)
            
            detection_array.detections.append(detection)
        
        self.detection_pub.publish(detection_array)

    def publish_segmentation(self, segmentation, header):
        """Publish segmentation mask."""
        seg_msg = self.bridge.cv2_to_imgmsg(segmentation.astype(np.uint8), encoding="mono8")
        seg_msg.header = header
        self.segmentation_pub.publish(seg_msg)

    def publish_depth(self, depth_map, header):
        """Publish depth map."""
        # Normalize for visualization
        depth_normalized = ((depth_map - depth_map.min()) / 
                           (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        depth_msg = self.bridge.cv2_to_imgmsg(depth_normalized, encoding="mono8")
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)
```

## Performance Evaluation

### Evaluation Metrics

Evaluating perception system performance is crucial:

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, jaccard_score

class PerceptionEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_object_detection(self, predictions, ground_truths, iou_threshold=0.5):
        """Evaluate object detection performance."""
        # Calculate mAP (mean Average Precision)
        aps = []
        
        for class_id in set([gt['class_id'] for gt in ground_truths]):
            # Get predictions and ground truths for this class
            class_preds = [p for p in predictions if p['class_id'] == class_id]
            class_gts = [gt for gt in ground_truths if gt['class_id'] == class_id]
            
            # Calculate AP for this class
            ap = self.calculate_ap_for_class(class_preds, class_gts, iou_threshold)
            aps.append(ap)
        
        # Mean Average Precision
        mAP = np.mean(aps) if aps else 0.0
        
        return {
            'mAP': mAP,
            'AP_per_class': dict(zip(set([gt['class_id'] for gt in ground_truths]), aps))
        }
    
    def calculate_ap_for_class(self, predictions, ground_truths, iou_threshold):
        """Calculate Average Precision for a specific class."""
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize matches
        matched_gts = set()
        tp = []  # true positives
        fp = []  # false positives
        
        for pred in predictions:
            max_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for i, gt in enumerate(ground_truths):
                if i in matched_gts:
                    continue
                
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i
                    
            if max_iou >= iou_threshold:
                # True positive
                tp.append(1)
                fp.append(0)
                matched_gts.add(best_gt_idx)
            else:
                # False positive
                tp.append(0)
                fp.append(1)
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(ground_truths) if len(ground_truths) > 0 else np.array([])
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            if len(recall) > 0:
                precisions_at_recall = precision[recall >= t]
                if len(precisions_at_recall) > 0:
                    ap += np.max(precisions_at_recall) / 11
        
        return ap
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        intersection = inter_width * inter_height
        
        # Calculate union
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def evaluate_segmentation(self, predictions, ground_truths):
        """Evaluate semantic segmentation performance."""
        # Calculate IoU for each class
        ious = []
        for class_id in range(80):  # COCO dataset has 80 classes
            pred_mask = (predictions == class_id)
            gt_mask = (ground_truths == class_id)
            
            if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                # Both are empty, consider as correct
                iou = 1.0
            elif np.sum(gt_mask) == 0 or np.sum(pred_mask) == 0:
                # One is empty, IoU is 0
                iou = 0.0
            else:
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = intersection / union
            
            ious.append(iou)
        
        # Mean IoU
        miou = np.mean(ious)
        
        return {
            'mIoU': miou,
            'IoU_per_class': ious
        }
    
    def evaluate_depth_estimation(self, predictions, ground_truths):
        """Evaluate depth estimation performance."""
        # Calculate common depth metrics
        abs_rel = np.mean(np.abs(predictions - ground_truths) / ground_truths)
        sq_rel = np.mean(((predictions - ground_truths) ** 2) / ground_truths)
        rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(predictions) - np.log(ground_truths)) ** 2))
        
        # Calculate threshold accuracy (how often prediction is within x% of ground truth)
        thresh_1 = np.maximum((ground_truths / predictions), (predictions / ground_truths))
        a1 = np.mean(thresh_1 < 1.25)
        a2 = np.mean(thresh_1 < 1.25 ** 2)
        a3 = np.mean(thresh_1 < 1.25 ** 3)
        
        return {
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'a1': a1,
            'a2': a2,
            'a3': a3
        }
```

## Practical Example: Human Detection and Tracking

Here's a practical example combining perception techniques:

```python
class HumanDetectionTrackingNode(PerceptionPipelineNode):
    def __init__(self):
        super().__init__()
        
        # Additional publishers for human tracking
        self.human_pose_pub = self.create_publisher(PoseArray, '/detected_humans', 10)
        self.human_tracking_pub = self.create_publisher(TrackArray, '/human_tracks', 10)
        
        # Human tracking variables
        self.human_trackers = {}  # Trackers for each detected human
        self.next_human_id = 0
        self.tracking_window = 5  # Track for 5 frames without detection
        
        self.get_logger().info('Human Detection and Tracking Node initialized')

    def image_callback(self, msg):
        """Process image for human detection and tracking."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run human detection (class ID 0 in COCO dataset)
            humans = self.detect_humans(cv_image)
            
            # Update trackers
            active_tracks = self.update_trackers(humans)
            
            # Publish results
            self.publish_human_poses(active_tracks, msg.header)
            self.publish_human_tracks(active_tracks, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Human tracking error: {e}')

    def detect_humans(self, image):
        """Detect humans in image using object detection."""
        # Run object detection
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.object_detector(img_rgb)
        
        humans = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if int(cls) == 0 and conf > 0.7:  # Class 0 is 'person' in COCO, high confidence
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                
                human = {
                    'bbox': [int(x) for x in xyxy],
                    'center': (center_x, center_y),
                    'confidence': conf
                }
                humans.append(human)
        
        return humans

    def update_trackers(self, detected_humans):
        """Update human trackers with new detections."""
        # Implementation of tracking algorithm (e.g., using OpenCV's KCF tracker)
        # For simplicity, this is a basic implementation
        
        active_tracks = []
        
        # For each detected human, find the closest existing track or create new one
        for human in detected_humans:
            # Find closest existing track
            closest_track_id = self.find_closest_track(human, self.human_trackers)
            
            if closest_track_id is not None:
                # Update existing track
                self.human_trackers[closest_track_id]['bbox'] = human['bbox']
                self.human_trackers[closest_track_id]['center'] = human['center']
                self.human_trackers[closest_track_id]['confidence'] = human['confidence']
                self.human_trackers[closest_track_id]['frames_since_seen'] = 0
                active_tracks.append({
                    'id': closest_track_id,
                    'bbox': human['bbox'],
                    'center': human['center'],
                    'confidence': human['confidence']
                })
            else:
                # Create new track
                new_id = self.next_human_id
                self.next_human_id += 1
                
                self.human_trackers[new_id] = {
                    'bbox': human['bbox'],
                    'center': human['center'],
                    'confidence': human['confidence'],
                    'frames_since_seen': 0
                }
                
                active_tracks.append({
                    'id': new_id,
                    'bbox': human['bbox'],
                    'center': human['center'],
                    'confidence': human['confidence']
                })
        
        # Update trackers that weren't seen
        for track_id in list(self.human_trackers.keys()):
            self.human_trackers[track_id]['frames_since_seen'] += 1
            
            # Remove tracks that haven't been seen for too long
            if self.human_trackers[track_id]['frames_since_seen'] > self.tracking_window:
                del self.human_trackers[track_id]
        
        return active_tracks

    def find_closest_track(self, human, trackers):
        """Find the closest existing track to a detection."""
        min_distance = float('inf')
        closest_id = None
        
        for track_id, track in trackers.items():
            if track['frames_since_seen'] > self.tracking_window:
                continue  # Skip tracks that are too old
                
            distance = np.sqrt(
                (human['center'][0] - track['center'][0])**2 + 
                (human['center'][1] - track['center'][1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_id = track_id
        
        # Only return if distance is below threshold
        return closest_id if min_distance < 50 else None  # 50 pixel threshold

    def publish_human_poses(self, tracks, header):
        """Publish detected human poses."""
        pose_array = PoseArray()
        pose_array.header = header
        
        for track in tracks:
            pose = Pose()
            # Convert image coordinates to world coordinates using camera parameters
            if self.camera_matrix is not None:
                # Simple conversion - in practice, you'd use depth information
                pose.position.x = track['center'][0]  # Would need depth conversion
                pose.position.y = track['center'][1]  # Would need depth conversion
                pose.position.z = 1.0  # Assume average human height
                
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.human_pose_pub.publish(pose_array)

    def publish_human_tracks(self, tracks, header):
        """Publish human tracks."""
        # Implementation for track publishing
        # This would create a custom message type for tracking
        pass
```

## Exercises

1. **Implementation Exercise**: Create a complete perception pipeline that combines object detection, segmentation, and depth estimation. Test with sample images and evaluate performance.

2. **Training Exercise**: Collect a small dataset of images from a simulated humanoid robot environment and train a simple classifier to distinguish between different types of furniture.

3. **Integration Exercise**: Integrate the perception pipeline with a navigation system to enable obstacle avoidance based on segmentation results.

## Summary

This chapter covered advanced perception systems for humanoid robotics:

- Deep learning-based object detection, segmentation, and depth estimation
- Training methodologies for perception systems
- Synthetic data generation using simulation environments
- ROS 2 integration for real-time perception
- Performance evaluation metrics for perception systems

Advanced perception systems are essential for humanoid robots to understand and interact with their environment safely and effectively.

## Next Steps

In the next chapter, we'll explore NVIDIA Isaac Sim in detail, learning how to create photorealistic simulation environments and generate synthetic data for training perception systems.