# Exercise 4: Build a Simple Visual SLAM System

## Problem Statement
Build a simple Visual Simultaneous Localization and Mapping (VSLAM) system that can estimate camera pose and build a map of the environment from a sequence of images.

## Learning Objectives
- Understand the fundamentals of Visual SLAM
- Implement visual odometry using feature detection and matching
- Apply bundle adjustment concepts for map optimization
- Evaluate SLAM system performance and limitations

## Implementation Requirements

### 1. Feature Detection and Matching
- Implement feature detection (e.g., ORB, SIFT) to identify keypoints in images
- Implement feature matching between consecutive frames
- Apply RANSAC to filter out incorrect matches

### 2. Pose Estimation
- Calculate the essential matrix from matched features
- Decompose the essential matrix to get rotation and translation
- Implement pose integration to track camera trajectory

### 3. Mapping
- Initialize 3D points from triangulation of matched features
- Track 3D points across frames (data association)
- Implement simple map management (adding/removing points)

### 4. Evaluation
- Visualize the estimated trajectory
- Display tracked 3D points
- Calculate trajectory error if ground truth is available

## Starter Code Template

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

class SimpleVSLAM:
    def __init__(self, 
                 max_features: int = 1000,
                 min_matches: int = 10,
                 min_triangulation_angle: float = 5.0):  # Minimum angle for triangulation (degrees)
        """
        Initialize the VSLAM system.
        
        Args:
            max_features: Maximum number of features to detect per frame
            min_matches: Minimum number of matches required for pose estimation
            min_triangulation_angle: Minimum angle for reliable triangulation
        """
        self.max_features = max_features
        self.min_matches = min_matches
        self.min_triangulation_angle = math.radians(min_triangulation_angle)
        
        # Feature detector and matcher
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Camera parameters (example values - adjust as needed)
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                      [0, self.fy, self.cy],
                                      [0, 0, 1]])
        
        # Current camera pose (relative to initial position)
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.poses = [self.current_pose.copy()]  # Store all poses
        
        # 3D map points (stored in global coordinates)
        self.map_points = []  # List of (x, y, z) coordinates
        self.map_point_observations = []  # List of lists of (frame_id, keypoint_idx)
        
        # Frame data
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.frame_count = 0
        
    def detect_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect features in an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # TODO: Implement feature detection
        pass
    
    def match_features(self, 
                      desc1: np.ndarray, 
                      desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of matches
        """
        # TODO: Implement feature matching with ratio test
        pass
    
    def estimate_pose(self, 
                     kp1: List[cv2.KeyPoint], 
                     kp2: List[cv2.KeyPoint], 
                     matches: List[cv2.DMatch]) -> Tuple[bool, np.ndarray]:
        """
        Estimate relative pose between two frames.
        
        Args:
            kp1: Keypoints from first frame
            kp2: Keypoints from second frame
            matches: Matches between keypoints
            
        Returns:
            Tuple of (success, 4x4 transformation matrix)
        """
        # TODO: Estimate essential matrix and decompose to get pose
        pass
    
    def triangulate_points(self, 
                          kp1: List[cv2.KeyPoint], 
                          kp2: List[cv2.KeyPoint], 
                          matches: List[cv2.DMatch],
                          transform: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Triangulate 3D points from matched keypoints.
        
        Args:
            kp1: Keypoints from first frame
            kp2: Keypoints from second frame
            matches: Matches between keypoints
            transform: Relative transformation between frames
            
        Returns:
            List of 3D points (x, y, z)
        """
        # TODO: Implement triangulation of 3D points
        pass
    
    def process_frame(self, image: np.ndarray) -> bool:
        """
        Process a new frame and update the map.
        
        Args:
            image: Input image
            
        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement the main processing pipeline
        # 1. Detect features in current frame
        # 2. Match with previous frame
        # 3. Estimate pose
        # 4. Triangulate new points
        # 5. Update map
        pass
    
    def visualize_trajectory(self):
        """
        Visualize the estimated camera trajectory.
        """
        # TODO: Create 3D visualization of the trajectory
        pass
    
    def visualize_map(self):
        """
        Visualize the 3D map points.
        """
        # TODO: Create 3D visualization of the map
        pass

class VSLAMTester:
    def __init__(self):
        """Initialize the VSLAM tester."""
        self.slam = SimpleVSLAM()
        
    def test_with_synthetic_data(self):
        """
        Test the VSLAM system with synthetic data.
        """
        # TODO: Generate synthetic images or use a dataset
        # For now, we'll outline the approach:
        # 1. Create synthetic trajectory and 3D points
        # 2. Generate images with known camera poses
        # 3. Run VSLAM on the synthetic images
        # 4. Compare estimated trajectory with ground truth
        pass
    
    def test_with_image_sequence(self, image_paths: List[str]):
        """
        Test the VSLAM system with a sequence of images.
        
        Args:
            image_paths: List of paths to image files
        """
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
                
            success = self.slam.process_frame(image)
            if success:
                print(f"Processed frame {i}, pose: {self.slam.current_pose[:3, 3]}")
            else:
                print(f"Failed to process frame {i}")

# Example usage:
if __name__ == "__main__":
    # Create VSLAM system
    slam = SimpleVSLAM()
    
    # Example: Process a sequence of images
    # For this example, we'll create synthetic images
    # In practice, you would load images from a dataset or camera
    
    # Create a simple synthetic test
    height, width = 480, 640
    for frame_idx in range(10):
        # Create a synthetic image with some features
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Add some random features
        for _ in range(50):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            cv2.circle(img, (x, y), 3, 255, -1)
        
        # Process the frame
        success = slam.process_frame(img)
        if success:
            print(f"Frame {frame_idx} processed successfully")
        else:
            print(f"Frame {frame_idx} processing failed")
    
    # Visualize results
    slam.visualize_trajectory()
```

## Evaluation Criteria
- Accuracy: Estimated trajectory should be close to ground truth (if available)
- Robustness: System should handle varying lighting and viewpoint changes
- Efficiency: Real-time performance on standard hardware
- Map Quality: 3D points should accurately represent the environment
- Code Quality: Well-documented, readable, and maintainable code

## Hints and Resources
- Start with visual odometry (tracking camera motion) before building the map
- Use RANSAC to filter out incorrect feature matches
- Consider the scale ambiguity in monocular SLAM
- Implement proper coordinate system management

## Extensions
- Implement loop closure detection
- Add bundle adjustment for map optimization
- Integrate with ROS for real camera input
- Compare with state-of-the-art SLAM systems (ORB-SLAM, LSD-SLAM, etc.)