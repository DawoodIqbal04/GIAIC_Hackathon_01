"""
VSLAM Example with Isaac ROS

This script demonstrates a basic Visual SLAM implementation that could work with Isaac ROS.
It includes feature detection, tracking, and pose estimation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header


class IsaacVSLAM(Node):
    def __init__(self):
        super().__init__('isaac_vsalm')
        
        # Create publishers
        self.odom_pub = self.create_publisher(Odometry, 'visual_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'visual_pose', 10)
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # TF broadcaster for camera pose
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # SLAM state variables
        self.prev_image = None
        self.prev_kp = None
        self.prev_desc = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # w, x, y, z quaternion
        self.initialized = False
        
        # Feature detector (using ORB as an example)
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Parameters
        self.min_matches = 10
        self.max_distance = 50.0  # Maximum distance for valid matches
        
        self.get_logger().info('Isaac VSLAM node initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.get_logger().info('Received camera calibration data')

    def image_callback(self, msg):
        """Process incoming camera image for SLAM"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Initialize on first image
            if not self.initialized:
                self.initialize_slam(cv_image)
                return
            
            # Process frame for SLAM
            success, rvec, tvec = self.process_frame(cv_image)
            
            if success:
                # Update position based on estimated transformation
                self.update_position(rvec, tvec)
                
                # Publish odometry and pose
                self.publish_odometry(msg.header)
                self.publish_pose(msg.header)
                
                # Broadcast transform
                self.broadcast_transform(msg.header)
            
            # Store current frame for next iteration
            self.prev_image = cv_image.copy()
            
        except Exception as e:
            self.get_logger().error(f'Error in VSLAM: {e}')

    def initialize_slam(self, image):
        """Initialize SLAM with the first image"""
        # Detect features in the initial image
        kp, desc = self.detect_features(image)
        
        if kp is not None and desc is not None:
            self.prev_kp = kp
            self.prev_desc = desc
            self.initialized = True
            self.get_logger().info('VSLAM initialized with initial features')
        else:
            self.get_logger().warn('Could not initialize SLAM: no features detected in first image')

    def detect_features(self, image):
        """Detect features in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp = self.detector.detect(gray, None)
        
        if kp:
            kp, desc = self.detector.compute(gray, kp)
            return kp, desc
        
        return None, None

    def process_frame(self, curr_image):
        """Process current frame against previous frame"""
        if self.prev_desc is None:
            return False, None, None
        
        # Detect features in current image
        curr_kp, curr_desc = self.detect_features(curr_image)
        
        if curr_desc is None:
            return False, None, None
        
        # Match features between previous and current frames
        matches = self.match_features(self.prev_desc, curr_desc)
        
        if len(matches) < self.min_matches:
            self.get_logger().warn(f'Not enough matches: {len(matches)} < {self.min_matches}')
            return False, None, None
        
        # Get matched keypoints
        prev_matched_kp = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_matched_kp = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate essential matrix and decompose to get rotation and translation
        E, mask = cv2.findEssentialMat(
            curr_matched_kp, 
            prev_matched_kp, 
            self.camera_matrix, 
            method=cv2.RANSAC, 
            threshold=1.0
        )
        
        if E is None or E.size == 0:
            return False, None, None
        
        # Decompose essential matrix
        _, R, t, _ = cv2.recoverPose(E, curr_matched_kp, prev_matched_kp, self.camera_matrix)
        
        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(R)
        
        # Store current keypoints and descriptors for next iteration
        self.prev_kp = curr_kp
        self.prev_desc = curr_desc
        
        return True, rvec, t

    def match_features(self, desc1, desc2):
        """Match features between two sets of descriptors"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Use FLANN matcher for better performance with ORB descriptors
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches

    def update_position(self, rvec, tvec):
        """Update the estimated position based on rotation and translation"""
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Convert to homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        # Update position (scale translation based on expected movement)
        translation_scale = 0.1  # Adjust based on expected scale
        self.position += translation_scale * T[:3, 3]
        
        # Update orientation (simplified - in a real implementation, proper quaternion math would be used)
        # For now, we'll just keep the orientation as identity
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    def publish_odometry(self, header):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'camera'
        
        # Set position
        odom_msg.pose.pose.position.x = float(self.position[0])
        odom_msg.pose.pose.position.y = float(self.position[1])
        odom_msg.pose.pose.position.z = float(self.position[2])
        
        # Set orientation
        odom_msg.pose.pose.orientation.w = float(self.orientation[0])
        odom_msg.pose.pose.orientation.x = float(self.orientation[1])
        odom_msg.pose.pose.orientation.y = float(self.orientation[2])
        odom_msg.pose.pose.orientation.z = float(self.orientation[3])
        
        # For simplicity, set velocity to zero
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0
        
        self.odom_pub.publish(odom_msg)

    def publish_pose(self, header):
        """Publish pose message"""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = float(self.position[0])
        pose_msg.pose.position.y = float(self.position[1])
        pose_msg.pose.position.z = float(self.position[2])
        
        pose_msg.pose.orientation.w = float(self.orientation[0])
        pose_msg.pose.orientation.x = float(self.orientation[1])
        pose_msg.pose.orientation.y = float(self.orientation[2])
        pose_msg.pose.orientation.z = float(self.orientation[3])
        
        self.pose_pub.publish(pose_msg)

    def broadcast_transform(self, header):
        """Broadcast camera transform"""
        t = TransformStamped()
        
        t.header.stamp = header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera'
        
        t.transform.translation.x = float(self.position[0])
        t.transform.translation.y = float(self.position[1])
        t.transform.translation.z = float(self.position[2])
        
        t.transform.rotation.w = float(self.orientation[0])
        t.transform.rotation.x = float(self.orientation[1])
        t.transform.rotation.y = float(self.orientation[2])
        t.transform.rotation.z = float(self.orientation[3])
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    
    vsalm_node = IsaacVSLAM()
    
    try:
        rclpy.spin(vsalm_node)
    except KeyboardInterrupt:
        pass
    finally:
        vsalm_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()