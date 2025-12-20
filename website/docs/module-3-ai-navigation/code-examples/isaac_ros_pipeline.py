"""
Isaac ROS Pipeline Example

This script demonstrates setting up a basic Isaac ROS pipeline for perception and navigation.
It shows how to connect various sensors to Isaac ROS processing nodes.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np


class IsaacROSPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_pipeline')
        
        # Create publishers for processed data
        self.image_pub = self.create_publisher(Image, 'processed_image', 10)
        self.odom_pub = self.create_publisher(Odometry, 'robot_odom', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Create subscribers for sensor data
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            'lidar/points',
            self.pointcloud_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Initialize CvBridge for image processing
        self.bridge = CvBridge()
        
        # Timer for processing loop
        self.timer = self.create_timer(0.1, self.process_loop)  # 10Hz
        
        self.get_logger().info('Isaac ROS Pipeline initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Perform basic image processing (example: edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Convert back to ROS Image and publish
            processed_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
            processed_msg.header = msg.header
            self.image_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        self.get_logger().info(f'Received point cloud with {msg.height * msg.width} points')
        # In a real implementation, this would process the point cloud for navigation
        # For example, obstacle detection, ground plane segmentation, etc.

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.get_logger().info('Received camera calibration data')

    def process_loop(self):
        """Main processing loop"""
        # This is where the main perception and navigation logic would go
        # For example: path planning, obstacle avoidance, etc.
        
        # For now, just publish a simple odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Set dummy position (in a real implementation, this would come from localization)
        odom_msg.pose.pose.position.x = 0.0
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.position.z = 0.0
        
        # Publish odometry
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    
    pipeline = IsaacROSPipeline()
    
    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()