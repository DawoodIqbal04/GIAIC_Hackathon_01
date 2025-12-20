"""
Custom Perception Node for Isaac ROS

This script demonstrates creating a custom perception node that integrates
data from multiple sensors and provides fused perception output.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan, CameraInfo
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header, ColorRGBA
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
import tf2_py as tf2
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque
import threading
import time


class CustomPerceptionNode(Node):
    def __init__(self):
        super().__init__('custom_perception_node')
        
        # Create publishers
        self.detection_pub = self.create_publisher(MarkerArray, 'perception/detections', 10)
        self.fused_output_pub = self.create_publisher(MarkerArray, 'perception/fused_output', 10)
        
        # Create subscribers
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
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
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
        
        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Sensor data storage with timestamps
        self.latest_image = None
        self.latest_image_time = None
        self.latest_pointcloud = None
        self.latest_pointcloud_time = None
        self.latest_scan = None
        self.latest_scan_time = None
        self.camera_info = None
        
        # Processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Parameters
        self.processing_rate = 5.0  # Hz
        self.time_sync_tolerance = 0.1  # seconds
        
        self.get_logger().info('Custom Perception Node initialized')

    def image_callback(self, msg):
        """Store latest image with timestamp"""
        self.latest_image = msg
        self.latest_image_time = time.time()

    def pointcloud_callback(self, msg):
        """Store latest point cloud with timestamp"""
        self.latest_pointcloud = msg
        self.latest_pointcloud_time = time.time()

    def scan_callback(self, msg):
        """Store latest laser scan with timestamp"""
        self.latest_scan = msg
        self.latest_scan_time = time.time()

    def camera_info_callback(self, msg):
        """Store camera information"""
        self.camera_info = msg

    def processing_loop(self):
        """Main processing loop running in a separate thread"""
        rate = self.create_rate(self.processing_rate)
        
        while rclpy.ok():
            try:
                # Check if we have synchronized data
                if self.has_synced_data():
                    # Fuse sensor data
                    detections = self.fuse_sensor_data()
                    
                    # Publish fused detections
                    self.publish_detections(detections)
                
                rate.sleep()
            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {e}')

    def has_synced_data(self):
        """Check if we have reasonably synchronized data from all sensors"""
        if (self.latest_image_time is None or 
            self.latest_pointcloud_time is None or 
            self.latest_scan_time is None or
            self.camera_info is None):
            return False
        
        # Check if all data is within tolerance of each other
        times = [self.latest_image_time, self.latest_pointcloud_time, self.latest_scan_time]
        time_diffs = [abs(t - times[0]) for t in times]
        
        return all(diff < self.time_sync_tolerance for diff in time_diffs)

    def fuse_sensor_data(self):
        """Fuse data from multiple sensors to create detections"""
        detections = []
        
        try:
            # Process image data
            if self.latest_image is not None:
                image_detections = self.process_image_data(self.latest_image)
                detections.extend(image_detections)
            
            # Process point cloud data
            if self.latest_pointcloud is not None:
                pointcloud_detections = self.process_pointcloud_data(self.latest_pointcloud)
                detections.extend(pointcloud_detections)
            
            # Process laser scan data
            if self.latest_scan is not None:
                scan_detections = self.process_scan_data(self.latest_scan)
                detections.extend(scan_detections)
            
            # Fuse overlapping detections
            fused_detections = self.fuse_overlapping_detections(detections)
            
            return fused_detections
            
        except Exception as e:
            self.get_logger().error(f'Error fusing sensor data: {e}')
            return []

    def process_image_data(self, image_msg):
        """Process image data to detect objects"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Perform object detection (simplified example)
            # In a real implementation, this would use Isaac ROS perception packages
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detect red objects as an example
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert image coordinates to 3D world coordinates
                    # This is a simplified approach - in reality, you'd need depth information
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Project to 3D using camera parameters (simplified)
                    if self.camera_info:
                        # Calculate approximate distance (in a real system, use depth data)
                        distance = 2.0  # meters (simplified)
                        
                        # Convert pixel coordinates to world coordinates
                        cx = (center_x - self.camera_info.k[2]) * distance / self.camera_info.k[0]  # cx
                        cy = (center_y - self.camera_info.k[5]) * distance / self.camera_info.k[4]  # cy
                        cz = distance
                        
                        detections.append({
                            'type': 'image_detection',
                            'position': (cx, cy, cz),
                            'size': (w, h, 0.5),  # approximate depth
                            'confidence': 0.8,
                            'label': 'red_object'
                        })
            
            return detections
        except Exception as e:
            self.get_logger().error(f'Error processing image data: {e}')
            return []

    def process_pointcloud_data(self, pointcloud_msg):
        """Process point cloud data to detect objects"""
        try:
            # In a real implementation, you would convert the PointCloud2 message
            # and perform 3D object detection, clustering, or segmentation
            
            # For this example, we'll simulate detection of planar surfaces
            # (like the ground plane or walls)
            
            detections = []
            
            # Simplified ground plane detection (in a real system, use PCL or similar)
            # For now, we'll just simulate detecting a ground plane
            detections.append({
                'type': 'pointcloud_detection',
                'position': (0.0, 0.0, 0.0),
                'size': (5.0, 5.0, 0.1),  # 5x5m ground plane, 0.1m thick
                'confidence': 0.95,
                'label': 'ground_plane'
            })
            
            # Simulate detection of an obstacle
            detections.append({
                'type': 'pointcloud_detection',
                'position': (1.5, 0.0, 0.5),
                'size': (0.5, 0.5, 1.0),  # 0.5x0.5x1.0m obstacle
                'confidence': 0.85,
                'label': 'obstacle'
            })
            
            return detections
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud data: {e}')
            return []

    def process_scan_data(self, scan_msg):
        """Process laser scan data to detect objects"""
        try:
            # Process laser scan to detect objects
            ranges = np.array(scan_msg.ranges)
            angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
            
            # Remove invalid ranges
            valid_indices = np.isfinite(ranges)
            valid_ranges = ranges[valid_indices]
            valid_angles = angles[valid_indices]
            
            # Detect discontinuities in ranges (indicating objects)
            detections = []
            
            # Find significant changes in range values
            range_diff = np.abs(np.diff(valid_ranges))
            threshold = 0.3  # meters
            
            for i, diff in enumerate(range_diff):
                if diff > threshold:
                    # Calculate position of potential object
                    angle = valid_angles[i]
                    range_val = valid_ranges[i]
                    
                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    
                    detections.append({
                        'type': 'scan_detection',
                        'position': (x, y, 0.1),  # Assume height of 0.1m
                        'size': (0.3, 0.3, 0.5),  # approximate size
                        'confidence': 0.7,
                        'label': 'obstacle_from_scan'
                    })
            
            return detections
        except Exception as e:
            self.get_logger().error(f'Error processing scan data: {e}')
            return []

    def fuse_overlapping_detections(self, detections):
        """Fuse overlapping detections from different sensors"""
        if not detections:
            return []
        
        # Group detections by proximity
        fused_detections = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
                
            # Find overlapping detections
            overlapping = [det1]
            used_indices.add(i)
            
            for j, det2 in enumerate(detections):
                if j in used_indices:
                    continue
                    
                # Calculate distance between detections
                pos1 = det1['position']
                pos2 = det2['position']
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
                
                # If detections are close, consider them for fusion
                if dist < 0.5:  # 0.5m threshold
                    overlapping.append(det2)
                    used_indices.add(j)
            
            # Create fused detection from overlapping ones
            if len(overlapping) > 1:
                # Average positions
                avg_x = sum(det['position'][0] for det in overlapping) / len(overlapping)
                avg_y = sum(det['position'][1] for det in overlapping) / len(overlapping)
                avg_z = sum(det['position'][2] for det in overlapping) / len(overlapping)
                
                # Average confidence
                avg_conf = sum(det['confidence'] for det in overlapping) / len(overlapping)
                
                # Determine label based on majority
                labels = [det['label'] for det in overlapping]
                most_common_label = max(set(labels), key=labels.count)
                
                fused_detections.append({
                    'type': 'fused_detection',
                    'position': (avg_x, avg_y, avg_z),
                    'size': overlapping[0]['size'],  # Use size from first detection
                    'confidence': avg_conf,
                    'label': most_common_label
                })
            else:
                fused_detections.append(det1)
        
        return fused_detections

    def publish_detections(self, detections):
        """Publish detections as MarkerArray"""
        marker_array = MarkerArray()
        
        for i, detection in enumerate(detections):
            marker = self.create_detection_marker(i, detection)
            marker_array.markers.append(marker)
        
        self.fused_output_pub.publish(marker_array)

    def create_detection_marker(self, id, detection):
        """Create a visualization marker for a detection"""
        marker = MarkerArray().markers[0] if len(MarkerArray().markers) > 0 else None
        
        from visualization_msgs.msg import Marker
        
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'  # Assuming detections are in map frame
        marker.ns = "perception_detections"
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = detection['position'][0]
        marker.pose.position.y = detection['position'][1]
        marker.pose.position.z = detection['position'][2]
        
        # Orientation (keep upright)
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        
        # Scale
        marker.scale.x = detection['size'][0]
        marker.scale.y = detection['size'][1]
        marker.scale.z = detection['size'][2]
        
        # Color based on detection type
        if detection['label'] == 'ground_plane':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif 'obstacle' in detection['label']:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        
        marker.color.a = detection['confidence']
        
        return marker


def main(args=None):
    rclpy.init(args=args)
    
    perception_node = CustomPerceptionNode()
    
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()