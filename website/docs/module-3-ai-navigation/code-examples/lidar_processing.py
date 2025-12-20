"""
LiDAR Processing with Isaac ROS

This script demonstrates processing LiDAR data using Isaac ROS components,
including point cloud filtering, obstacle detection, and ground plane segmentation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point32, Vector3
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import struct
from collections import deque


class IsaacLidarProcessor(Node):
    def __init__(self):
        super().__init__('isaac_lidar_processor')
        
        # Create publishers
        self.obstacle_pub = self.create_publisher(MarkerArray, 'lidar/obstacles', 10)
        self.ground_pub = self.create_publisher(Marker, 'lidar/ground_plane', 10)
        self.filtered_pub = self.create_publisher(PointCloud2, 'lidar/filtered_points', 10)
        
        # Create subscriber for LiDAR data
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            'lidar/points',
            self.lidar_callback,
            10
        )
        
        # Parameters for processing
        self.ground_threshold = 0.1  # Threshold for ground detection (meters)
        self.obstacle_height_threshold = 0.3  # Minimum height for obstacles (meters)
        self.obstacle_distance_threshold = 3.0  # Maximum distance for obstacles (meters)
        
        self.get_logger().info('Isaac LiDAR Processor initialized')

    def lidar_callback(self, msg):
        """Process incoming LiDAR point cloud"""
        try:
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            
            if points.size == 0:
                return
                
            # Segment ground plane using RANSAC
            ground_points, obstacle_points = self.segment_ground_plane(points)
            
            # Filter obstacle points to remove distant points
            nearby_obstacles = obstacle_points[
                np.sqrt(obstacle_points[:, 0]**2 + obstacle_points[:, 1]**2) < self.obstacle_distance_threshold
            ]
            
            # Publish filtered point cloud (obstacles only)
            filtered_msg = self.create_filtered_pointcloud(msg, nearby_obstacles)
            self.filtered_pub.publish(filtered_msg)
            
            # Publish ground plane visualization
            ground_marker = self.create_ground_marker(msg.header, ground_points)
            self.ground_pub.publish(ground_marker)
            
            # Publish obstacle markers
            obstacle_markers = self.create_obstacle_markers(msg.header, nearby_obstacles)
            self.obstacle_pub.publish(obstacle_markers)
            
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        # Read the binary data and convert to numpy array
        points = []
        format_string = self.get_point_format(cloud_msg)
        
        for point in point_cloud2.read_points(cloud_msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])  # x, y, z
        
        return np.array(points)

    def get_point_format(self, cloud_msg):
        """Get the format string for unpacking point data"""
        fmt = ''
        for field in cloud_msg.fields:
            if field.datatype == PointField.FLOAT32:
                fmt += 'f'
            elif field.datatype == PointField.FLOAT64:
                fmt += 'd'
        return fmt

    def segment_ground_plane(self, points):
        """Segment ground plane using a simplified approach"""
        # This is a simplified ground segmentation
        # In Isaac ROS, you would use more sophisticated methods like RANSAC
        
        # Filter points near z=0 (ground level)
        ground_mask = np.abs(points[:, 2]) < self.ground_threshold
        ground_points = points[ground_mask]
        
        # Remaining points are potential obstacles
        obstacle_mask = ~ground_mask
        obstacle_points = points[obstacle_mask]
        
        # Filter obstacles that are above ground level
        obstacle_points = obstacle_points[obstacle_points[:, 2] > self.obstacle_height_threshold]
        
        return ground_points, obstacle_points

    def create_filtered_pointcloud(self, header, points):
        """Create a PointCloud2 message from numpy array"""
        # Create PointCloud2 message
        filtered_msg = PointCloud2()
        filtered_msg.header = header
        filtered_msg.height = 1
        filtered_msg.width = len(points)
        filtered_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        filtered_msg.is_bigendian = False
        filtered_msg.point_step = 12  # 3 floats * 4 bytes
        filtered_msg.row_step = filtered_msg.point_step * filtered_msg.width
        filtered_msg.is_dense = True
        
        # Pack points into binary data
        data = []
        for point in points:
            data.append(struct.pack('fff', point[0], point[1], point[2]))
        
        filtered_msg.data = b''.join(data)
        
        return filtered_msg

    def create_ground_marker(self, header, ground_points):
        """Create a marker for visualizing the ground plane"""
        marker = Marker()
        marker.header = header
        marker.ns = "ground_plane"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Use a few points to represent the ground plane
        if len(ground_points) > 0:
            # Calculate ground plane center and dimensions
            x_mean = np.mean(ground_points[:, 0])
            y_mean = np.mean(ground_points[:, 1])
            x_range = np.max(ground_points[:, 0]) - np.min(ground_points[:, 0])
            y_range = np.max(ground_points[:, 1]) - np.min(ground_points[:, 1])
            
            # Create a rectangle representing the ground plane
            scale_factor = 1.5
            marker.points = [
                Point32(x=x_mean - x_range*scale_factor/2, y=y_mean - y_range*scale_factor/2, z=0.0),
                Point32(x=x_mean + x_range*scale_factor/2, y=y_mean - y_range*scale_factor/2, z=0.0),
                Point32(x=x_mean + x_range*scale_factor/2, y=y_mean + y_range*scale_factor/2, z=0.0),
                Point32(x=x_mean - x_range*scale_factor/2, y=y_mean + y_range*scale_factor/2, z=0.0),
                Point32(x=x_mean - x_range*scale_factor/2, y=y_mean - y_range*scale_factor/2, z=0.0),
            ]
        
        marker.scale.x = 0.1  # Line width
        marker.color.a = 0.5  # Alpha
        marker.color.r = 0.0  # Red
        marker.color.g = 1.0  # Green
        marker.color.b = 0.0  # Blue
        
        return marker

    def create_obstacle_markers(self, header, obstacle_points):
        """Create markers for visualizing obstacles"""
        marker_array = MarkerArray()
        
        # Group nearby points into obstacles
        obstacles = self.group_obstacles(obstacle_points)
        
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header = header
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position at the center of the obstacle cluster
            marker.pose.position.x = np.mean(obstacle[:, 0])
            marker.pose.position.y = np.mean(obstacle[:, 1])
            marker.pose.position.z = np.mean(obstacle[:, 2]) + 0.5  # Height offset
            
            # Orientation (upright cylinder)
            marker.pose.orientation.w = 1.0
            
            # Scale based on cluster size
            marker.scale.x = max(0.2, np.std(obstacle[:, 0]) * 2)  # x-diameter
            marker.scale.y = max(0.2, np.std(obstacle[:, 1]) * 2)  # y-diameter
            marker.scale.z = max(0.5, np.max(obstacle[:, 2]) - np.min(obstacle[:, 2]))  # height
            
            # Color based on distance
            distance = np.sqrt(marker.pose.position.x**2 + marker.pose.position.y**2)
            intensity = min(1.0, distance / self.obstacle_distance_threshold)
            marker.color.r = intensity
            marker.color.g = 1.0 - intensity
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        return marker_array

    def group_obstacles(self, points, distance_threshold=0.5):
        """Group nearby points into obstacle clusters"""
        if len(points) == 0:
            return []
        
        # Simple clustering: group points within distance threshold
        clusters = []
        visited = [False] * len(points)
        
        for i, point in enumerate(points):
            if visited[i]:
                continue
                
            cluster = [point]
            visited[i] = True
            
            # Find all points within threshold distance
            for j in range(i + 1, len(points)):
                if not visited[j] and np.linalg.norm(point - points[j]) < distance_threshold:
                    cluster.append(points[j])
                    visited[j] = True
            
            clusters.append(np.array(cluster))
        
        return clusters


def main(args=None):
    rclpy.init(args=args)
    
    processor = IsaacLidarProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()