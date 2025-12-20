"""
Nav2 Integration with Isaac ROS

This script demonstrates how to integrate Nav2 with Isaac ROS for navigation.
It shows how to send goals to Nav2 and handle feedback from the navigation system.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import tf2_py as tf2
from std_msgs.msg import String
import numpy as np


class IsaacNav2Integrator(Node):
    def __init__(self):
        super().__init__('isaac_nav2_integrator')
        
        # Create action client for Nav2
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Create subscribers for sensor data
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        
        # Publisher for status
        self.status_pub = self.create_publisher(String, 'nav2_status', 10)
        
        # TF buffer and listener for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Navigation state
        self.current_goal = None
        self.navigation_active = False
        
        # Wait for Nav2 action server
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info('Nav2 action server available')

    def send_goal(self, x, y, theta):
        """Send a navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        
        # Set the goal pose
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # Convert Euler angle to quaternion
        from math import sin, cos
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = sin(theta / 2.0)
        quat.w = cos(theta / 2.0)
        goal_msg.pose.pose.orientation = quat
        
        # Store the current goal
        self.current_goal = (x, y, theta)
        
        # Send the goal
        self.get_logger().info(f'Sending navigation goal: ({x}, {y}, {theta})')
        self.navigation_active = True
        
        # Send async goal
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return send_goal_future

    def goal_response_callback(self, future):
        """Handle the goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.navigation_active = False
            return

        self.get_logger().info('Goal accepted')
        
        # Get result future
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the result of the navigation action"""
        result = future.result().result
        status = future.result().status
        
        if status == 4:  # Goal succeeded
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().info(f'Navigation failed with status: {status}')
        
        self.navigation_active = False
        
        # Publish status
        status_msg = String()
        status_msg.data = f'Navigation completed with status: {status}'
        self.status_pub.publish(status_msg)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the navigation action"""
        feedback = feedback_msg.feedback
        # In a real implementation, you might use this feedback for additional processing
        self.get_logger().debug(f'Navigation feedback: {feedback.current_pose}')

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        if not self.navigation_active:
            return
            
        # Check for obstacles in the path
        min_distance = min(msg.ranges)
        min_angle_idx = msg.ranges.index(min_distance)
        angle_increment = msg.angle_increment
        min_angle = msg.angle_min + min_angle_idx * angle_increment
        
        # If obstacle is too close, consider stopping navigation
        if min_distance < 0.5:  # 0.5 meters threshold
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, angle {min_angle:.2f}rad')
            # In a real implementation, you might cancel the current goal or replan

    def transform_pose(self, pose, target_frame):
        """Transform a pose to a different coordinate frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            transformed_pose = do_transform_pose(pose, transform)
            return transformed_pose
        except tf2.LookupException as ex:
            self.get_logger().error(f'Could not transform pose: {ex}')
            return None
        except tf2.ExtrapolationException as ex:
            self.get_logger().error(f'Could not transform pose: {ex}')
            return None


def main(args=None):
    rclpy.init(args=args)
    
    integrator = IsaacNav2Integrator()
    
    # Example: Send a goal after a short delay
    def send_example_goal():
        # Send a goal to (2.0, 2.0, 0.0) - 2m forward and 2m left, no rotation
        integrator.send_goal(2.0, 2.0, 0.0)
    
    # Use a timer to send an example goal after 2 seconds
    timer = integrator.create_timer(2.0, send_example_goal)
    
    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        integrator.get_logger().info('Interrupted, cancelling goal if active...')
        # In a real implementation, you would cancel the active goal here
    finally:
        integrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()