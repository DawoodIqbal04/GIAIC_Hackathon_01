"""
Camera Processing with Isaac ROS

This script demonstrates processing camera data using Isaac ROS components,
including image rectification, feature detection, and object recognition.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String


class IsaacCameraProcessor(Node):
    def __init__(self):
        super().__init__('isaac_camera_processor')
        
        # Create publisher for detections
        self.detection_pub = self.create_publisher(Detection2DArray, 'camera/detections', 10)
        self.annotated_image_pub = self.create_publisher(Image, 'camera/annotated_image', 10)
        
        # Create subscriber for camera images
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
        
        # Store camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Load pre-trained model (example with YOLO)
        # In a real Isaac ROS setup, you'd use Isaac ROS perception packages
        self.get_logger().info('Isaac Camera Processor initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.get_logger().info('Received camera calibration data')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Create a copy for annotation
            annotated_image = cv_image.copy()
            
            # Perform object detection (simplified example)
            detections = self.detect_objects(cv_image)
            
            # Create Detection2DArray message
            detection_array_msg = Detection2DArray()
            detection_array_msg.header = msg.header
            
            # Process detections and draw on image
            for detection in detections:
                # Draw bounding box
                pt1 = (int(detection['bbox'][0]), int(detection['bbox'][1]))
                pt2 = (int(detection['bbox'][0] + detection['bbox'][2]), 
                       int(detection['bbox'][1] + detection['bbox'][3]))
                cv2.rectangle(annotated_image, pt1, pt2, (0, 255, 0), 2)
                
                # Add label
                cv2.putText(annotated_image, 
                           detection['label'], 
                           (pt1[0], pt1[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 2)
                
                # Create detection message
                detection_msg = self.create_detection_msg(detection, msg.header)
                detection_array_msg.detections.append(detection_msg)
            
            # Publish detections
            self.detection_pub.publish(detection_array_msg)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def detect_objects(self, image):
        """Detect objects in the image (simplified example)"""
        # In a real Isaac ROS setup, you would use Isaac ROS perception packages
        # like Isaac ROS DetectNet for object detection
        
        # For this example, we'll simulate detection of colored objects
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (as an example)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': [x, y, w, h],
                    'label': 'red_object',
                    'confidence': 0.8
                })
        
        return detections

    def create_detection_msg(self, detection, header):
        """Create a Detection2D message from detection data"""
        from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
        
        detection_msg = Detection2D()
        detection_msg.header = header
        
        # Set bounding box
        detection_msg.bbox.center.x = detection['bbox'][0] + detection['bbox'][2] / 2
        detection_msg.bbox.center.y = detection['bbox'][1] + detection['bbox'][3] / 2
        detection_msg.bbox.size_x = detection['bbox'][2]
        detection_msg.bbox.size_y = detection['bbox'][3]
        
        # Set hypothesis
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = detection['label']
        hypothesis.hypothesis.score = detection['confidence']
        detection_msg.results.append(hypothesis)
        
        return detection_msg


def main(args=None):
    rclpy.init(args=args)
    
    processor = IsaacCameraProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()