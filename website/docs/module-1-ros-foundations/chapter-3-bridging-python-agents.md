---
id: chapter-3-bridging-python-agents
title: "Chapter 3: Bridging Python Agents to ROS Controllers using rclpy"
sidebar_label: "Chapter 3: Bridging Python Agents to ROS Controllers"
description: "Connecting AI agents written in Python to ROS 2 controllers for robotic systems"
keywords: [ros2, rclpy, python, ai, agents, robotics, control]
tags: [ros, ai, python, integration]
authors: [book-authors]
difficulty: advanced
estimated_time: "75 minutes"
module: 1
chapter: 3
prerequisites: [python-basics, ros2-nodes-topics-services, ai-fundamentals]
learning_objectives:
  - Understand the integration patterns between Python AI agents and ROS 2
  - Implement bridge nodes that connect AI logic to robotic controllers
  - Design message passing between AI decision-making and robot execution
  - Create robust communication patterns for real-time AI-robot interaction
related:
  - next: chapter-4-urdf-humanoids
  - previous: chapter-2-nodes-topics-services
  - see_also: [chapter-1-middleware-control, chapter-2-nodes-topics-services, chapter-4-urdf-humanoids]
---

# Chapter 3: Bridging Python Agents to ROS Controllers using rclpy

## Learning Objectives

After completing this chapter, you will be able to:
- Design integration patterns between Python AI agents and ROS 2 controllers
- Implement bridge nodes that connect high-level AI logic with low-level robot execution
- Handle real-time communication requirements between AI and robotic systems
- Create robust error handling and fallback mechanisms for AI-robot interaction
- Optimize message passing for performance and reliability

## Introduction

The integration of AI agents with robotic systems represents a critical convergence in modern robotics. This chapter explores how to effectively bridge Python-based AI agents with ROS 2 controllers, creating seamless interaction between high-level decision-making and low-level robot execution. This is particularly important for humanoid robotics, where sophisticated AI systems must coordinate with complex mechanical systems in real-time.

We'll cover the architectural patterns, implementation strategies, and best practices for connecting AI agents to robotic controllers using the rclpy client library.

## AI-Agent to ROS Integration Patterns

![Python AI Agents Bridged to ROS Controllers](/img/python-ros-bridge-diagram.svg)

When connecting AI agents to robotic systems, several architectural patterns emerge based on the nature of the interaction:

### 1. Command-and-Control Pattern
- AI agent makes high-level decisions
- ROS nodes execute low-level control commands
- Communication is primarily one-way from AI to robot

### 2. Perception-Action Loop
- AI agent processes sensor data from ROS topics
- AI agent publishes control commands to ROS topics
- Closed-loop system with continuous feedback

### 3. Hybrid Planning-Execution Pattern
- AI agent handles path planning and high-level decision making
- ROS nodes handle motion control and hardware interfaces
- Complex coordination between multiple systems

## Implementing AI-ROS Bridges

### Basic Bridge Node Structure

Here's a foundational pattern for an AI-ROS bridge:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Imu
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
import threading
import time

class AIBridgeNode(Node):
    def __init__(self):
        super().__init__('ai_bridge_node')

        # Create QoS profiles for different types of data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=5
        )
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )

        # Subscribers for sensor data
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            sensor_qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            sensor_qos
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            sensor_qos
        )

        # Publishers for AI decisions
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            command_qos
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            command_qos
        )

        # AI processing timer
        self.ai_timer = self.create_timer(0.1, self.ai_processing_callback)  # 10 Hz

        # Shared data structures for AI processing
        self.latest_joint_state = None
        self.latest_imu_data = None
        self.latest_camera_data = None

        self.get_logger().info('AI Bridge Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        self.latest_joint_state = msg
        # Process joint state in AI context if needed
        self.process_joint_state(msg)

    def imu_callback(self, msg):
        """Callback for IMU data"""
        self.latest_imu_data = msg
        # Process IMU data for balance/stability in AI context
        self.process_imu_data(msg)

    def camera_callback(self, msg):
        """Callback for camera data"""
        self.latest_camera_data = msg
        # Process camera data in AI context
        self.process_camera_data(msg)

    def ai_processing_callback(self):
        """Main AI processing loop"""
        # Gather current sensor data
        sensor_data = {
            'joint_state': self.latest_joint_state,
            'imu_data': self.latest_imu_data,
            'camera_data': self.latest_camera_data
        }

        # Run AI decision-making
        ai_decision = self.run_ai_decision(sensor_data)

        # Publish commands based on AI decision
        if ai_decision:
            self.publish_ai_commands(ai_decision)

    def process_joint_state(self, joint_state):
        """Process joint state data in AI context"""
        # AI-specific processing of joint states
        # This could include balance calculations, motion planning, etc.
        pass

    def process_imu_data(self, imu_data):
        """Process IMU data in AI context"""
        # AI-specific processing of IMU data
        # This could include stability analysis, fall detection, etc.
        pass

    def process_camera_data(self, camera_data):
        """Process camera data in AI context"""
        # AI-specific processing of camera data
        # This could include object detection, scene understanding, etc.
        pass

    def run_ai_decision(self, sensor_data):
        """Execute AI decision-making logic"""
        # This is where your AI agent would make decisions
        # based on the sensor data
        decision = {
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'goal_pose': None
        }
        return decision

    def publish_ai_commands(self, decision):
        """Publish AI decisions to ROS topics"""
        if 'linear_velocity' in decision and 'angular_velocity' in decision:
            twist_msg = Twist()
            twist_msg.linear.x = decision['linear_velocity']
            twist_msg.angular.z = decision['angular_velocity']
            self.cmd_vel_pub.publish(twist_msg)

        if 'goal_pose' in decision and decision['goal_pose']:
            pose_msg = PoseStamped()
            pose_msg.pose = decision['goal_pose']
            self.goal_pub.publish(pose_msg)
```

### Advanced Bridge with Action Integration

For more complex tasks requiring long-running operations, here's an enhanced bridge using actions:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AdvancedAIBridgeNode(Node):
    def __init__(self):
        super().__init__('advanced_ai_bridge_node')

        # Create action clients for complex operations
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory'
        )

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # AI processing timer
        self.ai_timer = self.create_timer(0.05, self.ai_processing_callback)

        # Thread pool for AI processing
        self.ai_executor = ThreadPoolExecutor(max_workers=2)

        # Shared data
        self.latest_joint_state = None
        self.ai_task = None

        self.get_logger().info('Advanced AI Bridge Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        self.latest_joint_state = msg

    async def run_ai_async(self):
        """Asynchronous AI processing"""
        while rclpy.ok():
            if self.latest_joint_state:
                # Perform AI computations
                ai_result = await self.ai_computation_step(self.latest_joint_state)
                if ai_result:
                    await self.execute_ai_decision(ai_result)
            await asyncio.sleep(0.05)  # 20 Hz processing

    async def ai_computation_step(self, joint_state):
        """Single step of AI computation"""
        # This would contain actual AI logic (ML models, planning algorithms, etc.)
        # For now, we'll simulate a decision
        decision = {
            'type': 'trajectory',
            'positions': [0.1, 0.2, 0.3],  # Example joint positions
            'velocities': [0.0, 0.0, 0.0],
            'time_from_start': 1.0
        }
        return decision

    async def execute_ai_decision(self, decision):
        """Execute AI decision via ROS actions"""
        if decision['type'] == 'trajectory':
            await self.send_trajectory_command(decision)

    async def send_trajectory_command(self, trajectory_data):
        """Send trajectory command via action"""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint1', 'joint2', 'joint3']

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = trajectory_data['positions']
        point.velocities = trajectory_data['velocities']
        point.time_from_start.sec = int(trajectory_data['time_from_start'])
        point.time_from_start.nanosec = int(
            (trajectory_data['time_from_start'] % 1) * 1e9
        )

        goal_msg.trajectory.points.append(point)

        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Trajectory action server not available')
            return

        # Send goal asynchronously
        future = self.trajectory_client.send_goal_async(goal_msg)
        result = await future
        return result

    def ai_processing_callback(self):
        """Timer callback to trigger AI processing"""
        # This callback can initiate AI processing
        pass
```

## Integration with Popular AI Frameworks

### Integration with OpenAI API

Here's how to integrate an AI agent using the OpenAI API for natural language processing:

```python
import rclpy
from rclpy.node import Node
import openai
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json

class OpenAIBridgeNode(Node):
    def __init__(self):
        super().__init__('openai_bridge_node')

        # Initialize OpenAI client
        openai.api_key = "YOUR_API_KEY"  # In practice, use environment variables

        # Publishers and subscribers
        self.command_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.voice_cmd_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )

        self.get_logger().info('OpenAI Bridge Node initialized')

    def voice_command_callback(self, msg):
        """Process voice command through AI"""
        try:
            # Use OpenAI to interpret the command
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a robot command interpreter. Convert natural language commands to robot actions. Respond with a JSON object containing linear and angular velocity values. Example: {\"linear\": 0.5, \"angular\": 0.0}"
                    },
                    {
                        "role": "user",
                        "content": msg.data
                    }
                ]
            )

            # Parse the AI response
            ai_response = response.choices[0].message.content
            command_data = json.loads(ai_response)

            # Create and publish robot command
            twist_cmd = Twist()
            twist_cmd.linear.x = command_data.get('linear', 0.0)
            twist_cmd.angular.z = command_data.get('angular', 0.0)

            self.command_pub.publish(twist_cmd)
            self.get_logger().info(f'Executed command: {command_data}')

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')
```

### Integration with TensorFlow/PyTorch Models

For on-device AI processing with machine learning models:

```python
import rclpy
from rclpy.node import Node
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class MLBridgeNode(Node):
    def __init__(self):
        super().__init__('ml_bridge_node')

        # Load pre-trained model
        self.model = tf.keras.models.load_model('path/to/your/model.h5')
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('ML Bridge Node initialized')

    def image_callback(self, msg):
        """Process image through ML model"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image for model
            processed_image = self.preprocess_image(cv_image)

            # Run inference
            prediction = self.model.predict(np.expand_dims(processed_image, axis=0))

            # Convert prediction to robot command
            twist_cmd = self.prediction_to_command(prediction)

            # Publish command
            self.cmd_vel_pub.publish(twist_cmd)

        except Exception as e:
            self.get_logger().error(f'Error in ML processing: {e}')

    def preprocess_image(self, image):
        """Preprocess image for ML model"""
        # Resize, normalize, etc.
        resized = cv2.resize(image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def prediction_to_command(self, prediction):
        """Convert ML prediction to robot command"""
        twist = Twist()

        # Example: prediction[0] contains [linear, angular] velocities
        twist.linear.x = float(prediction[0][0])
        twist.angular.z = float(prediction[0][1])

        return twist
```

## Real-Time Performance Considerations

### Threading and Concurrency

For real-time performance, consider using threading to separate AI processing from ROS communication:

```python
import rclpy
from rclpy.node import Node
import threading
import queue
import time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class ThreadedAIBridgeNode(Node):
    def __init__(self):
        super().__init__('threaded_ai_bridge_node')

        # Create queues for thread communication
        self.sensor_queue = queue.Queue(maxsize=10)
        self.command_queue = queue.Queue(maxsize=10)

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Start AI processing thread
        self.ai_thread = threading.Thread(target=self.ai_processing_thread)
        self.ai_thread.daemon = True
        self.ai_thread.start()

        # Timer for command publishing
        self.command_timer = self.create_timer(0.05, self.publish_commands)

        self.get_logger().info('Threaded AI Bridge Node initialized')

    def joint_state_callback(self, msg):
        """Add sensor data to queue for AI processing"""
        try:
            self.sensor_queue.put_nowait(msg)
        except queue.Full:
            # Drop old data if queue is full
            try:
                self.sensor_queue.get_nowait()
                self.sensor_queue.put_nowait(msg)
            except queue.Empty:
                pass

    def ai_processing_thread(self):
        """Dedicated thread for AI processing"""
        while rclpy.ok():
            try:
                # Get sensor data
                sensor_data = self.sensor_queue.get(timeout=0.1)

                # Process with AI
                ai_command = self.run_ai_logic(sensor_data)

                # Put command in output queue
                if ai_command:
                    try:
                        self.command_queue.put_nowait(ai_command)
                    except queue.Full:
                        # Update with newer command
                        try:
                            self.command_queue.get_nowait()
                            self.command_queue.put_nowait(ai_command)
                        except queue.Empty:
                            pass
            except queue.Empty:
                continue  # No new sensor data, continue loop

    def run_ai_logic(self, sensor_data):
        """AI processing logic (runs in separate thread)"""
        # This is where your AI algorithm would run
        # It should return a command for the robot
        command = {
            'linear': 0.0,
            'angular': 0.0,
            'timestamp': time.time()
        }
        return command

    def publish_commands(self):
        """Publish commands from AI thread to ROS"""
        try:
            while True:  # Process all available commands
                command = self.command_queue.get_nowait()

                # Convert to ROS message and publish
                twist_msg = Twist()
                twist_msg.linear.x = command['linear']
                twist_msg.angular.z = command['angular']

                self.cmd_vel_pub.publish(twist_msg)
        except queue.Empty:
            pass  # No commands to publish
```

## Error Handling and Fallback Strategies

### Graceful Degradation

Implement fallback strategies when AI systems fail:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time

class RobustAIBridgeNode(Node):
    def __init__(self):
        super().__init__('robust_ai_bridge_node')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.ai_status_pub = self.create_publisher(Bool, '/ai_status', 10)

        # AI processing timer
        self.ai_timer = self.create_timer(0.1, self.ai_processing_callback)

        # Fallback timer (in case AI fails)
        self.fallback_timer = self.create_timer(0.5, self.fallback_callback)

        # State tracking
        self.latest_joint_state = None
        self.ai_failure_count = 0
        self.max_failures = 5
        self.ai_active = True
        self.last_ai_success = time.time()

        self.get_logger().info('Robust AI Bridge Node initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        self.latest_joint_state = msg

    def ai_processing_callback(self):
        """Main AI processing with error handling"""
        try:
            if self.latest_joint_state:
                # Attempt AI processing
                ai_command = self.safe_ai_processing(self.latest_joint_state)

                if ai_command is not None:
                    # AI succeeded
                    self.ai_failure_count = 0
                    self.ai_active = True
                    self.last_ai_success = time.time()

                    # Publish command
                    self.cmd_vel_pub.publish(ai_command)

                    # Publish AI status
                    status_msg = Bool()
                    status_msg.data = True
                    self.ai_status_pub.publish(status_msg)
                else:
                    # AI returned no command
                    self.handle_ai_no_output()

        except Exception as e:
            # AI processing failed
            self.ai_failure_count += 1
            self.get_logger().error(f'AI processing failed: {e}')

            if self.ai_failure_count >= self.max_failures:
                self.ai_active = False
                self.get_logger().warn('AI system deactivated due to repeated failures')

                # Publish AI status
                status_msg = Bool()
                status_msg.data = False
                self.ai_status_pub.publish(status_msg)

    def safe_ai_processing(self, sensor_data):
        """AI processing with error handling"""
        try:
            # Your AI logic here with proper error handling
            # This is a simplified example
            if sensor_data.position:  # If we have valid sensor data
                twist = Twist()
                # Process sensor data to generate command
                # This would be your actual AI logic
                twist.linear.x = 0.1  # Example command
                return twist
            return None
        except Exception as e:
            self.get_logger().error(f'AI logic error: {e}')
            return None

    def handle_ai_no_output(self):
        """Handle case when AI returns no output"""
        # Implement strategy for handling no AI output
        # Could be fallback behavior or waiting
        pass

    def fallback_callback(self):
        """Fallback behavior when AI fails"""
        if not self.ai_active:
            # Implement safe fallback behavior
            # For example, stop the robot or move to safe state
            twist = Twist()
            # Stop robot
            self.cmd_vel_pub.publish(twist)
```

## Practical Example: AI-Guided Humanoid Robot Navigation

Here's a complete example combining all concepts for a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import threading
import queue
import time

class HumanoidAIBridgeNode(Node):
    def __init__(self):
        super().__init__('humanoid_ai_bridge_node')

        # QoS profiles
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', cmd_qos)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', cmd_qos)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, sensor_qos)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, sensor_qos)
        self.voice_cmd_sub = self.create_subscription(String, '/voice_command', self.voice_command_callback, 10)

        # Internal state
        self.current_pose = None
        self.joint_positions = {}
        self.imu_data = None
        self.laser_data = None
        self.voice_command = None

        # Threading for AI processing
        self.ai_queue = queue.Queue(maxsize=5)
        self.ai_thread = threading.Thread(target=self.ai_processing_loop)
        self.ai_thread.daemon = True
        self.ai_thread.start()

        # Timers
        self.ai_timer = self.create_timer(0.2, self.ai_trigger_callback)  # 5 Hz AI processing
        self.safety_timer = self.create_timer(0.05, self.safety_callback)  # 20 Hz safety checks

        self.get_logger().info('Humanoid AI Bridge Node initialized')

    def odom_callback(self, msg):
        """Handle odometry data"""
        self.current_pose = msg.pose.pose

    def joint_state_callback(self, msg):
        """Handle joint state data"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def voice_command_callback(self, msg):
        """Handle voice commands"""
        self.voice_command = msg.data

    def ai_trigger_callback(self):
        """Trigger AI processing"""
        # Gather current sensor data
        sensor_data = {
            'pose': self.current_pose,
            'joints': self.joint_positions.copy(),
            'imu': self.imu_data,
            'laser': self.laser_data,
            'voice': self.voice_command
        }

        # Put data in queue for AI processing
        try:
            self.ai_queue.put_nowait(sensor_data)
        except queue.Full:
            # Replace with newest data
            try:
                self.ai_queue.get_nowait()
                self.ai_queue.put_nowait(sensor_data)
            except queue.Empty:
                pass

    def ai_processing_loop(self):
        """AI processing in separate thread"""
        while rclpy.ok():
            try:
                # Get sensor data
                sensor_data = self.ai_queue.get(timeout=0.1)

                # Run AI decision making
                ai_decision = self.humanoid_ai_logic(sensor_data)

                if ai_decision:
                    # Publish decision
                    self.publish_ai_decision(ai_decision)

            except queue.Empty:
                continue

    def humanoid_ai_logic(self, sensor_data):
        """Main AI logic for humanoid robot"""
        if not sensor_data['pose'] or not sensor_data['laser']:
            return None

        # Example: Simple navigation AI
        command = {
            'linear': 0.0,
            'angular': 0.0,
            'goal': None
        }

        # Simple obstacle avoidance
        laser = sensor_data['laser']
        if len(laser.ranges) > 0:
            min_distance = min(laser.ranges)
            if min_distance < 0.5:  # Too close to obstacle
                command['angular'] = 0.5  # Turn away
            else:
                command['linear'] = 0.2  # Move forward

        # If voice command received, process it
        if sensor_data['voice']:
            if 'stop' in sensor_data['voice'].lower():
                command['linear'] = 0.0
                command['angular'] = 0.0
            elif 'forward' in sensor_data['voice'].lower():
                command['linear'] = 0.3
            elif 'turn' in sensor_data['voice'].lower():
                command['angular'] = 0.4

            # Clear voice command after processing
            self.voice_command = None

        return command

    def publish_ai_decision(self, decision):
        """Publish AI decisions to ROS"""
        if 'linear' in decision and 'angular' in decision:
            twist = Twist()
            twist.linear.x = decision['linear']
            twist.angular.z = decision['angular']
            self.cmd_vel_pub.publish(twist)

    def safety_callback(self):
        """Safety checks to prevent dangerous behavior"""
        # Check IMU for stability
        if self.imu_data:
            # Check if robot is tilted too much
            roll = self.get_roll_from_quaternion(self.imu_data.orientation)
            pitch = self.get_pitch_from_quaternion(self.imu_data.orientation)

            # If tilted beyond safe limits, emergency stop
            if abs(roll) > 0.5 or abs(pitch) > 0.5:  # 0.5 radians ≈ 28.6 degrees
                emergency_stop = Twist()
                self.cmd_vel_pub.publish(emergency_stop)
                self.get_logger().warn('Emergency stop: robot tilt exceeded safe limits')

    def get_roll_from_quaternion(self, q):
        """Extract roll from quaternion"""
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        return np.arctan2(sinr_cosp, cosr_cosp)

    def get_pitch_from_quaternion(self, q):
        """Extract pitch from quaternion"""
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            return np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
        return np.arcsin(sinp)

def main(args=None):
    rclpy.init(args=args)
    ai_bridge_node = HumanoidAIBridgeNode()

    try:
        rclpy.spin(ai_bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        ai_bridge_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example: Advanced AI Integration with Reinforcement Learning

Here's an example of how to integrate a reinforcement learning agent with ROS 2:

```python
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import torch
import torch.nn as nn

class RLDemoNode(Node):
    def __init__(self):
        super().__init__('rl_demo_node')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.reward_pub = self.create_publisher(Float32, '/rl_reward', 10)

        # RL components
        self.q_network = self.create_simple_q_network()
        self.latest_scan = None
        self.previous_distance = 0

        # RL parameters
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

        # Timer for RL decisions
        self.rl_timer = self.create_timer(0.2, self.rl_decision_callback)

        self.get_logger().info('RL Demo Node initialized')

    def create_simple_q_network(self):
        """Create a simple neural network for Q-learning"""
        class QNetwork(nn.Module):
            def __init__(self):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(10, 64)  # 10 laser readings as input
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 4)  # 4 possible actions

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return QNetwork()

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Take 10 equidistant readings from the front 180 degrees
        step = len(msg.ranges) // 10
        front_readings = [msg.ranges[i] for i in range(0, len(msg.ranges), step) if i < len(msg.ranges)]
        front_readings = [min(d, 10.0) for d in front_readings]  # Cap at 10m

        self.latest_scan = np.array(front_readings, dtype=np.float32)

    def rl_decision_callback(self):
        """Make RL-based navigation decision"""
        if self.latest_scan is not None:
            # Convert to tensor for neural network
            state_tensor = torch.FloatTensor(self.latest_scan)

            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Explore: random action
                action = np.random.choice(4)
            else:
                # Exploit: best action according to network
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action = torch.argmax(q_values).item()

            # Convert action to velocity command
            twist = Twist()
            if action == 0:  # Move forward
                twist.linear.x = 0.3
                twist.angular.z = 0.0
            elif action == 1:  # Turn left
                twist.linear.x = 0.1
                twist.angular.z = 0.5
            elif action == 2:  # Turn right
                twist.linear.x = 0.1
                twist.angular.z = -0.5
            else:  # Stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            # Calculate reward based on distance to obstacles
            min_distance = min(self.latest_scan) if len(self.latest_scan) > 0 else 0
            reward = self.calculate_reward(min_distance)

            # Publish reward for monitoring
            reward_msg = Float32()
            reward_msg.data = reward
            self.reward_pub.publish(reward_msg)

            # Publish velocity command
            self.cmd_vel_pub.publish(twist)

            # Update previous distance for next reward calculation
            self.previous_distance = min_distance

    def calculate_reward(self, current_distance):
        """Calculate reward based on distance to obstacles"""
        # Reward for moving away from obstacles
        distance_reward = max(0, current_distance - self.previous_distance) * 10

        # Penalty for being too close to obstacles
        proximity_penalty = 0
        if current_distance < 0.5:
            proximity_penalty = -10
        elif current_distance < 1.0:
            proximity_penalty = -5

        # Small reward for maintaining safe distance
        safety_reward = 1 if current_distance > 1.0 else 0

        total_reward = distance_reward + proximity_penalty + safety_reward
        return total_reward

def main(args=None):
    rclpy.init(args=args)
    node = RLDemoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down RL Demo Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for AI-ROS Integration

### 1. Message Optimization
- Use appropriate message types for your data
- Consider message size for bandwidth-constrained environments
- Use compressed formats for images when possible

### 2. Error Handling
- Implement graceful degradation when AI systems fail
- Include fallback behaviors for safety-critical applications
- Log AI decisions for debugging and analysis

### 3. Performance Considerations
- Separate AI processing from ROS communication threads
- Use appropriate QoS settings for different data types
- Consider using dedicated hardware for AI processing

### 4. Safety and Reliability
- Implement safety monitors for AI-decisions
- Include validation of AI outputs before execution
- Design for graceful failure modes

## Exercises

1. **Integration Exercise**: Create an AI-ROS bridge that connects a simple neural network for object detection to a mobile robot's navigation system. The AI should process camera images and command the robot to approach detected objects while avoiding obstacles.

2. **Architecture Design**: Design an AI-ROS bridge architecture for a humanoid robot that can handle both real-time control (balance, walking) and high-level decision making (task planning, natural language understanding). Consider the different timing requirements and communication patterns needed.

3. **Performance Analysis**: Implement two versions of an AI-ROS bridge - one using synchronous processing and another using threaded processing. Compare their performance in terms of latency and throughput for a simple navigation task.

4. **Practical Implementation**: Create a Python-based AI agent that uses OpenAI's API to interpret natural language commands and convert them to ROS 2 movement commands for a humanoid robot. The agent should handle commands like "walk forward", "turn left", "raise your left arm", etc.

5. **Safety and Fallback Exercise**: Enhance the AI-ROS bridge with safety mechanisms that monitor IMU data to detect when the robot is losing balance. If dangerous conditions are detected, the system should switch from AI control to a safe recovery behavior.

6. **Real-time Performance Task**: Implement an AI-ROS bridge that processes sensor data at 50 Hz while maintaining communication with ROS 2 topics and services. Use threading or asyncio to ensure that the AI processing doesn't block ROS communication.

## Summary

This chapter covered the critical aspects of bridging Python AI agents with ROS 2 controllers:

- Various integration patterns for connecting AI logic to robotic systems
- Implementation of bridge nodes with proper error handling and threading
- Integration with popular AI frameworks like OpenAI, TensorFlow, and PyTorch
- Real-time performance considerations and optimization techniques
- Safety mechanisms and fallback strategies for robust operation
- A complete example of an AI-guided humanoid robot navigation system

Successfully integrating AI agents with robotic systems requires careful consideration of timing, safety, and reliability. The patterns and techniques covered in this chapter provide a foundation for building robust AI-robot systems.

## Next Steps

In the next chapter, we'll explore URDF (Unified Robot Description Format) and how to describe humanoid robots for simulation and control in ROS 2. This will build upon the communication patterns and AI integration concepts we've learned in this module, allowing us to create complete robotic systems with proper robot descriptions.