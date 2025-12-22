---
id: chapter-2-nodes-topics-services
title: "Chapter 2: ROS 2 Nodes, Topics, and Services"
sidebar_label: "Chapter 2: ROS 2 Nodes, Topics, and Services"
description: "Understanding the core communication patterns in ROS 2: nodes, topics, and services"
keywords: [ros2, nodes, topics, services, communication, robotics]
tags: [ros, communication, patterns]
authors: [book-authors]
difficulty: intermediate
estimated_time: "60 minutes"
module: 1
chapter: 2
prerequisites: [python-basics, ros2-middleware-concepts]
learning_objectives:
  - Understand the architecture of ROS 2 nodes
  - Explain publisher-subscriber communication pattern using topics
  - Implement service-client communication for request-response patterns
  - Compare synchronous and asynchronous communication in robotics
related:
  - next: chapter-3-bridging-python-agents
  - previous: chapter-1-middleware-control
  - see_also: [chapter-1-middleware-control, chapter-3-bridging-python-agents, chapter-4-urdf-humanoids]
---

# Chapter 2: ROS 2 Nodes, Topics, and Services

## Learning Objectives

After completing this chapter, you will be able to:
- Design and implement ROS 2 nodes for different robot subsystems
- Create publisher-subscriber communication patterns for streaming data
- Implement service-client communication for request-response interactions
- Choose appropriate communication patterns for different robotic use cases
- Understand Quality of Service (QoS) settings and their impact on communication

## Introduction

In the previous chapter, we explored the middleware architecture of ROS 2 and its role as the communication layer for robotic systems. Now we'll dive deeper into the core communication patterns that make ROS 2 powerful: nodes, topics, and services. These building blocks form the foundation of how different components of a robotic system interact with each other.

Understanding these patterns is crucial for designing effective robotic applications, especially for complex systems like humanoid robots where multiple subsystems need to coordinate in real-time.

## ROS 2 Nodes

A node is the fundamental building block of a ROS 2 system. It's an executable process that performs specific tasks within the robot system. Nodes can be thought of as microservices in a distributed robotic application.

![ROS 2 Communication Patterns](/img/ros2-communication-patterns.svg)

### Node Architecture

Each ROS 2 node typically includes:

- **Node interface**: The core ROS 2 communication interface
- **Publishers**: For sending messages to topics
- **Subscribers**: For receiving messages from topics
- **Services**: For providing request-response functionality
- **Actions**: For long-running operations with feedback
- **Parameters**: For configuration management
- **Timers**: For periodic execution of functions

### Creating a Node in Python

Let's examine the structure of a ROS 2 node using the rclpy client library:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher_subscriber_node')

        # Create a publisher
        self.publisher = self.create_publisher(String, 'topic', 10)

        # Create a subscriber
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)

        # Create a timer to publish messages periodically
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_node = MinimalNode()

    try:
        rclpy.spin(minimal_node)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics and Publisher-Subscriber Pattern

Topics are the primary mechanism for asynchronous, one-way communication in ROS 2. They use a publish-subscribe pattern where publishers send messages to topics and subscribers receive messages from those topics without direct knowledge of each other.

### Topic Characteristics

- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **One-to-many**: A single publisher can have multiple subscribers
- **Many-to-one**: Multiple publishers can send to the same topic (should be avoided)
- **Decoupled**: Publishers and subscribers are independent of each other

### Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a custom QoS profile for sensor data
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Create a custom QoS profile for critical commands
command_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL,
    depth=1
)
```

### Publisher Implementation

Here's how to implement a publisher with appropriate QoS settings:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create QoS profile for joint state data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create publisher for joint states
        self.publisher = self.create_publisher(JointState, '/joint_states', qos_profile)

        # Timer to publish joint states periodically
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        msg.position = [0.1, 0.2, 0.3]  # Example positions
        msg.velocity = [0.0, 0.0, 0.0]  # Example velocities
        msg.effort = [0.0, 0.0, 0.0]    # Example efforts

        self.publisher.publish(msg)
        self.get_logger().debug('Published joint states')
```

### Subscriber Implementation

Here's how to implement a subscriber:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Create subscriber for joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10  # QoS depth
        )
        self.subscription  # Prevent unused variable warning

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Received joint states: {len(msg.name)} joints')
        for i, name in enumerate(msg.name):
            self.get_logger().debug(f'{name}: pos={msg.position[i]}, vel={msg.velocity[i]}')
```

## Services and Service-Client Pattern

Services provide synchronous, request-response communication between nodes. This pattern is useful when you need to get a response from an operation or when the operation needs to be acknowledged.

### Service Characteristics

- **Synchronous**: The client waits for a response from the service
- **Request-Response**: The client sends a request and receives a response
- **Stateful**: Services can maintain state between calls
- **Blocking**: The client is blocked until the response is received

### Creating a Service

First, let's define a custom service interface. Services use .srv files that define the request and response format. For example, a service to calculate robot forward kinematics:

```
# Request
geometry_msgs/PoseStamped pose
---
# Response
geometry_msgs/TransformStamped transform
```

### Service Server Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\n')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()

    try:
        response = minimal_client.send_request(1, 2)
        minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    except KeyboardInterrupt:
        pass
    finally:
        minimal_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Communication Patterns

### Actions

Actions are designed for long-running operations that require feedback and the ability to cancel. They're particularly useful for navigation, manipulation, and other complex robotic tasks:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback,
            cancel_callback=self.cancel_callback)

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

## Communication Pattern Selection Guidelines

Choosing the right communication pattern is crucial for robotic system design:

### Use Topics When:
- Streaming sensor data (camera images, laser scans, IMU readings)
- Broadcasting state information
- Real-time control commands
- When the publisher doesn't need acknowledgment

### Use Services When:
- Request-response interactions
- Operations that should return immediately
- Configuration or parameter changes
- When you need guaranteed delivery

### Use Actions When:
- Long-running operations
- Operations requiring feedback
- Operations that might be canceled
- Complex robotic tasks (navigation, manipulation)

## Practical Example: Humanoid Robot Communication Architecture

Consider a humanoid robot with multiple subsystems:

```python
# Joint controller node
class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Subscribe to desired joint positions (topics)
        self.joint_sub = self.create_subscription(
            JointState,
            '/desired_joint_positions',
            self.joint_command_callback,
            10)

        # Publish actual joint positions (topics)
        self.joint_pub = self.create_publisher(
            JointState,
            '/actual_joint_positions',
            10)

        # Service for immediate position setting
        self.pos_service = self.create_service(
            SetJointPosition,
            'set_joint_position',
            self.set_joint_position_callback)

# Walking controller node
class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Action for complex walking tasks
        self.walk_action_server = ActionServer(
            self,
            WalkAction,
            'walk_to_target',
            self.execute_walk_callback)

        # Subscribe to IMU data for balance feedback
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
```

### Complete Example: Service Server and Client

Here's a complete example of a service for controlling a humanoid robot's joint:

**Service Definition (custom_srv/srv/SetJointPosition.srv):**
```
# Request
string joint_name
float64 position
---
# Response
bool success
string message
```

**Service Server Implementation:**
```python
import rclpy
from rclpy.node import Node
from your_package.srv import SetJointPosition  # Custom service definition

class JointPositionController(Node):
    def __init__(self):
        super().__init__('joint_position_controller')

        # Create service
        self.srv = self.create_service(
            SetJointPosition,
            'set_joint_position',
            self.set_joint_position_callback
        )

        # Store current joint positions
        self.joint_positions = {
            'hip_joint': 0.0,
            'knee_joint': 0.0,
            'ankle_joint': 0.0
        }

        self.get_logger().info('Joint Position Controller initialized')

    def set_joint_position_callback(self, request, response):
        joint_name = request.joint_name
        position = request.position

        if joint_name in self.joint_positions:
            # In a real robot, this would send commands to the hardware
            self.joint_positions[joint_name] = position

            response.success = True
            response.message = f'Successfully set {joint_name} to {position}'

            self.get_logger().info(f'Set {joint_name} to {position}')
        else:
            response.success = False
            response.message = f'Joint {joint_name} not found'

            self.get_logger().warn(f'Invalid joint name: {joint_name}')

        return response

def main(args=None):
    rclpy.init(args=args)
    node = JointPositionController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down joint position controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Service Client Implementation:**
```python
import rclpy
from rclpy.node import Node
from your_package.srv import SetJointPosition  # Custom service definition

class JointPositionClient(Node):
    def __init__(self):
        super().__init__('joint_position_client')

        # Create client
        self.cli = self.create_client(SetJointPosition, 'set_joint_position')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, joint_name, position):
        request = SetJointPosition.Request()
        request.joint_name = joint_name
        request.position = position

        # Call service asynchronously
        self.future = self.cli.call_async(request)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    client = JointPositionClient()

    # Example: Set hip joint to 0.5 radians
    future = client.send_request('hip_joint', 0.5)

    try:
        rclpy.spin_until_future_complete(client, future)
        response = future.result()

        if response.success:
            client.get_logger().info(f'Service call successful: {response.message}')
        else:
            client.get_logger().error(f'Service call failed: {response.message}')

    except KeyboardInterrupt:
        client.get_logger().info('Service call interrupted')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Communication Design

### 1. Message Design
- Use appropriate data types (geometry_msgs for spatial data, sensor_msgs for sensors)
- Consider message size for bandwidth-constrained environments
- Use standard message types when possible for interoperability

### 2. Naming Conventions
- Use descriptive names that indicate the content and purpose
- Follow ROS conventions: `/namespace/subsystem/action`
- Use consistent naming across related topics

### 3. QoS Considerations
- Match QoS settings to the importance and timing requirements of data
- Use RELIABLE for critical commands, BEST_EFFORT for sensor data when appropriate
- Adjust history depth based on how much past data is needed

### 4. Error Handling
- Always implement proper error handling in services
- Monitor for dropped messages in topics
- Handle action cancellations gracefully

## Exercises

1. **Implementation Exercise**: Create a ROS 2 node that publishes temperature sensor data at 10 Hz using a custom message type. Create a separate node that subscribes to this topic and logs values above a threshold.

2. **Design Exercise**: Design a communication architecture for a humanoid robot's perception system with nodes for camera, LiDAR, and object detection. Specify which communication patterns to use and justify your choices.

3. **Analysis Task**: Compare the advantages and disadvantages of using topics vs services for controlling a humanoid robot's walking pattern. Consider factors like real-time performance, error handling, and system complexity.

4. **Practical Exercise**: Implement a service server that calculates the forward kinematics for a simple 2-DOF arm and a client that requests the end-effector position for specific joint angles. Test the service with various inputs and verify the results.

5. **Architecture Task**: Create a complete ROS 2 package that implements a humanoid robot's joint controller. The package should include:
   - A publisher for joint state commands
   - A service for setting joint positions immediately
   - An action server for executing complex joint trajectories
   - A test script that demonstrates all three communication patterns

6. **Performance Analysis**: Implement two versions of a data publisher - one with reliable QoS settings and one with best-effort settings. Compare their performance characteristics in terms of latency and throughput when publishing sensor data.

## Summary

This chapter covered the fundamental communication patterns in ROS 2: nodes, topics, and services. We learned that:

- Nodes are the basic building blocks of ROS 2 systems
- Topics enable asynchronous, one-way communication through publish-subscribe patterns
- Services provide synchronous, request-response communication
- QoS settings allow fine-tuning of communication behavior
- Actions are suitable for long-running operations with feedback
- Choosing the right communication pattern is crucial for system design

Understanding these patterns is essential for building distributed robotic systems, particularly for complex platforms like humanoid robots where multiple subsystems must coordinate effectively.

## Next Steps

In the next chapter, we'll explore how to bridge Python AI agents with ROS controllers using rclpy, building upon the communication patterns we've learned here. This will enable you to connect high-level AI systems with low-level robotic control systems.