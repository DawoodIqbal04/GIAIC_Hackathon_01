---
id: chapter-4-urdf-humanoids
title: "Chapter 4: Understanding URDF (Unified Robot Description Format) for Humanoids"
sidebar_label: "Chapter 4: Understanding URDF for Humanoids"
description: "Understanding the Unified Robot Description Format for describing humanoid robots in ROS 2"
keywords: [urdf, ros2, humanoid, robot description, robotics, xacro]
tags: [ros, robot-description, modeling]
authors: [book-authors]
difficulty: advanced
estimated_time: "90 minutes"
module: 1
chapter: 4
prerequisites: [python-basics, ros2-nodes-topics-services, xml-basics]
learning_objectives:
  - Understand the structure and components of URDF files
  - Create URDF descriptions for humanoid robot joints and links
  - Implement complex kinematic chains for humanoid robots
  - Use Xacro to simplify complex URDF definitions
  - Validate URDF files and visualize robot models
related:
  - next: module-2-intro
  - previous: chapter-3-bridging-python-agents
  - see_also: [chapter-1-middleware-control, chapter-2-nodes-topics-services, chapter-3-bridging-python-agents]
---

# Chapter 4: Understanding URDF (Unified Robot Description Format) for Humanoids

## Learning Objectives

After completing this chapter, you will be able to:
- Create complete URDF descriptions for humanoid robot models
- Understand the structure of links, joints, and transmissions in URDF
- Implement complex kinematic chains for humanoid robot limbs
- Use Xacro macros to simplify repetitive URDF definitions
- Validate and visualize URDF models in RViz and Gazebo
- Integrate URDF with ROS 2 control systems for humanoid robots

## Introduction

The Unified Robot Description Format (URDF) is a fundamental component of the ROS ecosystem that allows for the complete description of robot models. For humanoid robots, which have complex kinematic structures with multiple degrees of freedom, URDF becomes particularly important as it defines the physical and kinematic properties of the robot.

In this chapter, we'll explore how to create detailed URDF descriptions specifically tailored for humanoid robots, including proper joint definitions, collision geometries, and visualization properties. We'll also cover Xacro, an XML macro language that helps simplify complex URDF definitions.

![URDF - Unified Robot Description Format](/img/urdf-diagram.svg)

## URDF Fundamentals

URDF (Unified Robot Description Format) is an XML-based format used to describe robots in ROS. It defines the physical structure of a robot including:

- **Links**: Rigid parts of the robot (e.g., torso, limbs, head)
- **Joints**: Connections between links with specific degrees of freedom
- **Visual**: How the robot appears in simulation and visualization
- **Collision**: How the robot interacts with the environment in physics simulation
- **Inertial**: Mass properties for physics simulation

### Basic URDF Structure

A minimal URDF file contains at least one link and typically multiple joint-link pairs:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## Links in URDF

Links represent rigid bodies in the robot. Each link can have visual, collision, and inertial properties.

### Link Properties

1. **Visual**: Defines how the link appears in visualization tools
2. **Collision**: Defines how the link interacts in physics simulation
3. **Inertial**: Defines mass properties for dynamics calculations

```xml
<link name="link_name">
  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <mass value="0.1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.00083" ixy="0.0" ixz="0.0" iyy="0.00083" iyz="0.0" izz="0.00083"/>
  </inertial>
</link>
```

### Geometry Types

URDF supports several geometry types:

- **Box**: Rectangular prism with specified dimensions
- **Cylinder**: Cylinder with specified radius and length
- **Sphere**: Sphere with specified radius
- **Mesh**: Complex geometry loaded from external files (STL, DAE, etc.)

## Joints in URDF

Joints define the connections between links and specify the allowed motion between them.

### Joint Types

1. **Fixed**: No motion allowed (welded connection)
2. **Revolute**: Single-axis rotation with limits
3. **Continuous**: Single-axis rotation without limits
4. **Prismatic**: Single-axis translation with limits
5. **Planar**: Motion in a plane
6. **Floating**: 6-DOF motion

### Joint Definition

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## URDF for Humanoid Robots

Humanoid robots have complex kinematic structures with multiple limbs. Here's a comprehensive example of a humanoid torso and head:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <!-- Left shoulder -->
  <link name="left_shoulder">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <origin xyz="0.2 0 0.8"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50.0" velocity="3.0"/>
  </joint>

  <!-- Left upper arm -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 -0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" effort="40.0" velocity="4.0"/>
  </joint>

  <!-- Left lower arm -->
  <link name="left_lower_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="5.0"/>
  </joint>

  <!-- Left hand -->
  <link name="left_hand">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.08"/>
      </geometry>
    </visual>
  </link>

  <joint name="left_hand_joint" type="fixed">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.2"/>
  </joint>

</robot>
```

## Xacro for Complex Humanoid URDFs

Xacro (XML Macros) is an XML macro language that helps simplify complex URDF definitions. It's essential for humanoid robots due to their repetitive structures.

### Basic Xacro Concepts

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_mass" value="10.0"/>
  <xacro:property name="arm_mass" value="1.5"/>

  <!-- Macro for arm definition -->
  <xacro:macro name="arm" params="side parent xyz rpy">
    <link name="${side}_shoulder">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0"/>
        <geometry>
          <box size="0.1 0.1 0.1"/>
        </geometry>
      </visual>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="50.0" velocity="3.0"/>
    </joint>

    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${arm_mass}"/>
        <origin xyz="0 0 -0.15"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 -0.1"/>
      <axis xyz="0 0 1"/>
      <limit lower="-2.0" upper="2.0" effort="40.0" velocity="4.0"/>
    </joint>
  </xacro:macro>

  <!-- Base torso -->
  <link name="base_link">
    <inertial>
      <mass value="${torso_mass}"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Use the arm macro for both left and right arms -->
  <xacro:arm side="left" parent="base_link" xyz="0.2 0 0.8" rpy="0 0 0"/>
  <xacro:arm side="right" parent="base_link" xyz="-0.2 0 0.8" rpy="0 0 0"/>

</robot>
```

### Advanced Xacro Features for Humanoids

Here's a more comprehensive example with additional features:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid">

  <!-- Include other xacro files -->
  <xacro:include filename="$(find humanoid_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find humanoid_description)/urdf/properties.xacro"/>

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_height" value="1.0"/>
  <xacro:property name="torso_width" value="0.3"/>
  <xacro:property name="torso_depth" value="0.3"/>

  <!-- Macro for generic link with standard properties -->
  <xacro:macro name="generic_link" params="name mass xyz size color">
    <link name="${name}">
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="${xyz}"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="${xyz}"/>
        <geometry>
          <box size="${size}"/>
        </geometry>
        <material name="${color}"/>
      </visual>
      <collision>
        <origin xyz="${xyz}"/>
        <geometry>
          <box size="${size}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Macro for leg with 6 DOF -->
  <xacro:macro name="leg" params="side parent xyz">
    <!-- Hip link -->
    <link name="${side}_hip">
      <inertial>
        <mass value="2.0"/>
        <origin xyz="0 0 -0.1"/>
        <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.1"/>
        <geometry>
          <cylinder radius="0.06" length="0.2"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 -0.1"/>
        <geometry>
          <cylinder radius="0.06" length="0.2"/>
        </geometry>
      </collision>
    </link>

    <!-- Hip joint (3 DOF) -->
    <joint name="${side}_hip_yaw_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_hip"/>
      <origin xyz="${xyz}"/>
      <axis xyz="0 0 1"/>
      <limit lower="-0.5" upper="0.5" effort="100.0" velocity="2.0"/>
    </joint>

    <joint name="${side}_hip_roll_joint" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_hip"/>
      <origin xyz="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.3" upper="0.3" effort="100.0" velocity="2.0"/>
    </joint>

    <joint name="${side}_hip_pitch_joint" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_hip"/>
      <origin xyz="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.0" upper="0.5" effort="100.0" velocity="2.0"/>
    </joint>

    <!-- Thigh -->
    <link name="${side}_thigh">
      <inertial>
        <mass value="3.0"/>
        <origin xyz="0 0 -0.2"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.07" length="0.4"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.07" length="0.4"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_thigh_joint" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 0 -0.2"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="2.0" effort="150.0" velocity="2.0"/>
    </joint>

    <!-- Shin -->
    <link name="${side}_shin">
      <inertial>
        <mass value="2.5"/>
        <origin xyz="0 0 -0.2"/>
        <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.015"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.06" length="0.4"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.06" length="0.4"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_shin_joint" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 -0.4"/>
      <axis xyz="0 1 0"/>
      <limit lower="0.0" upper="2.0" effort="120.0" velocity="2.0"/>
    </joint>

    <!-- Foot -->
    <link name="${side}_foot">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0.1 0 -0.05"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0.1 0 -0.05"/>
        <geometry>
          <box size="0.25 0.1 0.1"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0.1 0 -0.05"/>
        <geometry>
          <box size="0.25 0.1 0.1"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_foot_joint" type="revolute">
      <parent link="${side}_shin"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -0.4"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="0.5" effort="50.0" velocity="2.0"/>
    </joint>
  </xacro:macro>

  <!-- Main torso -->
  <link name="base_link">
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 ${torso_height/2}"/>
      <inertia ixx="1.5" ixy="0.0" ixz="0.0" iyy="1.5" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 ${torso_height/2}"/>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${torso_height/2}"/>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin_color">
        <color rgba="0.8 0.6 0.4 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 ${torso_height}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="2.0"/>
  </joint>

  <!-- Use leg macros -->
  <xacro:leg side="left" parent="base_link" xyz="${torso_width/4} 0 0"/>
  <xacro:leg side="right" parent="base_link" xyz="${-torso_width/4} 0 0"/>

  <!-- Use arm macros (defined elsewhere) -->
  <xacro:include filename="$(find humanoid_description)/urdf/arms.xacro"/>
  <xacro:arms parent="base_link"/>

</robot>
```

## URDF Validation and Visualization

### Validating URDF Files

Before using a URDF file, it's important to validate it:

```bash
# Check if URDF is well-formed
check_urdf /path/to/robot.urdf

# Check for joint limits and other issues
urdf_to_graphiz /path/to/robot.urdf
```

### Visualizing URDF in RViz

To visualize your URDF model in RViz:

1. Launch your robot state publisher:
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)
```

2. Launch RViz:
```bash
ros2 run rviz2 rviz2
```

3. Add a RobotModel display and set the Robot Description parameter to your robot description topic.

### Testing with Joint State Publisher

For testing and visualization, use the joint state publisher:

```bash
# Launch with GUI to manually control joints
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

## Integration with ROS 2 Control Systems

For humanoid robots to be functional, URDF must be integrated with ROS 2 control systems.

### Transmission Definitions

Transmissions define how actuators connect to joints:

```xml
<transmission name="left_elbow_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_elbow_joint">
    <hardwareInterface>position_controllers/JointPositionController</hardwareInterface>
  </joint>
  <actuator name="left_elbow_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Control Configuration

Create a control configuration file (e.g., `config/robot_control.yaml`):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_arm_controller:
      type: position_controllers/JointTrajectoryController

    right_arm_controller:
      type: position_controllers/JointTrajectoryController

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_joint
      - left_elbow_joint
      - left_wrist_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_joint
      - right_elbow_joint
      - right_wrist_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
```

## Best Practices for Humanoid URDF

### 1. Proper Mass and Inertia Values

Accurate mass and inertia properties are crucial for physics simulation:

```xml
<inertial>
  <mass value="1.5"/>
  <origin xyz="0 0 -0.15"/>
  <!-- Calculate inertia tensor properly for the geometry -->
  <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
</inertial>
```

### 2. Realistic Joint Limits

Set appropriate joint limits based on the physical capabilities of the robot:

```xml
<joint name="hip_pitch_joint" type="revolute">
  <parent link="torso"/>
  <child link="thigh"/>
  <origin xyz="0 0 0"/>
  <axis xyz="0 1 0"/>
  <!-- Realistic limits for a humanoid hip joint -->
  <limit lower="-0.5" upper="2.0" effort="150.0" velocity="2.0"/>
</joint>
```

### 3. Collision Geometry Optimization

Use simplified collision geometries for better performance:

```xml
<!-- Use simple shapes for collision, detailed meshes for visualization -->
<collision>
  <geometry>
    <cylinder radius="0.05" length="0.3"/>
  </geometry>
</collision>
<visual>
  <geometry>
    <mesh filename="package://humanoid_description/meshes/detailed_arm.dae"/>
  </geometry>
</visual>
```

### 4. Consistent Naming Conventions

Use consistent naming for joints and links:

```xml
<!-- Good naming convention -->
<joint name="left_shoulder_pitch_joint" type="revolute">
  <parent link="torso"/>
  <child link="left_shoulder"/>
</joint>
```

### 5. Use Xacro for Complex Models

For humanoid robots with many similar components, use Xacro macros:

```xml
<!-- Define a finger macro instead of repeating for each finger -->
<xacro:macro name="finger" params="name parent side segment_length">
  <!-- Finger definition -->
</xacro:macro>

<!-- Use for all fingers -->
<xacro:finger name="thumb" parent="hand" side="left" segment_length="0.03"/>
<xacro:finger name="index" parent="hand" side="left" segment_length="0.04"/>
<!-- etc. -->
```

## Troubleshooting Common URDF Issues

### 1. Invalid URDF Structure

Common issues and solutions:

- **Floating joints**: Every link except the base should have exactly one parent joint
- **Cycles in kinematic tree**: URDF must be a tree structure, not a graph with cycles
- **Missing properties**: Each link must have inertial properties for physics simulation

### 2. Visualization Problems

- **Mesh files not found**: Ensure mesh files are in the correct package location
- **Wrong coordinate frames**: Check that origins and axes are correctly defined
- **Material issues**: Verify that materials are properly defined and referenced

### 3. Physics Simulation Issues

- **Unstable simulation**: Check mass and inertia values
- **Joints not moving**: Verify joint limits and transmission definitions
- **Collisions not working**: Check collision geometries and link connections

## Practical Example: Complete Humanoid URDF

Here's a complete example of a simplified humanoid robot using Xacro:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_humanoid">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_mass" value="15.0"/>
  <xacro:property name="head_mass" value="2.0"/>
  <xacro:property name="arm_mass" value="1.5"/>
  <xacro:property name="leg_mass" value="3.0"/>

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.4235 0.0431 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.8706 0.8118 0.7647 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base torso -->
  <link name="base_link">
    <inertial>
      <mass value="${torso_mass}"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.5" ixy="0.0" ixz="0.0" iyy="1.5" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="${head_mass}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="2.0"/>
  </joint>

  <!-- Macro for arm -->
  <xacro:macro name="arm" params="side parent xyz">
    <link name="${side}_shoulder">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0 0 0"/>
        <geometry>
          <box size="0.1 0.1 0.1"/>
        </geometry>
        <material name="blue"/>
      </visual>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${xyz}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="50.0" velocity="3.0"/>
    </joint>

    <link name="${side}_upper_arm">
      <inertial>
        <mass value="${arm_mass}"/>
        <origin xyz="0 0 -0.15"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.15"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <material name="orange"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15"/>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 -0.1"/>
      <axis xyz="0 0 1"/>
      <limit lower="-2.0" upper="2.0" effort="40.0" velocity="4.0"/>
    </joint>

    <link name="${side}_lower_arm">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 -0.1"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.003"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.1"/>
        <geometry>
          <cylinder radius="0.04" length="0.2"/>
        </geometry>
        <material name="orange"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.1"/>
        <geometry>
          <cylinder radius="0.04" length="0.2"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_wrist_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -0.3"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="20.0" velocity="5.0"/>
    </joint>
  </xacro:macro>

  <!-- Macro for leg -->
  <xacro:macro name="leg" params="side parent xyz">
    <link name="${side}_hip">
      <inertial>
        <mass value="2.0"/>
        <origin xyz="0 0 -0.1"/>
        <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.1"/>
        <geometry>
          <cylinder radius="0.06" length="0.2"/>
        </geometry>
        <material name="green"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.1"/>
        <geometry>
          <cylinder radius="0.06" length="0.2"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_hip_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_hip"/>
      <origin xyz="${xyz}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="0.5" effort="100.0" velocity="2.0"/>
    </joint>

    <link name="${side}_thigh">
      <inertial>
        <mass value="${leg_mass}"/>
        <origin xyz="0 0 -0.2"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.02"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.07" length="0.4"/>
        </geometry>
        <material name="green"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.07" length="0.4"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 0 -0.2"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.1" upper="2.0" effort="150.0" velocity="2.0"/>
    </joint>

    <link name="${side}_shin">
      <inertial>
        <mass value="2.5"/>
        <origin xyz="0 0 -0.2"/>
        <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.015"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.06" length="0.4"/>
        </geometry>
        <material name="green"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.2"/>
        <geometry>
          <cylinder radius="0.06" length="0.4"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_ankle_joint" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 -0.4"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="0.5" effort="50.0" velocity="2.0"/>
    </joint>

    <link name="${side}_foot">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0.1 0 -0.05"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <origin xyz="0.1 0 -0.05"/>
        <geometry>
          <box size="0.25 0.1 0.1"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <origin xyz="0.1 0 -0.05"/>
        <geometry>
          <box size="0.25 0.1 0.1"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_foot_joint" type="revolute">
      <parent link="${side}_shin"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -0.4"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.5" upper="0.5" effort="50.0" velocity="2.0"/>
    </joint>
  </xacro:macro>

  <!-- Create arms and legs -->
  <xacro:arm side="left" parent="base_link" xyz="0.2 0 0.8"/>
  <xacro:arm side="right" parent="base_link" xyz="-0.2 0 0.8"/>
  <xacro:leg side="left" parent="base_link" xyz="0.1 0 0"/>
  <xacro:leg side="right" parent="base_link" xyz="-0.1 0 0"/>

  <!-- ROS 2 Control -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="left_shoulder_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_elbow_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_wrist_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_shoulder_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_elbow_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_wrist_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="neck_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

</robot>
```

## Exercises

1. **URDF Creation Exercise**: Create a URDF file for a simplified humanoid robot with a torso, head, two arms, and two legs. Use Xacro macros to avoid repetition in the arm and leg definitions.

2. **Kinematic Chain Analysis**: For the humanoid model you created, analyze the kinematic chains for reaching and walking. Identify which joints need to be coordinated for specific movements.

3. **Xacro Optimization**: Take an existing complex URDF file and refactor it using Xacro macros to make it more maintainable and readable.

4. **Practical URDF Exercise**: Create a complete URDF model of an Atlas or similar humanoid robot with at least 28 degrees of freedom. Include realistic dimensions, masses, and inertial properties based on published specifications.

5. **Gazebo Simulation**: Integrate your humanoid URDF with Gazebo physics simulation by adding appropriate material properties, friction coefficients, and transmission elements for joint control.

6. **Validation and Visualization**: Validate your URDF model using ROS tools like `check_urdf` and visualize it in RViz. Ensure there are no errors in the kinematic chain and that all joints move as expected.

7. **Advanced URDF Features**: Enhance your humanoid URDF with transmission elements for simulated joint control, joint limits based on human-like ranges of motion, and realistic inertial properties calculated from geometric approximations.

## Summary

This chapter covered the essential aspects of URDF for humanoid robots:

- URDF structure with links, joints, visual, collision, and inertial properties
- Joint types and their appropriate use in humanoid robots
- Xacro macros for simplifying complex humanoid URDF definitions
- Integration with ROS 2 control systems
- Best practices for creating realistic humanoid models
- Validation and visualization techniques

Understanding URDF is crucial for humanoid robotics as it provides the foundation for simulation, visualization, and control of complex robotic systems. Properly defined URDF models enable accurate physics simulation, collision detection, and integration with ROS 2 control frameworks.

## Next Steps

With Module 1 complete, you now have a comprehensive understanding of ROS 2 fundamentals for humanoid robotics:
- Middleware architecture and communication patterns
- Node implementation and communication
- AI-robot integration techniques
- Robot description and modeling with URDF

In the next module, we'll explore digital twins using Gazebo and Unity, building upon these foundational concepts to create simulation environments for humanoid robots.