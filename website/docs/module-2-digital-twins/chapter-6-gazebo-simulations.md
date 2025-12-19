---
id: chapter-6-gazebo-simulations
title: "Chapter 6: Simulating Physics, Gravity, and Collisions in Gazebo"
sidebar_label: "Chapter 6: Simulating Physics, Gravity, and Collisions in Gazebo"
description: "Advanced physics simulation techniques in Gazebo for humanoid robotics applications"
keywords: [gazebo, physics, simulation, collisions, gravity, robotics, humanoid]
tags: [gazebo, physics, simulation]
authors: [book-authors]
difficulty: advanced
estimated_time: "90 minutes"
module: 2
chapter: 6
prerequisites: [physics-basics, ros2-foundations, urdf-basics, physics-simulation]
learning_objectives:
  - Configure Gazebo physics engines for humanoid robot simulation
  - Implement realistic gravity and environmental physics
  - Design advanced collision detection and response systems
  - Optimize simulation performance for complex humanoid models
  - Integrate Gazebo with ROS 2 for seamless robot simulation
related:
  - next: chapter-7-unity-interaction
  - previous: chapter-5-physics-simulation
  - see_also: [chapter-5-physics-simulation, chapter-7-unity-interaction, ../module-1-ros-foundations/chapter-4-urdf-humanoids]
---

# Chapter 6: Simulating Physics, Gravity, and Collisions in Gazebo

## Learning Objectives

After completing this chapter, you will be able to:
- Configure and optimize Gazebo physics engines for humanoid robot applications
- Implement realistic gravity, friction, and collision properties
- Design advanced collision detection and response systems
- Optimize simulation performance for complex humanoid models
- Integrate Gazebo with ROS 2 for seamless robot simulation workflows

## Introduction

Gazebo stands as one of the most powerful and widely-used simulation environments in robotics, particularly for humanoid robot development. Its robust physics engine, realistic collision detection, and deep integration with ROS make it an ideal platform for testing and validating humanoid robot behaviors before deployment on physical hardware.

This chapter delves deep into Gazebo's physics capabilities, focusing specifically on how to configure and optimize physics simulation for the unique requirements of humanoid robots. We'll explore how to model realistic physical interactions, configure gravity and environmental conditions, and optimize performance for complex multi-degree-of-freedom systems.

## Gazebo Physics Architecture

### Overview of Gazebo's Physics System

Gazebo's physics system is built around a plugin architecture that allows for different physics engines to be used interchangeably. The core components include:

1. **Physics Engine Plugins**: ODE, Bullet, or DART implementations
2. **Collision Detection System**: Broad-phase and narrow-phase collision detection
3. **Contact Manager**: Handles collision responses and contact forces
4. **Constraint Solver**: Resolves joint constraints and contact forces
5. **World Update Loop**: Integrates physics over time steps

### Physics Engine Comparison in Gazebo

| Engine | Strengths | Weaknesses | Best Use Cases |
|--------|-----------|------------|----------------|
| ODE | Fast, stable, well-tested | Less accurate contacts | Simple robots, fast simulation |
| Bullet | Good balance, excellent collision detection | Complex contact modeling | General robotics, humanoid robots |
| DART | Advanced contact modeling, stable | Newer, less community support | Complex dynamics, humanoid robots |

## Configuring Gazebo Physics for Humanoid Robots

### Physics Engine Selection

For humanoid robots, the choice of physics engine can significantly impact simulation quality:

```xml
<!-- Example SDF configuration with detailed physics settings -->
<world name="humanoid_world">
  <physics name="humanoid_physics" type="dart">  <!-- DART for advanced contact modeling -->
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>

    <dart>
      <solver>
        <type>PGS</type>  <!-- Projected Gauss-Seidel solver -->
        <iters>50</iters>  <!-- More iterations for stability -->
        <sor>1.3</sor>    <!-- Successive Over-Relaxation parameter -->
      </solver>

      <constraints>
        <contact_surface_layer>0.001</contact_surface_layer>
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
        <cfm>1e-10</cfm>  <!-- Constraint Force Mixing -->
        <erp>0.2</erp>    <!-- Error Reduction Parameter -->
      </constraints>
    </dart>
  </physics>
</world>
```

### Gravity Configuration

Humanoid robots require precise gravity simulation to test balance and locomotion:

```xml
<!-- Gravity configuration -->
<world name="humanoid_world">
  <!-- Standard Earth gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- Or for different gravity environments -->
  <gravity>0 0 -1.62</gravity>  <!-- Moon gravity -->
  <gravity>0 0 -3.72</gravity>  <!-- Mars gravity -->
</world>
```

### Time Step Optimization

For humanoid robots with many joints and complex dynamics, careful time step selection is crucial:

```xml
<physics name="humanoid_physics" type="dart">
  <!-- Smaller time steps for stability with complex humanoid models -->
  <max_step_size>0.0005</max_step_size>  <!-- 500 microseconds -->

  <!-- Update rate affects simulation speed -->
  <real_time_update_rate>2000</real_time_update_rate>

  <!-- Real-time factor (1.0 = real-time, >1.0 = faster than real-time) -->
  <real_time_factor>1.0</real_time_factor>
</physics>
```

## Advanced Collision Detection for Humanoid Robots

### Collision Geometry Selection

Humanoid robots require careful collision geometry selection to balance accuracy with performance:

```xml
<!-- Example of different collision geometries for humanoid robot parts -->
<link name="torso">
  <collision name="torso_collision">
    <geometry>
      <!-- Use box for simple torso collision -->
      <box>
        <size>0.3 0.25 0.5</size>
      </box>
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.001</soft_cfm>
          <soft_erp>0.8</soft_erp>
          <kp>1e5</kp>  <!-- Contact stiffness -->
          <kd>1.0</kd>  <!-- Damping coefficient -->
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.7</mu>    <!-- Static friction -->
          <mu2>0.5</mu2>  <!-- Dynamic friction -->
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
        </ode>
      </friction>
    </surface>
  </collision>
</link>

<link name="foot">
  <collision name="foot_collision">
    <geometry>
      <!-- Use box for foot to prevent interpenetration -->
      <box>
        <size>0.25 0.1 0.05</size>
      </box>
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.0001</soft_cfm>  <!-- Very stiff for stable contact -->
          <soft_erp>0.9</soft_erp>
          <kp>1e6</kp>
          <kd>10.0</kd>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.8</mu>   <!-- Higher friction for better grip -->
          <mu2>0.7</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
</link>
```

### Collision Filtering

For humanoid robots with many links, collision filtering can improve performance:

```xml
<!-- Example of collision filtering -->
<link name="left_upper_arm">
  <collision name="upper_arm_collision">
    <geometry>
      <cylinder>
        <radius>0.05</radius>
        <length>0.3</length>
      </cylinder>
    </geometry>

    <!-- Disable self-collision with adjacent links -->
    <self_collide>false</self_collide>
  </collision>
</link>

<!-- Using collision groups in more complex scenarios -->
<collision name="collision_with_environment_only">
  <geometry>
    <mesh>
      <uri>model://humanoid/meshes/complex_shape.stl</uri>
    </geometry>
  </collision>
  <surface>
    <contact>
      <collide_without_contact>true</collide_without_contact>
    </contact>
  </surface>
</collision>
```

## Contact Modeling for Humanoid Locomotion

### Foot-Ground Contact

For humanoid robots, foot-ground contact modeling is critical for stable walking:

```xml
<link name="left_foot">
  <collision name="foot_collision">
    <geometry>
      <box>
        <size>0.25 0.1 0.02</size>  <!-- Thin box for better contact detection -->
      </box>
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>1e-5</soft_cfm>  <!-- Very stiff contact for stability -->
          <soft_erp>0.99</soft_erp>  <!-- High error reduction -->
          <kp>1e7</kp>              <!-- Very high stiffness -->
          <kd>100.0</kd>            <!-- Appropriate damping -->
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.8</mu>    <!-- High friction for grip -->
          <mu2>0.8</mu2>
          <fdir1>1 0 0</fdir1>  <!-- Friction direction for anisotropic friction -->
        </ode>
        <torsional>
          <coefficient>0.1</coefficient>
          <use_patch_radius>false</use_patch_radius>
          <surface_radius>0.01</surface_radius>
        </torsional>
      </friction>
    </surface>
  </collision>
</link>
```

### Soft Contact Modeling

For more realistic humanoid interactions, soft contact modeling can be used:

```xml
<!-- Soft contact for hand-object interaction -->
<link name="left_hand">
  <collision name="hand_collision">
    <geometry>
      <mesh>
        <uri>model://humanoid/meshes/hand_collision.dae</uri>
      </mesh>
    </geometry>
    <surface>
      <contact>
        <ode>
          <soft_cfm>0.01</soft_cfm>  <!-- Softer for manipulation tasks -->
          <soft_erp>0.5</soft_erp>
          <kp>1e4</kp>              <!-- Lower stiffness for softer contact -->
          <kd>1.0</kd>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.5</mu>    <!-- Moderate friction for manipulation -->
          <mu2>0.4</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
</link>
```

## Performance Optimization for Complex Humanoid Models

### Multi-Threaded Physics

Gazebo can use multi-threading to improve performance with complex models:

```xml
<physics name="humanoid_physics" type="dart">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Enable threading for better performance -->
  <dart>
    <threads>4</threads>  <!-- Use multiple threads for physics calculations -->
  </dart>
</physics>
```

### Simplified Collision Models

For performance, use simplified collision models while maintaining accuracy:

```xml
<!-- Complex visual model with simplified collision -->
<link name="complex_link">
  <!-- Detailed visual geometry -->
  <visual name="detailed_visual">
    <geometry>
      <mesh>
        <uri>model://humanoid/meshes/complex_visual.dae</uri>
      </mesh>
    </geometry>
  </visual>

  <!-- Simplified collision geometry -->
  <collision name="simple_collision">
    <geometry>
      <box>
        <size>0.1 0.1 0.1</size>
      </box>
    </geometry>
  </collision>
</link>
```

### Contact Reduction

Reduce the number of contacts to improve performance:

```xml
<world name="humanoid_world">
  <physics name="humanoid_physics" type="dart">
    <max_step_size>0.001</max_step_size>

    <dart>
      <constraints>
        <!-- Limit maximum contacts to improve performance -->
        <max_contacts>20</max_contacts>

        <!-- Contact parameters -->
        <contact_surface_layer>0.001</contact_surface_layer>
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
      </constraints>
    </dart>
  </physics>
</world>
```

## Gazebo-ROS Integration for Humanoid Robots

### Gazebo ROS Packages

Gazebo integrates with ROS through specialized packages:

```xml
<!-- Example model with ROS plugins -->
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Transmission definitions for ROS control -->
  <transmission name="left_hip_yaw_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_yaw_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_yaw_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugins for ROS integration -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <robotNamespace>/humanoid</robotNamespace>
      <jointName>left_hip_yaw_joint, left_hip_roll_joint, left_hip_pitch_joint</jointName>
      <updateRate>50</updateRate>
    </plugin>
  </gazebo>
</robot>
```

### Sensor Integration

Integrate sensors with realistic physics properties:

```xml
<!-- IMU sensor with realistic noise properties -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>  <!-- ~0.1 deg/s stddev -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>  <!-- ~0.017 m/s² stddev -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>

<!-- Force/Torque sensor for foot contact -->
<gazebo reference="left_foot">
  <sensor name="left_foot_ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>500</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
  </sensor>
</gazebo>
```

## Advanced Physics Scenarios for Humanoid Robots

### Walking and Balance Simulation

Configure physics for realistic walking and balance:

```xml
<!-- World with specific parameters for walking simulation -->
<world name="walking_test_world">
  <physics name="walking_physics" type="dart">
    <max_step_size>0.0005</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>2000</real_time_update_rate>

    <dart>
      <solver>
        <type>PGS</type>
        <iters>100</iters>  <!-- More iterations for stable balance -->
        <sor>1.0</sor>
      </solver>

      <constraints>
        <contact_surface_layer>0.0001</contact_surface_layer>  <!-- Very thin for precise contact -->
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
        <cfm>1e-8</cfm>  <!-- Very low CFM for stiff contacts -->
        <erp>0.99</erp>  <!-- High ERP for quick error correction -->
        <max_contacts>50</max_contacts>  <!-- Allow more contacts for multi-contact scenarios -->
      </constraints>
    </dart>
  </physics>

  <!-- Flat ground with appropriate friction -->
  <model name="ground_plane">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.8</mu>    <!-- High friction for stable walking -->
              <mu2>0.8</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <soft_cfm>1e-6</soft_cfm>
              <soft_erp>0.99</soft_erp>
              <kp>1e8</kp>    <!-- Very stiff ground -->
              <kd>1000.0</kd>
            </ode>
          </contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</world>
```

### Manipulation and Grasping

Configure physics for manipulation tasks:

```xml
<!-- World optimized for manipulation -->
<world name="manipulation_world">
  <physics name="manipulation_physics" type="dart">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>

    <dart>
      <solver>
        <type>PGS</type>
        <iters>50</iters>
        <sor>1.3</sor>
      </solver>

      <constraints>
        <contact_surface_layer>0.001</contact_surface_layer>
        <contact_max_correcting_vel>10</contact_max_correcting_vel>  <!-- Lower for manipulation -->
        <cfm>1e-5</cfm>
        <erp>0.8</erp>  <!-- Lower ERP for softer contacts during manipulation -->
      </constraints>
    </dart>
  </physics>

  <!-- Table for manipulation tasks -->
  <model name="manipulation_table">
    <pose>0 0 0 0 0 0</pose>
    <link name="table_base">
      <collision name="collision">
        <geometry>
          <box>
            <size>1.2 0.8 0.8</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.6</mu>    <!-- Moderate friction for objects on table -->
              <mu2>0.5</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>1.2 0.8 0.8</size>
          </box>
        </geometry>
      </visual>
      <inertial>
        <mass>50.0</mass>
        <inertia>
          <ixx>6.33</ixx>
          <iyy>8.67</iyy>
          <izz>10.0</izz>
        </inertia>
      </inertial>
    </link>
  </model>

  <!-- Objects for manipulation -->
  <model name="manipulation_object_1">
    <pose>0.2 0.1 0.85 0 0 0</pose>
    <link name="object_link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.15</length>
          </cylinder>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.4</mu>    <!-- Lower friction for easier manipulation -->
              <mu2>0.3</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <soft_cfm>0.001</soft_cfm>
              <soft_erp>0.5</soft_erp>
              <kp>1e5</kp>
              <kd>1.0</kd>
            </ode>
          </contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.15</length>
          </cylinder>
        </geometry>
      </visual>
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.0014</ixx>
          <iyy>0.0014</iyy>
          <izz>0.000625</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</world>
```

## Debugging and Troubleshooting Physics Issues

### Common Physics Problems in Humanoid Simulation

#### 1. Unstable Joints
```xml
<!-- Fix for unstable joints -->
<joint name="unstable_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <axis>
    <xyz>0 0 1</xyz>
    <dynamics>
      <damping>1.0</damping>      <!-- Add damping to stabilize -->
      <friction>0.1</friction>    <!-- Add friction if needed -->
    </dynamics>
  </axis>
</joint>
```

#### 2. Penetration Issues
```xml
<!-- Fix for object penetration -->
<collision name="collision">
  <geometry>
    <box>
      <size>0.1 0.1 0.1</size>
    </box>
  </geometry>
  <surface>
    <contact>
      <ode>
        <soft_cfm>1e-7</soft_cfm>  <!-- Very low CFM for stiff contacts -->
        <soft_erp>0.99</soft_erp>  <!-- High ERP to correct errors quickly -->
        <kp>1e7</kp>              <!-- High stiffness -->
        <kd>100.0</kd>            <!-- Appropriate damping -->
      </ode>
    </contact>
  </surface>
</collision>
```

#### 3. Balance Instability
```xml
<!-- Physics settings for stable humanoid balance -->
<physics name="stable_balance_physics" type="dart">
  <max_step_size>0.0002</max_step_size>  <!-- Smaller time step for stability -->
  <real_time_update_rate>5000</real_time_update_rate>
  <dart>
    <solver>
      <iters>200</iters>  <!-- More iterations for stability -->
    </solver>
  </dart>
</physics>
```

### Physics Debugging Tools

#### Gazebo GUI Physics Visualization
Enable physics visualization in Gazebo to debug issues:

```bash
# Launch Gazebo with physics visualization
gz sim -r -v 4 manipulation_world.sdf
```

In the Gazebo GUI, you can enable:
- Contact visualization
- Center of mass visualization
- Inertia visualization
- Joint axis visualization

## Performance Monitoring and Optimization

### Monitoring Simulation Performance

Monitor simulation performance with these metrics:

1. **Real Time Factor (RTF)**: Ratio of simulation time to real time
2. **Update Rate**: Actual physics updates per second
3. **Contact Count**: Number of active contacts in the simulation
4. **Solver Iterations**: Convergence of the constraint solver

### Optimization Strategies

#### 1. Adaptive Time Stepping
```xml
<!-- Adaptive time stepping based on simulation complexity -->
<physics name="adaptive_physics" type="dart">
  <!-- Start with conservative settings -->
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Monitor and adjust based on RTF -->
  <!-- If RTF < 0.8, increase step size or reduce update rate -->
  <!-- If RTF > 1.2, decrease step size for more accuracy -->
</physics>
```

#### 2. Selective High-Fidelity Simulation
```xml
<!-- Use high-fidelity physics only where needed -->
<model name="high_fidelity_region">
  <!-- Objects in immediate vicinity of robot -->
  <link name="critical_object">
    <collision name="precise_collision">
      <!-- Detailed collision geometry -->
      <geometry>
        <mesh>
          <uri>model://detailed_collision.stl</uri>
        </mesh>
      </geometry>
      <surface>
        <contact>
          <ode>
            <soft_cfm>1e-6</soft_cfm>
            <soft_erp>0.99</soft_erp>
          </ode>
        </contact>
      </surface>
    </collision>
  </link>
</model>

<model name="low_fidelity_region">
  <!-- Distant objects can use simplified physics -->
  <link name="distant_object">
    <collision name="simple_collision">
      <geometry>
        <box>
          <size>1.0 1.0 1.0</size>
        </box>
      </geometry>
      <!-- Less demanding contact parameters -->
    </collision>
  </link>
</model>
```

## Best Practices for Humanoid Robot Simulation

### 1. Progressive Complexity
- Start with simple models and basic physics
- Gradually add complexity and fine-tune parameters
- Validate each addition before proceeding

### 2. Parameter Documentation
- Document all physics parameters used
- Include rationale for parameter choices
- Record performance metrics for different configurations

### 3. Validation Against Reality
- Compare simulation results with physical robot data
- Validate contact forces, motion patterns, and stability
- Use simulation as a predictive tool, not just a visualization

### 4. Performance vs. Accuracy Trade-offs
- Identify critical simulation elements that need high fidelity
- Simplify non-critical elements for better performance
- Use different physics settings for different simulation phases

## Practical Example: Complete Humanoid Simulation Setup

Here's a complete example of a humanoid robot simulation configuration:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_simulation">
    <!-- Physics configuration optimized for humanoid -->
    <physics name="humanoid_physics" type="dart">
      <max_step_size>0.0005</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>2000</real_time_update_rate>

      <dart>
        <solver>
          <type>PGS</type>
          <iters>100</iters>
          <sor>1.0</sor>
        </solver>

        <constraints>
          <contact_surface_layer>0.0001</contact_surface_layer>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <cfm>1e-8</cfm>
          <erp>0.99</erp>
          <max_contacts>50</max_contacts>
        </constraints>
      </dart>
    </physics>

    <!-- Gravity -->
    <gravity>0 0 -9.8</gravity>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Humanoid robot will be spawned here -->
    <!-- Robot configuration would be in a separate model file -->

    <!-- Additional environment elements as needed -->
    <model name="obstacle">
      <pose>2 0 0 0 0 0</pose>
      <link name="obstacle_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.7</mu>
                <mu2>0.5</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.0067</ixx>
            <iyy>0.0067</iyy>
            <izz>0.0067</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Exercises

1. **Physics Parameter Tuning**: Create a simple humanoid model and experiment with different physics parameters (time step, solver iterations, contact properties). Document the effects on simulation stability and performance for walking and standing behaviors.

2. **Contact Modeling Challenge**: Design and implement a scenario that tests the limits of contact modeling (e.g., humanoid robot walking on a compliant surface or manipulating delicate objects). Adjust physics parameters to achieve realistic behavior.

3. **Performance Optimization**: Create a complex humanoid environment with multiple robots and objects. Implement optimization techniques to maintain real-time simulation performance while preserving physical accuracy.

## Summary

This chapter covered advanced Gazebo physics configuration for humanoid robots:

- Physics engine selection and configuration for humanoid applications
- Advanced collision detection and contact modeling techniques
- Performance optimization strategies for complex models
- Integration with ROS for complete simulation workflows
- Debugging and troubleshooting common physics issues
- Best practices for realistic humanoid simulation

Proper physics configuration is essential for creating believable and useful simulations of humanoid robots. The parameters and techniques covered in this chapter provide the foundation for developing accurate, stable, and performant simulation environments.

## Next Steps

In the next chapter, we'll explore Unity-based simulation environments and how they complement Gazebo for humanoid robotics applications. Unity's high-fidelity rendering and human-robot interaction capabilities provide different advantages compared to Gazebo's physics-focused approach.