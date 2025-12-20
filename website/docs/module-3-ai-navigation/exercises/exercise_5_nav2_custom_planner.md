# Exercise 5: Create a Custom Nav2 Planner Plugin

## Problem Statement
Create a custom global planner plugin for Nav2 that implements a specific path planning algorithm (e.g., Dijkstra, Jump Point Search, or a bio-inspired algorithm like Ant Colony Optimization).

## Learning Objectives
- Understand the Nav2 plugin architecture
- Implement a custom global planner plugin
- Integrate with Nav2's costmap system
- Evaluate custom planner performance against default planners

## Implementation Requirements

### 1. Plugin Infrastructure
- Create a class that inherits from `nav2_core::GlobalPlanner`
- Implement required interface methods (`configure`, `cleanup`, `setPlan`)
- Register the plugin with Nav2 using pluginlib

### 2. Custom Path Planning Algorithm
- Implement the core path planning algorithm
- Integrate with Nav2's costmap for obstacle information
- Handle path optimization and smoothing

### 3. Parameter Configuration
- Use ROS parameters for algorithm configuration
- Implement proper logging and error handling
- Handle edge cases (no path found, invalid inputs)

## Starter Code Template

```cpp
// custom_planner.hpp
#ifndef CUSTOM_PLANNER_HPP_
#define CUSTOM_PLANNER_HPP_

#include <nav2_core/global_planner.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_util/lifecycle_node.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <string>

namespace custom_planner {

class CustomPlanner : public nav2_core::GlobalPlanner {
public:
    /**
     * @brief Configure the planner with ROS parameters and lifecycle node
     */
    void configure(
        const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
        std::string name,
        const std::shared_ptr<tf2_ros::Buffer>& tf,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS>& costmap_ros) override;

    /**
     * @brief Clean up resources
     */
    void cleanup() override;

    /**
     * @brief Activate the planner
     */
    void activate() override;

    /**
     * @brief Deactivate the planner
     */
    void deactivate() override;

    /**
     * @brief Create a plan from start to goal
     */
    nav_msgs::msg::Path createPlan(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal) override;

private:
    /**
     * @brief Implementation of the custom path planning algorithm
     */
    nav_msgs::msg::Path planPath(int start_x, int start_y, int goal_x, int goal_y);

    // ROS interfaces
    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;

    // Parameters
    double cost_scaling_factor_;
    bool allow_unknown_;
    std::string default_tolerance_;

    // Additional planner-specific variables
    // TODO: Add any additional variables needed for your algorithm
};

} // namespace custom_planner

#endif // CUSTOM_PLANNER_HPP_
```

```cpp
// custom_planner.cpp
#include "custom_planner.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

namespace custom_planner {

void CustomPlanner::configure(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    std::string name,
    const std::shared_ptr<tf2_ros::Buffer>& tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS>& costmap_ros) {
    
    // Initialize member variables
    node_ = node;
    name_ = name;
    tf_ = tf;
    costmap_ros_ = costmap_ros;
    costmap_ = costmap_ros_->getCostmap();
    
    // Declare parameters
    node_->declare_parameter(name_ + ".cost_scaling_factor", 3.0);
    node_->declare_parameter(name_ + ".allow_unknown", true);
    
    // Get parameters
    node_->get_parameter(name_ + ".cost_scaling_factor", cost_scaling_factor_);
    node_->get_parameter(name_ + ".allow_unknown", allow_unknown_);
    
    RCLCPP_INFO(node_->get_logger(), "Configured CustomPlanner");
}

void CustomPlanner::cleanup() {
    RCLCPP_INFO(node_->get_logger(), "Cleaning up CustomPlanner");
}

void CustomPlanner::activate() {
    RCLCPP_INFO(node_->get_logger(), "Activating CustomPlanner");
}

void CustomPlanner::deactivate() {
    RCLCPP_INFO(node_->get_logger(), "Deactivating CustomPlanner");
}

nav_msgs::msg::Path CustomPlanner::createPlan(
    const geometry_msgs::msg::PoseStamped& start,
    const geometry_msgs::msg::PoseStamped& goal) {
    
    // Initialize the path
    nav_msgs::msg::Path path;
    path.header.frame_id = costmap_ros_->getGlobalFrameID();
    path.header.stamp = node_->now();
    
    // Convert start and goal poses to costmap coordinates
    double start_x = start.pose.position.x;
    double start_y = start.pose.position.y;
    double goal_x = goal.pose.position.x;
    double goal_y = goal.pose.position.y;
    
    unsigned int start_m_x, start_m_y, goal_m_x, goal_m_y;
    
    // Convert world coordinates to map coordinates
    if (!costmap_->worldToMap(start_x, start_y, start_m_x, start_m_y)) {
        RCLCPP_WARN(node_->get_logger(), 
            "Start coordinates (%.2f, %.2f) are outside the map", start_x, start_y);
        return path;
    }
    
    if (!costmap_->worldToMap(goal_x, goal_y, goal_m_x, goal_m_y)) {
        RCLCPP_WARN(node_->get_logger(), 
            "Goal coordinates (%.2f, %.2f) are outside the map", goal_x, goal_y);
        return path;
    }
    
    // Check if start and goal are in free space
    if (costmap_->getCost(start_m_x, start_m_y) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        RCLCPP_WARN(node_->get_logger(), "Start is in an obstacle space");
        return path;
    }
    
    if (costmap_->getCost(goal_m_x, goal_m_y) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        RCLCPP_WARN(node_->get_logger(), "Goal is in an obstacle space");
        return path;
    }
    
    // Call the custom path planning algorithm
    path = planPath(start_m_x, start_m_y, goal_m_x, goal_m_y);
    
    return path;
}

nav_msgs::msg::Path CustomPlanner::planPath(
    int start_x, int start_y, int goal_x, int goal_y) {
    
    // TODO: Implement your custom path planning algorithm here
    // This is a placeholder that just creates a straight line
    
    nav_msgs::msg::Path path;
    path.header.frame_id = costmap_ros_->getGlobalFrameID();
    path.header.stamp = node_->now();
    
    // For this example, we'll implement a simple Dijkstra-like algorithm
    // You should replace this with your own algorithm
    
    // Create a simple straight-line path as a placeholder
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    
    // Convert map coordinates back to world coordinates
    double world_x, world_y;
    costmap_->mapToWorld(start_x, start_y, world_x, world_y);
    pose.pose.position.x = world_x;
    pose.pose.position.y = world_y;
    pose.pose.position.z = 0.0;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
    
    // Add intermediate points (simplified - in reality, implement your algorithm)
    for (int i = 1; i < 10; ++i) {
        double t = i / 10.0;
        costmap_->mapToWorld(
            start_x + (goal_x - start_x) * t,
            start_y + (goal_y - start_y) * t,
            world_x, world_y);
        
        pose.pose.position.x = world_x;
        pose.pose.position.y = world_y;
        path.poses.push_back(pose);
    }
    
    costmap_->mapToWorld(goal_x, goal_y, world_x, world_y);
    pose.pose.position.x = world_x;
    pose.pose.position.y = world_y;
    path.poses.push_back(pose);
    
    return path;
}

} // namespace custom_planner

// Register this plugin with pluginlib
PLUGINLIB_EXPORT_CLASS(custom_planner::CustomPlanner, nav2_core::GlobalPlanner)
```

```xml
<!-- custom_planner_plugin.xml -->
<class_libraries>
  <library path="custom_planner_lib">
    <class type="custom_planner::CustomPlanner" base_class_type="nav2_core::GlobalPlanner">
      <description>Custom Global Planner Plugin</description>
    </class>
  </library>
</class_libraries>
```

```cmake
# CMakeLists.txt (partial)
cmake_minimum_required(VERSION 3.5)
project(custom_nav2_planner)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_library(custom_planner_lib SHARED
  src/custom_planner.cpp
)

target_include_directories(custom_planner_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_definitions(custom_planner_lib
  PRIVATE "CUSTOM_PLANNER_BUILDING_DLL")

ament_target_dependencies(custom_planner_lib
  nav2_core
  nav2_costmap_2d
  pluginlib
  rclcpp
  geometry_msgs
  nav_msgs
  tf2_geometry_msgs)

pluginlib_export_plugin_description_file(nav2_core custom_planner_plugin.xml)

install(TARGETS custom_planner_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_include_directories(include)
ament_export_libraries(custom_planner_lib)
ament_export_dependencies(
  nav2_core
  nav2_costmap_2d
  pluginlib
  rclcpp
  geometry_msgs
  nav_msgs
  tf2_geometry_msgs)

ament_package()
```

## Configuration for Nav2

To use the custom planner, you need to update your Nav2 configuration:

```yaml
# planner_server.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "custom_planner::CustomPlanner"
      # Add any custom parameters for your planner
      cost_scaling_factor: 3.0
      allow_unknown: true
```

## Evaluation Criteria
- Correctness: The planner should find valid paths that avoid obstacles
- Efficiency: The planner should run within the expected frequency
- Integration: The plugin should properly integrate with Nav2
- Code Quality: Well-documented, readable, and maintainable code
- Performance: Compare with default planners (e.g., NavFn, A*)

## Hints and Resources
- Start with a simple algorithm like Dijkstra before implementing more complex ones
- Use Nav2's costmap to get obstacle information
- Implement proper error handling for edge cases
- Test with various map configurations and obstacle layouts

## Extensions
- Implement a more sophisticated algorithm (e.g., Jump Point Search, Theta*)
- Add dynamic obstacle avoidance
- Implement path optimization/simplification
- Add support for kinodynamic constraints