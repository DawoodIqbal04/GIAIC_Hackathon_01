# Exercise 3: Vector Field Histogram for Obstacle Avoidance

## Problem Statement
Implement the Vector Field Histogram (VFH) algorithm for local obstacle avoidance. The algorithm should generate velocity commands to navigate a robot around obstacles toward a goal while avoiding collisions.

## Learning Objectives
- Understand local navigation and obstacle avoidance techniques
- Implement the VFH algorithm for real-time navigation
- Apply sensor data processing for navigation decisions
- Evaluate the effectiveness of local navigation algorithms

## Implementation Requirements

### 1. Sensor Data Processing
- Process laser scan data to create a polar histogram
- Implement obstacle density calculation in different angular sectors
- Filter sensor data to remove noise and invalid readings

### 2. Vector Field Histogram
- Create a polar histogram representing obstacle density
- Identify navigable gaps in the histogram
- Select an appropriate direction based on goal direction and gaps

### 3. Velocity Commands
- Generate velocity commands (linear and angular) based on selected direction
- Implement adaptive velocity scaling based on obstacle proximity
- Ensure smooth transitions between directions

## Starter Code Template

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import math

class VectorFieldHistogram:
    def __init__(self, 
                 sector_count: int = 72,  # 72 sectors = 5 degree resolution
                 min_distance: float = 0.3,  # Minimum sensor distance (m)
                 max_distance: float = 2.0,  # Maximum sensor distance (m)
                 safety_threshold: float = 0.5):  # Safety threshold for obstacles (m)
        """
        Initialize the VFH algorithm.
        
        Args:
            sector_count: Number of angular sectors for the histogram
            min_distance: Minimum distance for valid sensor readings
            max_distance: Maximum distance for sensor readings
            safety_threshold: Minimum safe distance from obstacles
        """
        self.sector_count = sector_count
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.safety_threshold = safety_threshold
        self.angle_resolution = 2 * math.pi / sector_count
        
    def create_histogram(self, scan_data: List[float]) -> np.ndarray:
        """
        Create a polar histogram from laser scan data.
        
        Args:
            scan_data: List of distances from laser scanner
            
        Returns:
            Polar histogram representing obstacle density
        """
        # TODO: Create histogram from scan data
        # Each sector should represent obstacle density
        pass
    
    def smooth_histogram(self, histogram: np.ndarray, smoothing_window: int = 2) -> np.ndarray:
        """
        Smooth the histogram to reduce noise.
        
        Args:
            histogram: Original histogram
            smoothing_window: Size of smoothing window
            
        Returns:
            Smoothed histogram
        """
        # TODO: Apply smoothing to reduce noise in histogram
        pass
    
    def find_navigable_gaps(self, 
                            histogram: np.ndarray, 
                            threshold: float) -> List[Tuple[int, int]]:
        """
        Find navigable gaps in the histogram.
        
        Args:
            histogram: Polar histogram
            threshold: Threshold to determine navigable space
            
        Returns:
            List of (start_idx, end_idx) for each navigable gap
        """
        # TODO: Identify gaps in the histogram where navigation is possible
        pass
    
    def select_direction(self, 
                        gaps: List[Tuple[int, int]], 
                        goal_angle: float,
                        current_heading: float) -> float:
        """
        Select the best direction to navigate based on gaps and goal.
        
        Args:
            gaps: List of navigable gaps
            goal_angle: Angle to the goal relative to robot
            current_heading: Current robot heading
            
        Returns:
            Selected direction angle
        """
        # TODO: Select best direction based on goal and available gaps
        pass
    
    def calculate_velocities(self, 
                            selected_direction: float, 
                            goal_direction: float,
                            obstacle_distance: float) -> Tuple[float, float]:
        """
        Calculate linear and angular velocities based on selected direction.
        
        Args:
            selected_direction: Selected navigation direction
            goal_direction: Direction to goal
            obstacle_distance: Distance to nearest obstacle
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # TODO: Calculate appropriate velocities based on direction and obstacles
        pass
    
    def plan_velocity(self, 
                     scan_data: List[float], 
                     goal_direction: float,
                     current_heading: float = 0.0) -> Tuple[float, float]:
        """
        Main method to plan velocity based on sensor data and goal.
        
        Args:
            scan_data: Laser scan data
            goal_direction: Direction to goal relative to robot
            current_heading: Current robot heading (relative to global frame)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # TODO: Implement the complete VFH algorithm
        # 1. Create histogram from scan data
        # 2. Smooth histogram
        # 3. Find navigable gaps
        # 4. Select direction based on goal
        # 5. Calculate velocities
        pass
    
    def visualize(self, scan_data: List[float], histogram: np.ndarray, selected_direction: float):
        """
        Visualize the VFH process.
        
        Args:
            scan_data: Original scan data
            histogram: Generated histogram
            selected_direction: Selected navigation direction
        """
        # TODO: Create visualization showing scan data, histogram, and selected direction
        pass

class VFHNavigator:
    def __init__(self):
        """Initialize the VFH navigator."""
        self.vfh = VectorFieldHistogram()
        self.linear_vel_limit = 0.5  # m/s
        self.angular_vel_limit = 1.0  # rad/s
        
    def navigate(self, scan_data: List[float], goal_pos: Tuple[float, float], 
                 robot_pos: Tuple[float, float], robot_heading: float) -> Tuple[float, float]:
        """
        Navigate toward a goal position using VFH algorithm.
        
        Args:
            scan_data: Laser scan data
            goal_pos: Goal position (x, y)
            robot_pos: Current robot position (x, y)
            robot_heading: Current robot heading (radians)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        # Calculate direction to goal
        dx = goal_pos[0] - robot_pos[0]
        dy = goal_pos[1] - robot_pos[1]
        goal_direction = math.atan2(dy, dx)
        
        # Calculate relative goal direction
        relative_goal_direction = goal_direction - robot_heading
        relative_goal_direction = math.atan2(math.sin(relative_goal_direction), 
                                           math.cos(relative_goal_direction))  # Normalize to [-pi, pi]
        
        # Plan velocity using VFH
        linear_vel, angular_vel = self.vfh.plan_velocity(
            scan_data, relative_goal_direction, robot_heading)
        
        # Limit velocities
        linear_vel = max(-self.linear_vel_limit, min(linear_vel, self.linear_vel_limit))
        angular_vel = max(-self.angular_vel_limit, min(angular_vel, self.angular_vel_limit))
        
        return linear_vel, angular_vel

# Example usage:
if __name__ == "__main__":
    # Simulated laser scan data (360 points, 1 degree resolution)
    scan_data = [float('inf')] * 360  # Initialize with max range
    
    # Add some obstacles (simulated)
    for i in range(80, 100):  # Obstacle at angles 80-100 degrees
        scan_data[i] = 0.8
    for i in range(260, 280):  # Obstacle at angles 260-280 degrees
        scan_data[i] = 1.2
    
    # Initialize VFH
    vfh = VectorFieldHistogram()
    
    # Example goal direction (in robot's frame)
    goal_direction = math.pi / 4  # 45 degrees
    current_heading = 0.0
    
    # Plan velocity
    linear_vel, angular_vel = vfh.plan_velocity(scan_data, goal_direction, current_heading)
    
    print(f"Planned velocity: linear={linear_vel:.2f}, angular={angular_vel:.2f}")
```

## Evaluation Criteria
- Safety: The robot should avoid collisions with obstacles
- Goal-seeking: The robot should make progress toward the goal
- Smoothness: Velocity commands should change smoothly over time
- Efficiency: The algorithm should run in real-time
- Code Quality: Well-documented, readable, and maintainable code

## Hints and Resources
- Use a polar coordinate system for the histogram
- Consider both primary and secondary goals (navigating to goal while avoiding obstacles)
- Implement hysteresis to prevent oscillation between directions
- Weight the goal direction more heavily when far from obstacles

## Extensions
- Implement VFH+ which considers robot dynamics and kinematics
- Integrate with global path planner for complete navigation system
- Add support for dynamic obstacles
- Implement in ROS with real robot hardware