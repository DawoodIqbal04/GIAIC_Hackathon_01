# Exercise 2: Particle Filter Implementation for Robot Localization

## Problem Statement
Implement a particle filter for robot localization in a known map. The particle filter should estimate the robot's position and orientation based on sensor measurements and motion commands.

## Learning Objectives
- Understand probabilistic robotics and state estimation
- Implement particle filtering for robot localization
- Apply sensor models and motion models in a probabilistic framework
- Evaluate localization accuracy and convergence

## Implementation Requirements

### 1. Environment Setup
- Create a 2D occupancy grid map
- Implement a simple robot model with position (x, y) and orientation (theta)
- Define motion commands (e.g., move forward, turn)

### 2. Particle Filter Implementation
- Initialize particles randomly across the map
- Implement motion model with noise
- Implement sensor model (e.g., range finder) with noise
- Implement importance sampling and resampling

### 3. Evaluation
- Compare estimated pose with ground truth
- Visualize particle distribution over time
- Calculate localization error metrics

## Starter Code Template

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

class Particle:
    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        """
        Initialize a particle with position and orientation.
        
        Args:
            x: X coordinate
            y: Y coordinate
            theta: Orientation in radians
            weight: Particle weight (probability)
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class ParticleFilter:
    def __init__(self, map: np.ndarray, num_particles: int = 1000):
        """
        Initialize the particle filter.
        
        Args:
            map: 2D occupancy grid map (0 = free space, 1 = occupied)
            num_particles: Number of particles to use
        """
        self.map = map
        self.num_particles = num_particles
        self.particles = []
        self.initialize_particles()
        
    def initialize_particles(self):
        """Initialize particles randomly across free space in the map."""
        # TODO: Initialize particles in free space with uniform distribution
        pass
    
    def motion_model(self, particle: Particle, control: Tuple[float, float], 
                     dt: float = 1.0) -> Particle:
        """
        Apply motion model with noise to predict particle movement.
        
        Args:
            particle: Particle to move
            control: Control command (linear velocity, angular velocity)
            dt: Time step
            
        Returns:
            New particle position after applying motion
        """
        # TODO: Implement motion model with added noise
        pass
    
    def sensor_model(self, particle: Particle, measurement: List[float]) -> float:
        """
        Calculate the likelihood of a measurement given the particle's pose.
        
        Args:
            particle: Particle to evaluate
            measurement: Sensor measurement (e.g., range measurements)
            
        Returns:
            Likelihood of the measurement given the particle's pose
        """
        # TODO: Implement sensor model (e.g., beam range finder model)
        pass
    
    def predict(self, control: Tuple[float, float]):
        """
        Predict step: update particles based on motion model.
        
        Args:
            control: Control command (linear velocity, angular velocity)
        """
        # TODO: Apply motion model to all particles
        pass
    
    def update(self, measurement: List[float]):
        """
        Update step: update particle weights based on sensor measurements.
        
        Args:
            measurement: Sensor measurement (e.g., range measurements)
        """
        # TODO: Update particle weights based on sensor model
        pass
    
    def resample(self):
        """Resample particles based on their weights."""
        # TODO: Implement resampling (e.g., low-variance resampler)
        pass
    
    def estimate(self) -> Tuple[float, float, float]:
        """
        Estimate the robot's pose based on particles.
        
        Returns:
            Estimated (x, y, theta) of the robot
        """
        # TODO: Calculate weighted average of particles
        pass
    
    def visualize(self, true_pose: Tuple[float, float, float] = None):
        """
        Visualize particles and estimated pose.
        
        Args:
            true_pose: True robot pose for comparison (optional)
        """
        # TODO: Implement visualization using matplotlib
        pass

# Example usage:
if __name__ == "__main__":
    # Create a simple map (0 = free space, 1 = obstacle)
    map = np.zeros((50, 50))
    # Add some obstacles
    map[20:30, 20:30] = 1
    
    # Initialize particle filter
    pf = ParticleFilter(map, num_particles=500)
    
    # True robot pose (in a real scenario, this would be unknown)
    true_pose = (10.0, 10.0, 0.0)
    
    # Simulate robot movement and sensor readings
    for step in range(100):
        # Control command (linear velocity, angular velocity)
        control = (0.1, 0.01)  # Move forward slightly and turn slowly
        
        # Simulated sensor measurement (simplified)
        measurement = [10.0, 15.0, 8.0, 12.0]  # Example range measurements
        
        # Prediction step
        pf.predict(control)
        
        # Update step
        pf.update(measurement)
        
        # Resample
        pf.resample()
        
        # Get estimate
        estimated_pose = pf.estimate()
        
        # Visualize every 10 steps
        if step % 10 == 0:
            pf.visualize(true_pose)
```

## Evaluation Criteria
- Accuracy: The estimated pose should converge to the true pose over time
- Robustness: The filter should handle sensor and motion noise appropriately
- Efficiency: The implementation should run in reasonable time
- Code Quality: Well-documented, readable, and maintainable code
- Visualization: Clear visualization of particle distribution and estimation

## Hints and Resources
- Use a low-variance resampler to avoid particle degeneracy
- Normalize particle weights after updating
- Consider using log probabilities to avoid numerical underflow
- The sensor model is critical for good performance - spend time on this component

## Extensions
- Implement adaptive particle filtering that changes the number of particles based on uncertainty
- Add multiple motion and sensor models for different environments
- Integrate with ROS for real robot localization
- Compare performance with other localization methods (e.g., EKF)