# Exercise 6: Fuse IMU and Odometry Data Using Kalman Filter

## Problem Statement
Implement a Kalman Filter to fuse IMU and odometry data for improved robot state estimation. The filter should provide a more accurate and robust estimate of the robot's position, velocity, and orientation than either sensor alone.

## Learning Objectives
- Understand state estimation and sensor fusion concepts
- Implement Kalman Filter for multi-sensor fusion
- Apply the filter to robotics-specific sensor data
- Evaluate the performance of the fused estimate

## Implementation Requirements

### 1. State Representation
- Define the state vector (position, velocity, orientation, angular velocity)
- Implement state transition model
- Define control inputs and process noise

### 2. Measurement Model
- Implement measurement models for odometry and IMU
- Handle different measurement rates
- Define measurement noise for each sensor

### 3. Kalman Filter Implementation
- Implement prediction step
- Implement update step for each sensor type
- Handle time synchronization between sensors

### 4. Evaluation
- Compare fused estimate with individual sensor estimates
- Calculate error metrics
- Visualize the results

## Starter Code Template

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import math

class KalmanFilter:
    def __init__(self, 
                 state_dim: int, 
                 measurement_dim: int,
                 dt: float = 0.01):
        """
        Initialize the Kalman Filter.
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            dt: Time step
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        
        # Initialize state vector [x, y, theta, vx, vy, omega]
        self.x = np.zeros((state_dim, 1))
        
        # Initialize state covariance matrix
        self.P = np.eye(state_dim) * 1000.0  # Large initial uncertainty
        
        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1
        
        # Measurement noise covariance (will be set per sensor)
        self.R = np.eye(measurement_dim)
        
        # State transition matrix (will be updated in predict)
        self.F = np.eye(state_dim)
        
        # Control matrix (if applicable)
        self.B = np.zeros((state_dim, 1))  # No control input in this example
        
        # Measurement matrix (will be set per sensor)
        self.H = np.zeros((measurement_dim, state_dim))
    
    def predict(self, u: Optional[np.ndarray] = None):
        """
        Prediction step of the Kalman Filter.
        
        Args:
            u: Control input (optional)
        """
        # TODO: Implement prediction step
        # 1. Update state transition matrix F based on dt
        # 2. Predict state: x = F*x + B*u
        # 3. Predict covariance: P = F*P*F^T + Q
        pass
    
    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Update step of the Kalman Filter.
        
        Args:
            z: Measurement vector
            H: Measurement matrix
            R: Measurement noise covariance
        """
        # TODO: Implement update step
        # 1. Compute innovation: y = z - H*x
        # 2. Compute innovation covariance: S = H*P*H^T + R
        # 3. Compute Kalman gain: K = P*H^T*S^(-1)
        # 4. Update state: x = x + K*y
        # 5. Update covariance: P = (I - K*H)*P
        pass

class RobotStateEstimator:
    def __init__(self, dt: float = 0.01):
        """
        Initialize the robot state estimator with a Kalman Filter.
        
        Args:
            dt: Time step for the filter
        """
        # State vector: [x, y, theta, vx, vy, omega]
        # x, y: position
        # theta: orientation
        # vx, vy: linear velocities
        # omega: angular velocity
        self.state_dim = 6
        self.dt = dt
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter(self.state_dim, 3, dt)  # 3 for odometry (x, y, theta)
        
        # Initial state [x, y, theta, vx, vy, omega]
        self.kf.x = np.zeros((self.state_dim, 1))
        
        # Initial covariance (high uncertainty in velocities)
        self.kf.P = np.diag([0.1, 0.1, 0.1, 10.0, 10.0, 1.0])
        
        # Process noise (higher for velocities)
        self.kf.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.05])
        
        # IMU measurement noise
        self.imu_R = np.diag([0.01, 0.01, 0.001])  # [ax, ay, omega]
        
        # Odometry measurement noise
        self.odom_R = np.diag([0.05, 0.05, 0.01])  # [x, y, theta]
        
    def predict(self):
        """Predict the next state using the motion model."""
        # State transition matrix for constant velocity model
        F = np.eye(self.state_dim)
        F[0, 3] = self.dt  # x = x + vx*dt
        F[1, 4] = self.dt  # y = y + vy*dt
        F[2, 5] = self.dt  # theta = theta + omega*dt
        
        self.kf.F = F
        
        # Control input (none in this case)
        u = None
        
        self.kf.predict(u)
    
    def update_odometry(self, x: float, y: float, theta: float):
        """
        Update the state estimate with odometry measurement.
        
        Args:
            x: Measured x position
            y: Measured y position
            theta: Measured orientation
        """
        # Measurement vector [x, y, theta]
        z = np.array([[x], [y], [theta]])
        
        # Measurement matrix for odometry (only measures x, y, theta)
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # Measure x
        H[1, 1] = 1  # Measure y
        H[2, 2] = 1  # Measure theta
        
        # Update with odometry measurement
        self.kf.update(z, H, self.odom_R)
    
    def update_imu(self, ax: float, ay: float, omega: float):
        """
        Update the state estimate with IMU measurement.
        
        Args:
            ax: Measured linear acceleration in x direction
            ay: Measured linear acceleration in y direction
            omega: Measured angular velocity
        """
        # Measurement vector [ax, ay, omega]
        z = np.array([[ax], [ay], [omega]])
        
        # Measurement matrix for IMU
        # In this case, we're measuring acceleration and angular velocity
        # We'll use a simplified model where we directly measure acceleration
        # and angular velocity
        H = np.zeros((3, self.state_dim))
        H[0, 3] = 1/self.dt  # Approximate ax from vx change
        H[1, 4] = 1/self.dt  # Approximate ay from vy change
        H[2, 5] = 1         # Measure omega directly
        
        # Update with IMU measurement
        self.kf.update(z, H, self.imu_R)
    
    def get_state(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get the current estimated state.
        
        Returns:
            Tuple of (x, y, theta, vx, vy, omega)
        """
        state = self.kf.x.flatten()
        return tuple(state)

class SensorFusionTester:
    def __init__(self):
        """Initialize the sensor fusion tester."""
        self.estimator = RobotStateEstimator(dt=0.01)
        self.time = 0.0
        self.dt = 0.01
        
        # Store results for visualization
        self.times = []
        self.gt_states = []  # Ground truth
        self.estimated_states = []
        self.odom_states = []  # Odometry only
        self.imu_states = []   # IMU only (integrated)
        
    def simulate_robot_motion(self, duration: float = 10.0):
        """
        Simulate robot motion and sensor readings.
        
        Args:
            duration: Duration of simulation in seconds
        """
        # Initial state
        x, y, theta = 0.0, 0.0, 0.0
        vx, vy, omega = 0.1, 0.05, 0.02  # Constant velocities
        
        # For odometry-only and IMU-only estimates
        odom_x, odom_y, odom_theta = x, y, theta
        imu_vx, imu_vy, imu_theta = 0.0, 0.0, theta
        
        # Measurement noise
        odom_noise = 0.02
        imu_noise = 0.01
        
        steps = int(duration / self.dt)
        
        for i in range(steps):
            # Update ground truth
            x += vx * self.dt
            y += vy * self.dt
            theta += omega * self.dt
            
            # Generate noisy measurements
            odom_x_meas = x + np.random.normal(0, odom_noise)
            odom_y_meas = y + np.random.normal(0, odom_noise)
            odom_theta_meas = theta + np.random.normal(0, odom_noise*0.1)
            
            ax = 0.0  # Assuming constant velocity, so acceleration is 0
            ay = 0.0
            omega_meas = omega + np.random.normal(0, imu_noise)
            
            # Update estimators
            self.estimator.predict()
            self.estimator.update_odometry(odom_x_meas, odom_y_meas, odom_theta_meas)
            self.estimator.update_imu(ax, ay, omega_meas)
            
            # Update odometry-only estimate (with noise)
            odom_x += (vx + np.random.normal(0, odom_noise*0.1)) * self.dt
            odom_y += (vy + np.random.normal(0, odom_noise*0.1)) * self.dt
            odom_theta += (omega + np.random.normal(0, odom_noise*0.05)) * self.dt
            
            # Update IMU-only estimate (integrate measurements)
            imu_vx += ax * self.dt
            imu_vy += ay * self.dt
            imu_theta += omega_meas * self.dt
            # Update position based on integrated velocity
            imu_x = imu_vx * (i * self.dt)  # Simplified - in reality, integrate position too
            imu_y = imu_vy * (i * self.dt)
            
            # Store results
            self.times.append(self.time)
            self.gt_states.append((x, y, theta, vx, vy, omega))
            self.estimated_states.append(self.estimator.get_state())
            self.odom_states.append((odom_x, odom_y, odom_theta, vx, vy, omega))
            self.imu_states.append((imu_x, imu_y, imu_theta, imu_vx, imu_vy, omega_meas))
            
            self.time += self.dt
    
    def calculate_error_metrics(self) -> dict:
        """
        Calculate error metrics for the fused estimate.
        
        Returns:
            Dictionary of error metrics
        """
        gt_array = np.array(self.gt_states)
        est_array = np.array(self.estimated_states)
        
        # Calculate position error
        pos_error = np.sqrt((gt_array[:, 0] - est_array[:, 0])**2 + 
                           (gt_array[:, 1] - est_array[:, 1])**2)
        
        # Calculate orientation error
        orient_error = np.abs(gt_array[:, 2] - est_array[:, 2])
        # Normalize angle difference to [-pi, pi]
        orient_error = np.arctan2(np.sin(orient_error), np.cos(orient_error))
        orient_error = np.abs(orient_error)
        
        # Calculate velocity error
        vel_error = np.sqrt((gt_array[:, 3] - est_array[:, 3])**2 + 
                           (gt_array[:, 4] - est_array[:, 4])**2)
        
        metrics = {
            'mean_pos_error': np.mean(pos_error),
            'std_pos_error': np.std(pos_error),
            'mean_orient_error': np.mean(orient_error),
            'std_orient_error': np.std(orient_error),
            'mean_vel_error': np.mean(vel_error),
            'std_vel_error': np.std(vel_error)
        }
        
        return metrics
    
    def visualize_results(self):
        """
        Visualize the results of the sensor fusion.
        """
        # Convert to numpy arrays for easier plotting
        times = np.array(self.times)
        gt_array = np.array(self.gt_states)
        est_array = np.array(self.estimated_states)
        odom_array = np.array(self.odom_states)
        imu_array = np.array(self.imu_states)
        
        # Plot position
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(gt_array[:, 0], gt_array[:, 1], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(est_array[:, 0], est_array[:, 1], 'b-', label='Fused Estimate', linewidth=2)
        plt.plot(odom_array[:, 0], odom_array[:, 1], 'r--', label='Odometry Only', alpha=0.7)
        plt.plot(imu_array[:, 0], imu_array[:, 1], 'm--', label='IMU Only', alpha=0.7)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Trajectory')
        plt.legend()
        plt.grid(True)
        
        # Plot X position over time
        plt.subplot(2, 3, 2)
        plt.plot(times, gt_array[:, 0], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(times, est_array[:, 0], 'b-', label='Fused Estimate', linewidth=2)
        plt.plot(times, odom_array[:, 0], 'r--', label='Odometry Only', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('X Position vs Time')
        plt.legend()
        plt.grid(True)
        
        # Plot Y position over time
        plt.subplot(2, 3, 3)
        plt.plot(times, gt_array[:, 1], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(times, est_array[:, 1], 'b-', label='Fused Estimate', linewidth=2)
        plt.plot(times, odom_array[:, 1], 'r--', label='Odometry Only', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position vs Time')
        plt.legend()
        plt.grid(True)
        
        # Plot orientation
        plt.subplot(2, 3, 4)
        plt.plot(times, gt_array[:, 2], 'g-', label='Ground Truth', linewidth=2)
        plt.plot(times, est_array[:, 2], 'b-', label='Fused Estimate', linewidth=2)
        plt.plot(times, odom_array[:, 2], 'r--', label='Odometry Only', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Orientation (rad)')
        plt.title('Orientation vs Time')
        plt.legend()
        plt.grid(True)
        
        # Plot position error
        plt.subplot(2, 3, 5)
        pos_error_fused = np.sqrt((gt_array[:, 0] - est_array[:, 0])**2 + 
                                 (gt_array[:, 1] - est_array[:, 1])**2)
        pos_error_odom = np.sqrt((gt_array[:, 0] - odom_array[:, 0])**2 + 
                                (gt_array[:, 1] - odom_array[:, 1])**2)
        plt.plot(times, pos_error_fused, 'b-', label='Fused Estimate Error', linewidth=2)
        plt.plot(times, pos_error_odom, 'r-', label='Odometry Only Error', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('Position Error vs Time')
        plt.legend()
        plt.grid(True)
        
        # Plot velocity
        plt.subplot(2, 3, 6)
        plt.plot(times, gt_array[:, 3], 'g-', label='Ground Truth Vx', linewidth=2)
        plt.plot(times, gt_array[:, 4], 'g--', label='Ground Truth Vy', linewidth=2)
        plt.plot(times, est_array[:, 3], 'b-', label='Fused Vx', linewidth=2)
        plt.plot(times, est_array[:, 4], 'b--', label='Fused Vy', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create sensor fusion tester
    tester = SensorFusionTester()
    
    # Run simulation
    print("Running simulation...")
    tester.simulate_robot_motion(duration=10.0)
    
    # Calculate error metrics
    metrics = tester.calculate_error_metrics()
    print("Error Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize results
    tester.visualize_results()
```

## Evaluation Criteria
- Accuracy: The fused estimate should be more accurate than individual sensors
- Robustness: The filter should handle sensor noise appropriately
- Efficiency: The implementation should run in real-time
- Code Quality: Well-documented, readable, and maintainable code
- Convergence: The estimate should converge to the true state over time

## Hints and Resources
- Start with a simple constant velocity model before implementing more complex dynamics
- Pay attention to units and coordinate systems
- Implement proper angle normalization for orientation values
- Consider using Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF) for nonlinear systems

## Extensions
- Implement Extended Kalman Filter (EKF) for nonlinear sensor models
- Add more sensor types (GPS, magnetometer, etc.)
- Implement sensor validation and fault detection
- Integrate with ROS for real sensor data