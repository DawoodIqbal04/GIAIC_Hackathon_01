---
id: chapter-8-sensor-simulation
title: "Chapter 8: Simulating Sensors: LiDAR, Depth Cameras, and IMUs"
sidebar_label: "Chapter 8: Simulating Sensors: LiDAR, Depth Cameras, and IMUs"
description: "Advanced sensor simulation techniques for humanoid robotics in Gazebo and Unity"
keywords: [lidar, camera, imu, sensors, simulation, robotics, depth, pointcloud, ros2]
tags: [sensors, simulation, perception]
authors: [book-authors]
difficulty: advanced
estimated_time: "90 minutes"
module: 2
chapter: 8
prerequisites: [physics-basics, ros2-foundations, urdf-basics, physics-simulation, unity-basics]
learning_objectives:
  - Implement realistic LiDAR sensor simulation in Gazebo and Unity
  - Configure depth camera simulation with realistic noise models
  - Simulate IMU sensors with accurate dynamics and noise characteristics
  - Integrate sensor data with ROS 2 perception pipelines
  - Validate sensor simulation accuracy against real-world measurements
related:
  - next: ../module-3-ai-navigation/intro
  - previous: chapter-7-unity-interaction
  - see_also: [chapter-6-gazebo-simulations, chapter-7-unity-interaction, ../module-1-ros-foundations/chapter-4-urdf-humanoids]
---

# Chapter 8: Simulating Sensors: LiDAR, Depth Cameras, and IMUs

## Learning Objectives

After completing this chapter, you will be able to:
- Implement realistic LiDAR sensor simulation in both Gazebo and Unity environments
- Configure depth camera simulation with realistic noise models and distortion
- Simulate IMU sensors with accurate dynamics and noise characteristics
- Integrate sensor data with ROS 2 perception pipelines for humanoid robotics
- Validate sensor simulation accuracy against real-world measurements

## Introduction

Sensor simulation is a critical component of digital twin systems for humanoid robotics. Realistic sensor simulation enables the development and testing of perception algorithms, navigation systems, and safety mechanisms without requiring access to expensive physical hardware. For humanoid robots, which rely on multiple sensor modalities for safe and effective operation, accurate sensor simulation is essential for bridging the reality gap between simulation and real-world deployment.

This chapter explores the simulation of three critical sensor types for humanoid robotics: LiDAR sensors for 3D mapping and obstacle detection, depth cameras for visual perception and manipulation, and IMUs for orientation and motion tracking. We'll cover implementation in both Gazebo and Unity environments, providing comprehensive coverage of sensor simulation techniques.

## LiDAR Sensor Simulation

### LiDAR Fundamentals

LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time-of-flight to determine distances to objects. For humanoid robots, LiDAR sensors provide crucial information for:

- Environment mapping and localization
- Obstacle detection and avoidance
- Path planning and navigation
- Safety monitoring around humans

### LiDAR Simulation in Gazebo

#### 2D LiDAR Simulation

```xml
<!-- 2D LiDAR sensor configuration in URDF/SDF -->
<gazebo reference="laser_link">
  <sensor name="laser_2d" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>  <!-- 720 samples = 0.5 degree resolution over 360 degrees -->
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>  <!-- -π radians -->
          <max_angle>3.14159</max_angle>    <!-- π radians -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>      <!-- 0.1m minimum range -->
        <max>30.0</max>     <!-- 30m maximum range -->
        <resolution>0.01</resolution>  <!-- 1cm resolution -->
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>laser_link</frame_name>
    </plugin>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
    </noise>
  </sensor>
</gazebo>
```

#### 3D LiDAR Simulation (Velodyne-style)

```xml
<!-- 3D LiDAR sensor configuration -->
<gazebo reference="velodyne_link">
  <sensor name="velodyne_vlp16" type="ray">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>  <!-- High resolution horizontal scan -->
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>16</samples>    <!-- 16 vertical channels for VLP-16 -->
          <resolution>1</resolution>
          <min_angle>-0.2618</min_angle>  <!-- -15 degrees -->
          <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.3</min>
        <max>100.0</max>
        <resolution>0.001</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=velodyne_points</remapping>
      </ros>
      <output_type>sensor_msgs/PointCloud2</output_type>
      <frame_name>velodyne_link</frame_name>
      <min_range>0.3</min_range>
      <max_range>100.0</max_range>
      <gaussian_noise>0.008</gaussian_noise>
    </plugin>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.008</stddev>
    </noise>
  </sensor>
</gazebo>
```

### LiDAR Simulation in Unity

#### Unity LiDAR Simulation Implementation

```csharp
// Unity LiDAR simulation using raycasting
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class UnityLidarSimulation : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int horizontalRays = 360;      // Number of horizontal rays (0.5 degree resolution)
    public int verticalRays = 16;         // Number of vertical rays (for 3D LiDAR)
    public float minAngle = -Mathf.PI;    // -180 degrees
    public float maxAngle = Mathf.PI;     // 180 degrees
    public float verticalMinAngle = -15 * Mathf.Deg2Rad;
    public float verticalMaxAngle = 15 * Mathf.Deg2Rad;
    public float maxRange = 30.0f;
    public float minRange = 0.1f;
    public LayerMask detectionLayers = -1; // All layers

    [Header("Noise Configuration")]
    public float rangeNoiseStdDev = 0.01f; // 1cm standard deviation
    public float angularNoiseStdDev = 0.001f; // Angular noise

    [Header("ROS Integration")]
    public string topicName = "/scan";
    public string frameId = "laser_link";

    private ROSConnection ros;
    private float[] ranges;
    private float[] intensities;
    private bool is3DLidar = false;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        if (verticalRays > 1)
        {
            is3DLidar = true;
            ranges = new float[horizontalRays * verticalRays];
            intensities = new float[horizontalRays * verticalRays];
        }
        else
        {
            ranges = new float[horizontalRays];
            intensities = new float[horizontalRays];
        }

        // Register publisher
        ros.RegisterPublisher<LaserScanMsg>(topicName);
    }

    void Update()
    {
        if (is3DLidar)
            Simulate3DLidar();
        else
            Simulate2DLidar();

        PublishLaserScan();
    }

    void Simulate2DLidar()
    {
        float angleIncrement = (maxAngle - minAngle) / horizontalRays;

        for (int i = 0; i < horizontalRays; i++)
        {
            float angle = minAngle + i * angleIncrement;

            // Add angular noise
            angle += RandomGaussian(0, angularNoiseStdDev);

            // Calculate ray direction
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange, detectionLayers))
            {
                float distance = hit.distance;

                // Add range noise
                distance += RandomGaussian(0, rangeNoiseStdDev);

                // Apply range limits
                if (distance < minRange)
                    ranges[i] = 0; // Invalid measurement
                else if (distance > maxRange)
                    ranges[i] = float.PositiveInfinity; // Max range
                else
                    ranges[i] = distance;
            }
            else
            {
                ranges[i] = float.PositiveInfinity; // No obstacle detected
            }

            intensities[i] = 1000.0f; // Simulated intensity
        }
    }

    void Simulate3DLidar()
    {
        float horizontalIncrement = (maxAngle - minAngle) / horizontalRays;
        float verticalIncrement = (verticalMaxAngle - verticalMinAngle) / verticalRays;

        int index = 0;
        for (int v = 0; v < verticalRays; v++)
        {
            float verticalAngle = verticalMinAngle + v * verticalIncrement;
            float verticalCos = Mathf.Cos(verticalAngle);
            float verticalSin = Mathf.Sin(verticalAngle);

            for (int h = 0; h < horizontalRays; h++)
            {
                float horizontalAngle = minAngle + h * horizontalIncrement;

                // Add noise
                horizontalAngle += RandomGaussian(0, angularNoiseStdDev);
                verticalAngle += RandomGaussian(0, angularNoiseStdDev * 0.1f); // Less vertical noise

                // Calculate 3D ray direction
                Vector3 direction = new Vector3(
                    verticalCos * Mathf.Cos(horizontalAngle),
                    verticalSin,
                    verticalCos * Mathf.Sin(horizontalAngle)
                );
                direction = transform.TransformDirection(direction);

                // Perform raycast
                RaycastHit hit;
                if (Physics.Raycast(transform.position, direction, out hit, maxRange, detectionLayers))
                {
                    float distance = hit.distance;

                    // Add range noise
                    distance += RandomGaussian(0, rangeNoiseStdDev);

                    if (distance < minRange)
                        ranges[index] = 0;
                    else if (distance > maxRange)
                        ranges[index] = float.PositiveInfinity;
                    else
                        ranges[index] = distance;
                }
                else
                {
                    ranges[index] = float.PositiveInfinity;
                }

                intensities[index] = 1000.0f;
                index++;
            }
        }
    }

    void PublishLaserScan()
    {
        var laserScan = new LaserScanMsg();
        laserScan.header = new HeaderMsg();
        laserScan.header.stamp = new TimeMsg(0, 0); // Will be filled by ROS connector
        laserScan.header.frame_id = frameId;

        laserScan.angle_min = minAngle;
        laserScan.angle_max = maxAngle;
        laserScan.angle_increment = (maxAngle - minAngle) / horizontalRays;
        laserScan.time_increment = 0.0f; // Not used in simulation
        laserScan.scan_time = 0.1f; // 10Hz
        laserScan.range_min = minRange;
        laserScan.range_max = maxRange;

        laserScan.ranges = System.Array.ConvertAll(ranges, x => (float)x);
        laserScan.intensities = System.Array.ConvertAll(intensities, x => (float)x);

        ros.Publish(topicName, laserScan);
    }

    float RandomGaussian(float mean, float stdDev)
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

## Depth Camera Simulation

### Depth Camera Fundamentals

Depth cameras provide both color and depth information, essential for humanoid robots to understand their 3D environment. They enable:

- 3D object recognition and manipulation
- Safe navigation around obstacles
- Human detection and interaction
- Scene understanding for planning

### Depth Camera Simulation in Gazebo

```xml
<!-- Depth camera configuration -->
<gazebo reference="camera_link">
  <sensor name="depth_camera" type="depth">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera name="depth_cam">
      <horizontal_fov>1.0472</horizontal_fov>  <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/depth/image_raw:=depth/image_raw</remapping>
        <remapping>~/rgb/image_raw:=rgb/image_raw</remapping>
        <remapping>~/depth/camera_info:=depth/camera_info</remapping>
      </ros>
      <output_type>sensor_msgs/Image</output_type>
      <frame_name>camera_link</frame_name>
      <baseline>0.2</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <point_cloud_cutoff>0.3</point_cloud_cutoff>
      <point_cloud_cutoff_max>5.0</point_cloud_cutoff_max>
      <Cx_prime>0</Cx_prime>
      <Cx>320.5</Cx>
      <Cy>240.5</Cy>
      <focal_length>320.0</focal_length>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera Simulation in Unity

```csharp
// Unity depth camera simulation
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;

public class UnityDepthCameraSimulation : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera depthCamera;
    public int width = 640;
    public int height = 480;
    public float maxRange = 10.0f;
    public float minRange = 0.1f;

    [Header("Noise Configuration")]
    public float depthNoiseStdDev = 0.01f;

    [Header("ROS Integration")]
    public string imageTopic = "/camera/rgb/image_raw";
    public string depthTopic = "/camera/depth/image_raw";
    public string infoTopic = "/camera/camera_info";
    public string frameId = "camera_link";

    private RenderTexture depthTexture;
    private Texture2D depthTexture2D;
    private ROSConnection ros;
    private CameraInfoMsg cameraInfo;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Create depth render texture
        depthTexture = new RenderTexture(width, height, 24, RenderTextureFormat.Depth);
        depthTexture.Create();

        // Create texture for reading
        depthTexture2D = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Setup camera if not assigned
        if (depthCamera == null)
            depthCamera = GetComponent<Camera>();

        if (depthCamera == null)
        {
            depthCamera = gameObject.AddComponent<Camera>();
            depthCamera.fieldOfView = 60.0f;
        }

        // Create camera info message
        SetupCameraInfo();

        // Register publishers
        ros.RegisterPublisher<ImageMsg>(imageTopic);
        ros.RegisterPublisher<ImageMsg>(depthTopic);
        ros.RegisterPublisher<CameraInfoMsg>(infoTopic);
    }

    void Update()
    {
        // Render depth to texture
        RenderDepthToTexture();

        // Publish camera data
        PublishCameraData();
    }

    void RenderDepthToTexture()
    {
        // Set the camera to render to our depth texture
        depthCamera.targetTexture = depthTexture;
        depthCamera.Render();

        // Read the depth texture
        RenderTexture.active = depthTexture;
        depthTexture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        depthTexture2D.Apply();

        // Reset render texture
        depthCamera.targetTexture = null;
        RenderTexture.active = null;
    }

    void PublishCameraData()
    {
        // Publish RGB image (simulated)
        var rgbImage = CreateSimulatedRGBImage();
        ros.Publish(imageTopic, rgbImage);

        // Publish depth image
        var depthImage = CreateDepthImage();
        ros.Publish(depthTopic, depthImage);

        // Publish camera info
        cameraInfo.header.stamp = new TimeMsg(0, 0);
        ros.Publish(infoTopic, cameraInfo);
    }

    ImageMsg CreateSimulatedRGBImage()
    {
        // Create a simulated RGB image
        Texture2D rgbTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color[] pixels = new Color[width * height];

        // Fill with some simulated scene
        for (int i = 0; i < pixels.Length; i++)
        {
            // Simulate a simple scene with some objects
            float x = (i % width) / (float)width;
            float y = (i / width) / (float)height;

            // Simulate a simple gradient or objects
            pixels[i] = new Color(x, y, 0.5f);
        }

        rgbTexture.SetPixels(pixels);
        rgbTexture.Apply();

        byte[] imageData = rgbTexture.EncodeToPNG();
        Destroy(rgbTexture);

        var image = new ImageMsg();
        image.header = new HeaderMsg();
        image.header.frame_id = frameId;
        image.height = (uint)height;
        image.width = (uint)width;
        image.encoding = "rgb8";
        image.is_bigendian = 0;
        image.step = (uint)(width * 3); // 3 bytes per pixel (RGB)
        image.data = imageData;

        return image;
    }

    ImageMsg CreateDepthImage()
    {
        // Extract depth information from the depth texture
        Color[] depthPixels = depthTexture2D.GetPixels();
        float[] depthValues = new float[width * height];

        // Convert depth texture to actual depth values
        for (int i = 0; i < depthPixels.Length; i++)
        {
            // The depth texture contains depth information in the color values
            // This is a simplified approach - in practice, you'd use a more sophisticated method
            float rawDepth = depthPixels[i].r; // Simplified depth reading
            float actualDepth = rawDepth * maxRange; // Scale to actual range

            // Add noise
            actualDepth += RandomGaussian(0, depthNoiseStdDev);

            // Apply range limits
            if (actualDepth < minRange)
                actualDepth = 0; // Invalid measurement
            else if (actualDepth > maxRange)
                actualDepth = maxRange;

            depthValues[i] = actualDepth;
        }

        // Convert to bytes for ROS message
        byte[] depthData = new byte[depthValues.Length * sizeof(float)];
        for (int i = 0; i < depthValues.Length; i++)
        {
            byte[] floatBytes = System.BitConverter.GetBytes(depthValues[i]);
            System.Array.Copy(floatBytes, 0, depthData, i * sizeof(float), sizeof(float));
        }

        var depthImage = new ImageMsg();
        depthImage.header = new HeaderMsg();
        depthImage.header.frame_id = frameId;
        depthImage.height = (uint)height;
        depthImage.width = (uint)width;
        depthImage.encoding = "32FC1"; // 32-bit float, single channel
        depthImage.is_bigendian = 0;
        depthImage.step = (uint)(width * sizeof(float));
        depthImage.data = depthData;

        return depthImage;
    }

    void SetupCameraInfo()
    {
        cameraInfo = new CameraInfoMsg();
        cameraInfo.header = new HeaderMsg();
        cameraInfo.height = (uint)height;
        cameraInfo.width = (uint)width;
        cameraInfo.distortion_model = "plumb_bob";

        // Camera intrinsic parameters (for 60 degree FOV, 640x480)
        cameraInfo.d = new double[5] { 0, 0, 0, 0, 0 }; // No distortion
        cameraInfo.k = new double[9] {
            320.0, 0.0, 320.0,    // fx, 0, cx
            0.0, 320.0, 240.0,    // 0, fy, cy
            0.0, 0.0, 1.0         // 0, 0, 1
        };
        cameraInfo.r = new double[9] {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        cameraInfo.p = new double[12] {
            320.0, 0.0, 320.0, 0,   // [fx' 0 cx' Tx]
            0.0, 320.0, 240.0, 0,   // [0 fy' cy' Ty]
            0.0, 0.0, 1.0, 0        // [0 0 1 0]
        };
    }

    float RandomGaussian(float mean, float stdDev)
    {
        float u1 = Random.value;
        float u2 = Random.value;
        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }

    void OnDestroy()
    {
        if (depthTexture != null)
            depthTexture.Release();
        if (depthTexture2D != null)
            DestroyImmediate(depthTexture2D);
    }
}
```

## IMU Sensor Simulation

### IMU Fundamentals

Inertial Measurement Units (IMUs) provide crucial information about a robot's orientation, angular velocity, and linear acceleration. For humanoid robots, IMUs are essential for:

- Balance and posture control
- Motion tracking and navigation
- Fall detection and recovery
- Orientation estimation

### IMU Simulation in Gazebo

```xml
<!-- IMU sensor configuration -->
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
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.00017</bias_stddev>  <!-- ~0.01 deg/s bias -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.00017</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.00017</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>  <!-- ~0.017 m/s² stddev -->
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.0017</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.0017</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.0</bias_mean>
            <bias_stddev>0.0017</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <output_type>sensor_msgs/Imu</output_type>
      <frame_name>imu_link</frame_name>
      <topic_name>imu/data</topic_name>
      <gaussian_noise>0.0017</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation in Unity

```csharp
// Unity IMU simulation
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;

public class UnityImuSimulation : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float updateRate = 100.0f;  // 100Hz
    public float angularVelocityNoiseStdDev = 0.0017f;  // ~0.1 deg/s
    public float linearAccelerationNoiseStdDev = 0.017f; // ~0.017 m/s²
    public float angularVelocityBiasStdDev = 0.00017f;   // ~0.01 deg/s
    public float linearAccelerationBiasStdDev = 0.0017f; // ~0.0017 m/s²

    [Header("Gravity")]
    public Vector3 gravity = new Vector3(0, -9.81f, 0);

    [Header("ROS Integration")]
    public string topicName = "/imu/data";
    public string frameId = "imu_link";

    private ROSConnection ros;
    private float updateInterval;
    private float lastUpdateTime;
    private Vector3 lastAngularVelocity;
    private Vector3 lastLinearAcceleration;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        updateInterval = 1.0f / updateRate;
        lastUpdateTime = Time.time;

        // Register publisher
        ros.RegisterPublisher<ImuMsg>(topicName);

        // Initialize with current transform
        lastAngularVelocity = Vector3.zero;
        lastLinearAcceleration = gravity;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishImuData();
            lastUpdateTime = Time.time;
        }
    }

    void PublishImuData()
    {
        var imuMsg = new ImuMsg();
        imuMsg.header = new HeaderMsg();
        imuMsg.header.stamp = new TimeMsg(0, 0);
        imuMsg.header.frame_id = frameId;

        // Get orientation from transform
        imuMsg.orientation = ConvertQuaternion(transform.rotation);

        // Get angular velocity (simulated from rotation changes)
        Vector3 angularVelocity = EstimateAngularVelocity();
        imuMsg.angular_velocity = AddAngularVelocityNoise(angularVelocity);

        // Get linear acceleration (simulated from movement)
        Vector3 linearAcceleration = EstimateLinearAcceleration();
        imuMsg.linear_acceleration = AddLinearAccelerationNoise(linearAcceleration);

        // Set covariance matrices (diagonal values only, others are zero)
        for (int i = 0; i < 9; i++)
        {
            imuMsg.orientation_covariance[i] = 0.0;
            imuMsg.angular_velocity_covariance[i] = 0.0;
            imuMsg.linear_acceleration_covariance[i] = 0.0;
        }

        // Set diagonal values for orientation covariance (unknown, so large)
        imuMsg.orientation_covariance[0] = -1; // -1 indicates unknown

        // Set diagonal values for angular velocity covariance
        float angularVelCov = angularVelocityNoiseStdDev * angularVelocityNoiseStdDev;
        imuMsg.angular_velocity_covariance[0] = angularVelCov;
        imuMsg.angular_velocity_covariance[4] = angularVelCov;
        imuMsg.angular_velocity_covariance[8] = angularVelCov;

        // Set diagonal values for linear acceleration covariance
        float linearAccCov = linearAccelerationNoiseStdDev * linearAccelerationNoiseStdDev;
        imuMsg.linear_acceleration_covariance[0] = linearAccCov;
        imuMsg.linear_acceleration_covariance[4] = linearAccCov;
        imuMsg.linear_acceleration_covariance[8] = linearAccCov;

        ros.Publish(topicName, imuMsg);
    }

    Vector3 EstimateAngularVelocity()
    {
        // Estimate angular velocity from rotation changes
        // This is a simplified approach - in practice, you'd use more sophisticated methods
        if (Time.deltaTime > 0)
        {
            // Calculate angular velocity based on rotation change
            // For now, return a combination of physics and transform changes
            Rigidbody rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                // Use physics-based angular velocity if available
                return new Vector3(
                    rb.angularVelocity.x,
                    rb.angularVelocity.y,
                    rb.angularVelocity.z
                );
            }
            else
            {
                // Estimate from transform changes
                return lastAngularVelocity; // Use previous value as approximation
            }
        }
        return Vector3.zero;
    }

    Vector3 EstimateLinearAcceleration()
    {
        // Estimate linear acceleration from movement
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // Remove gravity from the measured acceleration
            return new Vector3(
                rb.acceleration.x,
                rb.acceleration.y + 9.81f, // Add gravity back since IMU measures this
                rb.acceleration.z
            );
        }
        else
        {
            // Estimate from transform changes
            return lastLinearAcceleration;
        }
    }

    Vector3 AddAngularVelocityNoise(Vector3 angularVelocity)
    {
        // Add Gaussian noise to angular velocity
        return new Vector3(
            angularVelocity.x + RandomGaussian(0, angularVelocityNoiseStdDev),
            angularVelocity.y + RandomGaussian(0, angularVelocityNoiseStdDev),
            angularVelocity.z + RandomGaussian(0, angularVelocityNoiseStdDev)
        );
    }

    Vector3 AddLinearAccelerationNoise(Vector3 linearAcceleration)
    {
        // Add Gaussian noise to linear acceleration
        return new Vector3(
            linearAcceleration.x + RandomGaussian(0, linearAccelerationNoiseStdDev),
            linearAcceleration.y + RandomGaussian(0, linearAccelerationNoiseStdDev),
            linearAcceleration.z + RandomGaussian(0, linearAccelerationNoiseStdDev)
        );
    }

    geometry_msgs.Quaternion ConvertQuaternion(Quaternion q)
    {
        return new geometry_msgs.Quaternion(q.x, q.y, q.z, q.w);
    }

    float RandomGaussian(float mean, float stdDev)
    {
        float u1 = Random.value;
        float u2 = Random.value;
        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

## Sensor Fusion and Integration

### Multi-Sensor Integration

For humanoid robots, effective sensor fusion combines data from multiple sensors:

```csharp
// Multi-sensor fusion system
using UnityEngine;
using System.Collections.Generic;

public class SensorFusionSystem : MonoBehaviour
{
    [Header("Sensor Inputs")]
    public UnityImuSimulation imuSensor;
    public UnityLidarSimulation lidarSensor;
    public UnityDepthCameraSimulation cameraSensor;

    [Header("Fusion Configuration")]
    public float fusionUpdateRate = 50.0f;
    public bool enableEKF = true;  // Extended Kalman Filter

    private float fusionInterval;
    private float lastFusionTime;
    private Vector3 estimatedPosition;
    private Quaternion estimatedOrientation;

    void Start()
    {
        fusionInterval = 1.0f / fusionUpdateRate;
        lastFusionTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastFusionTime >= fusionInterval)
        {
            PerformSensorFusion();
            lastFusionTime = Time.time;
        }
    }

    void PerformSensorFusion()
    {
        // Get sensor readings
        var imuData = GetImuData();
        var lidarData = GetLidarData();
        var cameraData = GetCameraData();

        // Perform sensor fusion (simplified approach)
        if (enableEKF)
        {
            PerformEKF(imuData, lidarData, cameraData);
        }
        else
        {
            // Simple weighted average fusion
            PerformSimpleFusion(imuData, lidarData, cameraData);
        }

        // Publish fused data
        PublishFusedData();
    }

    (Vector3 position, Quaternion orientation, Vector3 velocity) GetImuData()
    {
        // Get IMU-based estimates (this would come from the IMU processing)
        return (transform.position, transform.rotation, Vector3.zero);
    }

    (Vector3[] positions, float[] distances) GetLidarData()
    {
        // Get LiDAR-based position estimates from landmark matching
        return (new Vector3[0], new float[0]);
    }

    (Vector3[] features) GetCameraData()
    {
        // Get camera-based feature positions
        return (new Vector3[0]);
    }

    void PerformEKF(
        (Vector3 position, Quaternion orientation, Vector3 velocity) imuData,
        (Vector3[] positions, float[] distances) lidarData,
        (Vector3[] features) cameraData)
    {
        // Extended Kalman Filter implementation
        // This is a simplified version - a full implementation would be more complex

        // Prediction step (using IMU data)
        var predictedState = PredictState(imuData);

        // Update step (using LiDAR and camera data)
        var updatedState = UpdateState(predictedState, lidarData, cameraData);

        // Store the result
        estimatedPosition = updatedState.position;
        estimatedOrientation = updatedState.orientation;
    }

    (Vector3 position, Quaternion orientation, Vector3 velocity) PredictState(
        (Vector3 position, Quaternion orientation, Vector3 velocity) imuData)
    {
        // Predict next state based on IMU measurements
        float deltaTime = fusionInterval;

        // Integrate acceleration to get velocity, then position
        Vector3 newVelocity = imuData.velocity + imuData.acceleration * deltaTime;
        Vector3 newPosition = imuData.position + newVelocity * deltaTime;

        // Integrate angular velocity to get orientation change
        Vector3 angularVelocity = imuData.angularVelocity;
        float angle = angularVelocity.magnitude * deltaTime;
        Vector3 axis = angularVelocity.normalized;

        Quaternion rotationIncrement = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
        Quaternion newOrientation = imuData.orientation * rotationIncrement;

        return (newPosition, newOrientation, newVelocity);
    }

    (Vector3 position, Quaternion orientation, Vector3 velocity) UpdateState(
        (Vector3 position, Quaternion orientation, Vector3 velocity) predictedState,
        (Vector3[] positions, float[] distances) lidarData,
        (Vector3[] features) cameraData)
    {
        // Update state based on LiDAR and camera measurements
        // This would involve measurement updates and covariance calculations
        return predictedState; // Simplified return
    }

    void PerformSimpleFusion(
        (Vector3 position, Quaternion orientation, Vector3 velocity) imuData,
        (Vector3[] positions, float[] distances) lidarData,
        (Vector3[] features) cameraData)
    {
        // Simple weighted average fusion
        // Weight IMU heavily for orientation, LiDAR for position
        estimatedPosition = imuData.position; // Placeholder
        estimatedOrientation = imuData.orientation; // Placeholder
    }

    void PublishFusedData()
    {
        // Publish the fused state estimate
        // This would send the fused data to ROS or other systems
        Debug.Log($"Fused Position: {estimatedPosition}, Orientation: {estimatedOrientation}");
    }
}
```

## Sensor Validation and Calibration

### Validation Techniques

Validating sensor simulation accuracy is crucial:

```csharp
// Sensor validation system
using UnityEngine;
using System.Collections.Generic;

public class SensorValidationSystem : MonoBehaviour
{
    [Header("Validation Configuration")]
    public float validationInterval = 1.0f;
    public float positionTolerance = 0.05f;  // 5cm tolerance
    public float orientationTolerance = 0.1f; // 0.1 rad tolerance
    public float velocityTolerance = 0.1f;    // 0.1 m/s tolerance

    [Header("Reference Data")]
    public Transform referenceTransform;  // Ground truth position
    public Rigidbody referenceRigidbody;  // Ground truth motion

    private float lastValidationTime;
    private List<SensorValidationResult> validationResults;

    void Start()
    {
        lastValidationTime = Time.time;
        validationResults = new List<SensorValidationResult>();
    }

    void Update()
    {
        if (Time.time - lastValidationTime >= validationInterval)
        {
            PerformValidation();
            lastValidationTime = Time.time;
        }
    }

    void PerformValidation()
    {
        // Get ground truth data
        var groundTruth = GetGroundTruthData();

        // Get sensor estimates
        var sensorEstimates = GetSensorEstimates();

        // Compare and validate
        var validationResult = CompareSensorData(groundTruth, sensorEstimates);
        validationResults.Add(validationResult);

        // Log results
        LogValidationResult(validationResult);

        // Check if validation is failing
        if (!validationResult.passed)
        {
            Debug.LogWarning($"Sensor validation failed at time {Time.time}");
        }
    }

    (Vector3 position, Quaternion orientation, Vector3 velocity) GetGroundTruthData()
    {
        if (referenceTransform != null)
        {
            Vector3 position = referenceTransform.position;
            Quaternion orientation = referenceTransform.rotation;
            Vector3 velocity = Vector3.zero;

            if (referenceRigidbody != null)
            {
                velocity = referenceRigidbody.velocity;
            }

            return (position, orientation, velocity);
        }

        // Fallback to current transform if no reference
        return (transform.position, transform.rotation, Vector3.zero);
    }

    (Vector3 position, Quaternion orientation, Vector3 velocity) GetSensorEstimates()
    {
        // This would get estimates from your sensor fusion system
        // For now, returning current transform as a placeholder
        return (transform.position, transform.rotation, Vector3.zero);
    }

    SensorValidationResult CompareSensorData(
        (Vector3 position, Quaternion orientation, Vector3 velocity) groundTruth,
        (Vector3 position, Quaternion orientation, Vector3 velocity) sensorEstimates)
    {
        float positionError = Vector3.Distance(groundTruth.position, sensorEstimates.position);
        float orientationError = Quaternion.Angle(groundTruth.orientation, sensorEstimates.orientation);
        float velocityError = Vector3.Distance(groundTruth.velocity, sensorEstimates.velocity);

        bool positionValid = positionError <= positionTolerance;
        bool orientationValid = orientationError <= orientationTolerance;
        bool velocityValid = velocityError <= velocityTolerance;

        return new SensorValidationResult
        {
            timestamp = Time.time,
            positionError = positionError,
            orientationError = orientationError,
            velocityError = velocityError,
            positionValid = positionValid,
            orientationValid = orientationValid,
            velocityValid = velocityValid,
            passed = positionValid && orientationValid && velocityValid
        };
    }

    void LogValidationResult(SensorValidationResult result)
    {
        string status = result.passed ? "PASSED" : "FAILED";
        Debug.Log($"Sensor Validation {status} at {result.timestamp:F2}s - " +
                 $"Pos Error: {result.positionError:F3}m, " +
                 $"Ori Error: {result.orientationError:F3}rad, " +
                 $"Vel Error: {result.velocityError:F3}m/s");
    }

    public float GetAveragePositionError()
    {
        if (validationResults.Count == 0) return float.MaxValue;

        float sum = 0;
        foreach (var result in validationResults)
        {
            sum += result.positionError;
        }
        return sum / validationResults.Count;
    }

    public float GetAverageOrientationError()
    {
        if (validationResults.Count == 0) return float.MaxValue;

        float sum = 0;
        foreach (var result in validationResults)
        {
            sum += result.orientationError;
        }
        return sum / validationResults.Count;
    }
}

[System.Serializable]
public class SensorValidationResult
{
    public float timestamp;
    public float positionError;
    public float orientationError;
    public float velocityError;
    public bool positionValid;
    public bool orientationValid;
    public bool velocityValid;
    public bool passed;
}
```

## Performance Optimization for Sensor Simulation

### Efficient Sensor Simulation

Optimizing sensor simulation performance:

```csharp
// Optimized sensor simulation manager
using UnityEngine;
using System.Collections.Generic;

public class OptimizedSensorManager : MonoBehaviour
{
    [Header("Performance Configuration")]
    public int maxRaysPerUpdate = 1000;  // Limit rays per frame for performance
    public float sensorUpdateInterval = 0.01f;  // 100Hz for critical sensors
    public float lessCriticalUpdateInterval = 0.1f;  // 10Hz for less critical

    [Header("Sensor Prioritization")]
    public List<SensorSimulationBase> criticalSensors;  // IMU, safety sensors
    public List<SensorSimulationBase> standardSensors;  // LiDAR, cameras
    public List<SensorSimulationBase> lowPrioritySensors;  // Environmental sensors

    private float lastCriticalUpdate;
    private float lastStandardUpdate;
    private float lastLowPriorityUpdate;

    void Start()
    {
        lastCriticalUpdate = Time.time;
        lastStandardUpdate = Time.time;
        lastLowPriorityUpdate = Time.time;
    }

    void Update()
    {
        // Update critical sensors at high frequency
        if (Time.time - lastCriticalUpdate >= sensorUpdateInterval)
        {
            UpdateSensors(criticalSensors);
            lastCriticalUpdate = Time.time;
        }

        // Update standard sensors at medium frequency
        if (Time.time - lastStandardUpdate >= sensorUpdateInterval * 3) // 33Hz
        {
            UpdateSensors(standardSensors);
            lastStandardUpdate = Time.time;
        }

        // Update low priority sensors at lower frequency
        if (Time.time - lastLowPriorityUpdate >= lessCriticalUpdateInterval)
        {
            UpdateSensors(lowPrioritySensors);
            lastLowPriorityUpdate = Time.time;
        }
    }

    void UpdateSensors(List<SensorSimulationBase> sensors)
    {
        if (sensors == null) return;

        int raysThisFrame = 0;
        foreach (var sensor in sensors)
        {
            if (sensor != null && sensor.enabled)
            {
                int raysNeeded = sensor.GetRayCount();

                if (raysThisFrame + raysNeeded <= maxRaysPerUpdate)
                {
                    sensor.UpdateSensor();
                    raysThisFrame += raysNeeded;
                }
                else
                {
                    // Skip sensor update to stay within ray budget
                    sensor.SkipUpdate();
                }
            }
        }
    }

    // Dynamic resolution adjustment based on performance
    public void AdjustSensorResolution(float performanceFactor)
    {
        // performanceFactor: 0.0 = lowest quality, 1.0 = highest quality
        foreach (var sensor in criticalSensors)
        {
            sensor.SetQuality(Mathf.Max(0.3f, performanceFactor)); // Critical sensors maintain min quality
        }

        foreach (var sensor in standardSensors)
        {
            sensor.SetQuality(performanceFactor);
        }

        foreach (var sensor in lowPrioritySensors)
        {
            sensor.SetQuality(performanceFactor * 0.7f); // Lower priority sensors get reduced quality
        }
    }
}

// Base class for sensor simulation
public abstract class SensorSimulationBase : MonoBehaviour
{
    public bool enabled = true;

    public abstract void UpdateSensor();
    public abstract int GetRayCount();
    public abstract void SkipUpdate();
    public abstract void SetQuality(float quality);  // 0.0 to 1.0
}

// Example implementation for LiDAR
public class OptimizedLidarSimulation : SensorSimulationBase
{
    [Header("Optimization Settings")]
    public int baseRayCount = 720;
    public float quality = 1.0f;
    public int currentRayCount;

    void Start()
    {
        UpdateRayCount();
    }

    public override void UpdateSensor()
    {
        // Perform LiDAR simulation with current ray count
        SimulateLidarRays(currentRayCount);
    }

    public override int GetRayCount()
    {
        return currentRayCount;
    }

    public override void SkipUpdate()
    {
        // Skip this update cycle
        Debug.Log("LiDAR update skipped for performance");
    }

    public override void SetQuality(float quality)
    {
        this.quality = Mathf.Clamp(quality, 0.1f, 1.0f);
        UpdateRayCount();
    }

    void UpdateRayCount()
    {
        currentRayCount = Mathf.RoundToInt(baseRayCount * quality);
        currentRayCount = Mathf.Max(1, currentRayCount); // Ensure at least 1 ray
    }

    void SimulateLidarRays(int rayCount)
    {
        // Simulate LiDAR with specified ray count
        float angleStep = 2 * Mathf.PI / rayCount;

        for (int i = 0; i < rayCount; i++)
        {
            float angle = i * angleStep;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, 30.0f))
            {
                // Process hit
            }
        }
    }
}
```

## Practical Example: Complete Sensor Suite for Humanoid Robot

Here's a complete example of a sensor suite for a humanoid robot:

```xml
<!-- Complete humanoid robot with multiple sensors -->
<robot name="humanoid_with_sensors" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="50.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="5.0" ixy="0.0" ixz="0.0" iyy="5.0" iyz="0.0" izz="3.0"/>
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

  <!-- IMU sensor on torso -->
  <link name="imu_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>

  <!-- 2D LiDAR on head -->
  <link name="laser_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
    </visual>
  </link>

  <joint name="laser_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_link"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
  </joint>

  <!-- RGB-D camera -->
  <link name="camera_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.05 0.1 0.03"/>
      </geometry>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0.95" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.0017</stddev>
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
              <stddev>0.017</stddev>
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
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>~/out:=imu/data</remapping>
        </ros>
        <frame_name>imu_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="laser_link">
    <sensor name="laser_2d" type="ray">
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <frame_name>laser_link</frame_name>
      </plugin>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </sensor>
  </gazebo>

  <gazebo reference="camera_link">
    <sensor name="depth_camera" type="depth">
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera name="depth_cam">
        <horizontal_fov>1.0472</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>~/depth/image_raw:=camera/depth/image_raw</remapping>
          <remapping>~/rgb/image_raw:=camera/rgb/image_raw</remapping>
          <remapping>~/depth/camera_info:=camera/depth/camera_info</remapping>
        </ros>
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Exercises

1. **LiDAR Simulation Challenge**: Create a LiDAR simulation in both Gazebo and Unity that can detect and map a complex indoor environment with furniture, walls, and moving objects. Compare the point clouds generated by both simulators.

2. **Sensor Fusion Project**: Implement a sensor fusion system that combines IMU, LiDAR, and camera data to estimate robot pose in a Unity environment. Validate the fused estimates against ground truth data.

3. **Performance Optimization Task**: Create a sensor simulation system with adjustable quality settings that can maintain real-time performance on different hardware configurations while preserving sensor accuracy.

## Summary

This chapter covered comprehensive sensor simulation techniques for humanoid robotics:

- LiDAR simulation with realistic noise models and 2D/3D configurations
- Depth camera simulation with proper calibration and noise characteristics
- IMU simulation with accurate dynamics and bias modeling
- Sensor fusion techniques for combining multiple sensor modalities
- Validation and calibration methods for ensuring simulation accuracy
- Performance optimization strategies for efficient sensor simulation

Accurate sensor simulation is crucial for developing robust perception and navigation systems for humanoid robots. The techniques covered in this chapter provide the foundation for creating realistic digital twin environments that can effectively bridge the gap between simulation and real-world deployment.

## Next Steps

With Module 2 complete, you now have a comprehensive understanding of digital twin technologies for humanoid robotics:
- Physics simulation and environment building
- Gazebo-based simulation with advanced physics
- Unity-based high-fidelity rendering and interaction
- Realistic sensor simulation for perception systems

In the next module, we'll explore AI perception and navigation systems with NVIDIA Isaac, building upon these simulation foundations to create intelligent robotic systems.