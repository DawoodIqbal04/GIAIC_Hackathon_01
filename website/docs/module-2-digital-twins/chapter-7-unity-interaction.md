---
id: chapter-7-unity-interaction
title: "Chapter 7: High-Fidelity Rendering and Human-Robot Interaction in Unity"
sidebar_label: "Chapter 7: High-Fidelity Rendering and Human-Robot Interaction in Unity"
description: "Advanced rendering and human-robot interaction techniques in Unity for humanoid robotics simulation"
keywords: [unity, rendering, interaction, humanoid, robotics, simulation, vr, ar]
tags: [unity, rendering, interaction]
authors: [book-authors]
difficulty: advanced
estimated_time: "105 minutes"
module: 2
chapter: 7
prerequisites: [unity-basics, csharp-programming, ros2-foundations, physics-simulation]
learning_objectives:
  - Implement high-fidelity rendering techniques in Unity for humanoid robots
  - Design intuitive human-robot interaction systems in Unity
  - Integrate Unity with ROS 2 for bidirectional communication
  - Create immersive VR/AR experiences for robot teleoperation
  - Develop realistic material and lighting systems for robotics
related:
  - next: chapter-8-sensor-simulation
  - previous: chapter-6-gazebo-simulations
  - see_also: [chapter-6-gazebo-simulations, ../module-1-ros-foundations/chapter-4-urdf-humanoids]
---

# Chapter 7: High-Fidelity Rendering and Human-Robot Interaction in Unity

## Learning Objectives

After completing this chapter, you will be able to:
- Implement advanced rendering techniques for realistic humanoid robot visualization
- Design intuitive human-robot interaction systems in Unity
- Integrate Unity with ROS 2 for bidirectional communication
- Create immersive VR/AR experiences for robot teleoperation and monitoring
- Develop realistic material and lighting systems for robotics applications

## Introduction

Unity has emerged as a powerful platform for high-fidelity simulation and visualization in robotics, particularly for humanoid robot applications. Unlike Gazebo's physics-focused approach, Unity excels in visual realism, human-robot interaction, and immersive experiences. Its advanced rendering capabilities, extensive asset ecosystem, and VR/AR support make it ideal for creating photorealistic environments and intuitive interfaces for robot teleoperation and monitoring.

This chapter explores how to leverage Unity's capabilities for humanoid robotics applications, focusing on high-fidelity rendering, realistic human-robot interaction, and seamless integration with ROS 2 systems. We'll cover advanced rendering techniques, interaction design principles, and practical implementation strategies for creating compelling and functional robot simulation environments.

## Unity for Robotics: Architecture and Capabilities

### Unity Robotics Ecosystem

Unity provides several specialized packages and tools for robotics:

1. **Unity Robotics Hub**: Centralized access to robotics packages and samples
2. **Unity Robotics Package (URP)**: ROS communication bridge
3. **ML-Agents**: Machine learning framework for robot training
4. **XR packages**: Virtual and augmented reality support
5. **ProBuilder/ProGrids**: Rapid environment prototyping tools

### Comparison with Gazebo

| Aspect | Unity | Gazebo |
|--------|-------|--------|
| Rendering Quality | High-fidelity, photorealistic | Basic, functional |
| Physics Simulation | Good (NVIDIA PhysX) | Excellent (specialized) |
| Human Interaction | Excellent (UI, VR, AR) | Basic (GUI) |
| ROS Integration | Good (ROS TCP Connector) | Excellent (native) |
| Asset Creation | Extensive (Asset Store) | Limited (models database) |
| Performance | High (GPU accelerated) | Moderate (CPU focused) |

## Advanced Rendering Techniques for Humanoid Robots

### Physically-Based Rendering (PBR)

Unity's PBR pipeline creates realistic materials for robot components:

```csharp
// Example PBR material setup for robot components
using UnityEngine;

public class RobotMaterialSetup : MonoBehaviour
{
    [Header("Material Properties")]
    public float metalness = 0.8f;      // Metallic surfaces like joints
    public float smoothness = 0.7f;     // Smoothness for polished surfaces
    public float emission = 0.0f;       // Emission for LED indicators
    public Color baseColor = Color.gray;

    [Header("Robot Specific Properties")]
    public bool hasLEDIndicators = true;
    public Color ledColor = Color.green;

    void Start()
    {
        SetupRobotMaterials();
    }

    void SetupRobotMaterials()
    {
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            Material material = renderer.material;

            // Set PBR properties
            material.SetFloat("_Metallic", metalness);
            material.SetFloat("_Smoothness", smoothness);
            material.SetColor("_Color", baseColor);

            // Configure emission for LED indicators
            if (hasLEDIndicators)
            {
                material.SetColor("_EmissionColor", ledColor);
                material.EnableKeyword("_EMISSION");
            }

            // Set up normal maps for surface details
            SetupSurfaceDetails(material);
        }
    }

    void SetupSurfaceDetails(Material material)
    {
        // Add normal maps for surface imperfections
        // Add occlusion maps for shadowing details
        // Configure anisotropic reflections for brushed metal
    }
}
```

### Realistic Lighting for Robotics

Proper lighting is crucial for realistic robot visualization:

```csharp
// Advanced lighting setup for robotics environments
using UnityEngine;

public class RoboticsLightingSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light mainLight;
    public Light fillLight;
    public Light rimLight;
    public bool useHDRP = false;  // High Definition Render Pipeline

    [Header("Environment Lighting")]
    public float environmentIntensity = 1.0f;
    public Color environmentColor = Color.white;
    public Cubemap reflectionCubemap;

    void Start()
    {
        SetupLighting();
    }

    void SetupLighting()
    {
        // Configure main light (key light)
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.intensity = 2.0f;
            mainLight.color = Color.white;
            mainLight.shadows = LightShadows.Soft;
            mainLight.shadowStrength = 0.8f;
        }

        // Configure fill light
        if (fillLight != null)
        {
            fillLight.type = LightType.Directional;
            fillLight.intensity = 0.5f;
            fillLight.color = Color.gray;
            fillLight.shadows = LightShadows.None;
        }

        // Configure rim light for depth
        if (rimLight != null)
        {
            rimLight.type = LightType.Spot;
            rimLight.intensity = 0.8f;
            rimLight.color = Color.white;
            rimLight.spotAngle = 45f;
            rimLight.shadows = LightShadows.Soft;
        }

        // Set up environment lighting
        RenderSettings.ambientIntensity = environmentIntensity;
        RenderSettings.ambientLight = environmentColor;

        if (reflectionCubemap != null)
        {
            RenderSettings.defaultReflectionMode = UnityEngine.Rendering.DefaultReflectionMode.Custom;
            RenderSettings.customReflection = reflectionCubemap;
        }
    }

    // Dynamic lighting based on robot state
    public void UpdateLightingForRobotState(string robotState)
    {
        switch (robotState)
        {
            case "active":
                mainLight.intensity = 2.0f;
                break;
            case "standby":
                mainLight.intensity = 1.0f;
                break;
            case "error":
                mainLight.color = Color.red;
                break;
            default:
                mainLight.intensity = 2.0f;
                mainLight.color = Color.white;
                break;
        }
    }
}
```

### Particle Systems for Robot Effects

Visual effects enhance the realism of robot operations:

```csharp
// Particle system for robot effects
using UnityEngine;

public class RobotEffects : MonoBehaviour
{
    [Header("Dust Effects")]
    public ParticleSystem dustSystem;
    public float dustIntensity = 1.0f;

    [Header("Heat Effects")]
    public ParticleSystem heatSystem;
    public float heatIntensity = 0.5f;

    [Header("Status Indicators")]
    public ParticleSystem statusSystem;

    [Header("Interaction Effects")]
    public ParticleSystem interactionSystem;

    void Start()
    {
        SetupParticleSystems();
    }

    void SetupParticleSystems()
    {
        // Configure dust particles for walking robots
        if (dustSystem != null)
        {
            var dustMain = dustSystem.main;
            dustMain.startLifetime = 2.0f;
            dustMain.startSpeed = 0.5f;
            dustMain.startSize = 0.1f;
            dustMain.maxParticles = 1000;
        }

        // Configure heat shimmer for active components
        if (heatSystem != null)
        {
            var heatMain = heatSystem.main;
            heatMain.startLifetime = 1.0f;
            heatMain.startSpeed = 0.1f;
            heatMain.startColor = new Color(1.0f, 0.5f, 0.0f, 0.3f); // Orange transparent
            heatMain.maxParticles = 500;
        }

        // Configure status indicator particles
        if (statusSystem != null)
        {
            var statusMain = statusSystem.main;
            statusMain.startLifetime = 5.0f;
            statusMain.startSpeed = 0.0f;
            statusMain.startSize = 0.05f;
        }

        // Configure interaction particles
        if (interactionSystem != null)
        {
            var interactionMain = interactionSystem.main;
            interactionMain.startLifetime = 0.5f;
            interactionMain.startSpeed = 1.0f;
            interactionMain.startSize = 0.02f;
        }
    }

    public void TriggerDustEffect(Vector3 position)
    {
        if (dustSystem != null)
        {
            dustSystem.transform.position = position;
            dustSystem.Emit((int)(dustIntensity * 10));
        }
    }

    public void TriggerInteractionEffect(Vector3 position, Color color)
    {
        if (interactionSystem != null)
        {
            var main = interactionSystem.main;
            main.startColor = color;

            interactionSystem.transform.position = position;
            interactionSystem.Emit(20);
        }
    }
}
```

## Human-Robot Interaction Design in Unity

### User Interface Design for Robot Control

Creating intuitive interfaces for robot operation:

```csharp
// Robot control interface
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class RobotControlInterface : MonoBehaviour
{
    [Header("Control Elements")]
    public Slider linearVelocitySlider;
    public Slider angularVelocitySlider;
    public Button moveButton;
    public Button stopButton;
    public Button resetButton;

    [Header("Status Display")]
    public TextMeshProUGUI statusText;
    public TextMeshProUGUI positionText;
    public TextMeshProUGUI batteryText;
    public Image batteryBar;

    [Header("Command Queue")]
    public Button queueCommandButton;
    public Button executeQueueButton;
    public Button clearQueueButton;

    [Header("Camera Controls")]
    public Button firstPersonButton;
    public Button thirdPersonButton;
    public Button topDownButton;

    void Start()
    {
        SetupEventHandlers();
        UpdateStatusDisplay();
    }

    void SetupEventHandlers()
    {
        // Velocity controls
        linearVelocitySlider.onValueChanged.AddListener(OnLinearVelocityChanged);
        angularVelocitySlider.onValueChanged.AddListener(OnAngularVelocityChanged);

        // Action buttons
        moveButton.onClick.AddListener(OnMoveButtonClicked);
        stopButton.onClick.AddListener(OnStopButtonClicked);
        resetButton.onClick.AddListener(OnResetButtonClicked);

        // Queue controls
        queueCommandButton.onClick.AddListener(OnQueueCommandClicked);
        executeQueueButton.onClick.AddListener(OnExecuteQueueClicked);
        clearQueueButton.onClick.AddListener(OnClearQueueClicked);

        // Camera controls
        firstPersonButton.onClick.AddListener(() => SwitchCameraView("first"));
        thirdPersonButton.onClick.AddListener(() => SwitchCameraView("third"));
        topDownButton.onClick.AddListener(() => SwitchCameraView("top"));
    }

    void OnLinearVelocityChanged(float value)
    {
        // Update robot linear velocity
        Debug.Log($"Linear velocity set to: {value}");
    }

    void OnAngularVelocityChanged(float value)
    {
        // Update robot angular velocity
        Debug.Log($"Angular velocity set to: {value}");
    }

    void OnMoveButtonClicked()
    {
        // Send move command to robot
        Debug.Log("Move command sent");
        UpdateStatus("Moving");
    }

    void OnStopButtonClicked()
    {
        // Send stop command to robot
        Debug.Log("Stop command sent");
        UpdateStatus("Stopped");
    }

    void OnResetButtonClicked()
    {
        // Send reset command to robot
        Debug.Log("Reset command sent");
        UpdateStatus("Resetting");
    }

    void OnQueueCommandClicked()
    {
        // Add command to queue
        Debug.Log("Command queued");
    }

    void OnExecuteQueueClicked()
    {
        // Execute command queue
        Debug.Log("Executing command queue");
    }

    void OnClearQueueClicked()
    {
        // Clear command queue
        Debug.Log("Command queue cleared");
    }

    void SwitchCameraView(string view)
    {
        // Switch camera perspective
        Debug.Log($"Switched to {view} person view");
    }

    public void UpdateStatus(string status)
    {
        statusText.text = $"Status: {status}";
    }

    public void UpdatePosition(Vector3 position, Vector3 rotation)
    {
        positionText.text = $"Position: X:{position.x:F2} Y:{position.y:F2} Z:{position.z:F2}";
    }

    public void UpdateBattery(float batteryLevel)
    {
        batteryText.text = $"Battery: {(int)(batteryLevel * 100)}%";
        batteryBar.fillAmount = batteryLevel;

        // Change color based on battery level
        if (batteryLevel < 0.2f)
        {
            batteryBar.color = Color.red;
        }
        else if (batteryLevel < 0.5f)
        {
            batteryBar.color = Color.yellow;
        }
        else
        {
            batteryBar.color = Color.green;
        }
    }

    void UpdateStatusDisplay()
    {
        UpdateStatus("Ready");
        UpdatePosition(Vector3.zero, Vector3.zero);
        UpdateBattery(1.0f);
    }
}
```

### Gesture and Voice Recognition

Integrating advanced interaction methods:

```csharp
// Advanced interaction system
using UnityEngine;
using UnityEngine.Windows.Speech;
using System.Collections.Generic;

public class AdvancedInteractionSystem : MonoBehaviour
{
    [Header("Voice Recognition")]
    public KeywordRecognizer keywordRecognizer;
    public Dictionary<string, System.Action> keywords = new Dictionary<string, System.Action>();

    [Header("Gesture Recognition")]
    public Camera gestureCamera;
    public float gestureDetectionRadius = 100f;

    [Header("Haptic Feedback")]
    public bool enableHapticFeedback = true;

    [Header("AR/VR Integration")]
    public bool isVRMode = false;
    public Camera vrCamera;

    void Start()
    {
        SetupVoiceRecognition();
        SetupGestureRecognition();
    }

    void SetupVoiceRecognition()
    {
        // Define voice commands
        keywords.Add("move forward", MoveForward);
        keywords.Add("move backward", MoveBackward);
        keywords.Add("turn left", TurnLeft);
        keywords.Add("turn right", TurnRight);
        keywords.Add("stop", StopRobot);
        keywords.Add("reset", ResetRobot);
        keywords.Add("follow me", FollowUser);

        keywordRecognizer = new KeywordRecognizer(keywords.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += OnPhraseRecognized;
        keywordRecognizer.Start();
    }

    void SetupGestureRecognition()
    {
        // Setup gesture recognition (requires additional plugins like Unity's AR Foundation)
        Debug.Log("Gesture recognition initialized");
    }

    void OnPhraseRecognized(PhraseRecognizedEventArgs args)
    {
        System.Action keywordAction;
        if (keywords.TryGetValue(args.text, out keywordAction))
        {
            keywordAction.Invoke();
            Debug.Log($"Voice command recognized: {args.text}");
        }
    }

    void MoveForward()
    {
        Debug.Log("Moving robot forward via voice command");
        // Send ROS command to move robot forward
    }

    void MoveBackward()
    {
        Debug.Log("Moving robot backward via voice command");
        // Send ROS command to move robot backward
    }

    void TurnLeft()
    {
        Debug.Log("Turning robot left via voice command");
        // Send ROS command to turn robot left
    }

    void TurnRight()
    {
        Debug.Log("Turning robot right via voice command");
        // Send ROS command to turn robot right
    }

    void StopRobot()
    {
        Debug.Log("Stopping robot via voice command");
        // Send ROS command to stop robot
    }

    void ResetRobot()
    {
        Debug.Log("Resetting robot via voice command");
        // Send ROS command to reset robot
    }

    void FollowUser()
    {
        Debug.Log("Robot will follow user via voice command");
        // Send ROS command to follow user
    }

    // Gesture recognition methods
    public void ProcessGesture(Vector2 gestureStart, Vector2 gestureEnd)
    {
        Vector2 gestureVector = gestureEnd - gestureStart;

        if (gestureVector.magnitude > gestureDetectionRadius)
        {
            if (Mathf.Abs(gestureVector.x) > Mathf.Abs(gestureVector.y))
            {
                // Horizontal gesture
                if (gestureVector.x > 0)
                    MoveRight();
                else
                    MoveLeft();
            }
            else
            {
                // Vertical gesture
                if (gestureVector.y > 0)
                    MoveUp();
                else
                    MoveDown();
            }
        }
    }

    void MoveLeft()
    {
        Debug.Log("Moving robot left via gesture");
    }

    void MoveRight()
    {
        Debug.Log("Moving robot right via gesture");
    }

    void MoveUp()
    {
        Debug.Log("Moving robot up via gesture");
    }

    void MoveDown()
    {
        Debug.Log("Moving robot down via gesture");
    }

    // Haptic feedback methods
    public void TriggerHapticFeedback(float intensity, float duration)
    {
        if (enableHapticFeedback)
        {
            // Trigger haptic feedback (implementation depends on device)
            Debug.Log($"Haptic feedback: intensity={intensity}, duration={duration}");
        }
    }

    void OnDestroy()
    {
        if (keywordRecognizer != null && keywordRecognizer.IsRunning)
        {
            keywordRecognizer.Stop();
        }
    }
}
```

## Unity-ROS Integration

### ROS TCP Connector

Establishing communication between Unity and ROS:

```csharp
// Unity ROS communication manager
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.RosBridgeClient.Protocols;

public class UnityRosManager : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeServerUrl = "ws://127.0.0.1:9090";
    public float connectionTimeout = 10.0f;

    [Header("Robot Topics")]
    public string robotStateTopic = "/joint_states";
    public string robotCommandTopic = "/cmd_vel";
    public string robotControlTopic = "/joint_group_position_controller/command";

    [Header("Sensors")]
    public string cameraTopic = "/camera/image_raw";
    public string laserTopic = "/scan";
    public string imuTopic = "/imu/data";

    private RosSocket rosSocket;
    private bool isConnected = false;

    void Start()
    {
        ConnectToRosBridge();
    }

    void ConnectToRosBridge()
    {
        IProtocol protocol = new WebSocketProtocol(new System.Uri(rosBridgeServerUrl));
        rosSocket = new RosSocket(protocol);

        // Subscribe to robot state
        rosSocket.Subscribe<JointStateMessage>(robotStateTopic, OnRobotStateReceived);

        // Subscribe to sensor data
        rosSocket.Subscribe<OdometryMessage>("/odom", OnOdometryReceived);
        rosSocket.Subscribe<ImuMessage>(imuTopic, OnImuReceived);

        // Connection status check
        InvokeRepeating("CheckConnectionStatus", 1.0f, 1.0f);

        Debug.Log($"Connecting to ROS Bridge at {rosBridgeServerUrl}");
    }

    void OnRobotStateReceived(JointStateMessage message)
    {
        // Update robot model in Unity based on joint states
        UpdateRobotModel(message);
    }

    void OnOdometryReceived(OdometryMessage message)
    {
        // Update robot position and orientation in Unity
        UpdateRobotPosition(message);
    }

    void OnImuReceived(ImuMessage message)
    {
        // Update robot orientation based on IMU data
        UpdateRobotOrientation(message);
    }

    void UpdateRobotModel(JointStateMessage jointState)
    {
        // Update Unity robot model based on joint positions
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];

            // Find corresponding joint in Unity model and update
            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint rotation based on position
                jointTransform.localRotation = Quaternion.Euler(0, jointPosition * Mathf.Rad2Deg, 0);
            }
        }
    }

    void UpdateRobotPosition(OdometryMessage odometry)
    {
        // Update robot position in Unity world
        Vector3 position = new Vector3(
            (float)odometry.pose.pose.position.x,
            (float)odometry.pose.pose.position.y,
            (float)odometry.pose.pose.position.z
        );

        transform.position = position;
    }

    void UpdateRobotOrientation(ImuMessage imu)
    {
        // Update robot orientation based on IMU data
        Quaternion orientation = new Quaternion(
            (float)imu.orientation.x,
            (float)imu.orientation.y,
            (float)imu.orientation.z,
            (float)imu.orientation.w
        );

        transform.rotation = orientation;
    }

    Transform FindJointByName(string jointName)
    {
        // Find joint by name in the robot hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == jointName)
                return child;
        }
        return null;
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (rosSocket != null && isConnected)
        {
            // Create and send velocity command
            var cmdVel = new TwistMessage();
            cmdVel.linear = new Vector3Message(linearX, 0, 0);
            cmdVel.angular = new Vector3Message(0, 0, angularZ);

            rosSocket.Publish(robotCommandTopic, cmdVel);
        }
    }

    public void SendJointPositionCommand(string[] jointNames, float[] positions)
    {
        if (rosSocket != null && isConnected)
        {
            // Create and send joint position command
            var jointCmd = new JointTrajectoryMessage();
            jointCmd.joint_names = jointNames;

            var point = new JointTrajectoryPointMessage();
            point.positions = new double[positions.Length];
            for (int i = 0; i < positions.Length; i++)
            {
                point.positions[i] = positions[i];
            }
            point.time_from_start = new DurationMessage(1, 0); // 1 second

            jointCmd.points = new JointTrajectoryPointMessage[] { point };

            rosSocket.Publish(robotControlTopic, jointCmd);
        }
    }

    void CheckConnectionStatus()
    {
        // Check if connection is still active
        // This is a simplified check - in practice, you'd implement proper connection monitoring
        isConnected = rosSocket != null;
    }

    void OnDestroy()
    {
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

### Unity Robotics Package Integration

Using the Unity Robotics Package for more advanced integration:

```csharp
// Enhanced ROS integration using Unity Robotics Package
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class EnhancedRobotInterface : MonoBehaviour
{
    [Header("ROS Topics")]
    public string jointStateTopic = "/joint_states";
    public string cmdVelTopic = "/cmd_vel";
    public string imageTopic = "/camera/image_raw";
    public string pointCloudTopic = "/point_cloud";

    [Header("Robot Configuration")]
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;

    private ROSConnection ros;
    private JointStateMsg lastJointState;

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(cmdVelTopic);

        // Subscribe to joint states
        ros.Subscribe<JointStateMsg>(jointStateTopic, OnJointStateReceived);

        // Subscribe to camera data
        ros.Subscribe<ImageMsg>(imageTopic, OnImageReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        lastJointState = jointState;
        UpdateRobotModel(jointState);
    }

    void OnImageReceived(ImageMsg image)
    {
        // Process camera image data
        ProcessCameraImage(image);
    }

    void UpdateRobotModel(JointStateMsg jointState)
    {
        // Update Unity robot model based on joint states
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];

            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint based on position
                UpdateJointTransform(jointTransform, jointPosition);
            }
        }
    }

    void UpdateJointTransform(Transform jointTransform, float position)
    {
        // Apply position to joint transform
        // This could involve setting rotation, position, or scale depending on joint type
        jointTransform.localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        // Clamp values to maximum velocities
        linearX = Mathf.Clamp(linearX, -maxLinearVelocity, maxLinearVelocity);
        angularZ = Mathf.Clamp(angularZ, -maxAngularVelocity, maxAngularVelocity);

        // Create and publish Twist message
        var twist = new TwistMsg();
        twist.linear = new Vector3Msg(linearX, 0, 0);
        twist.angular = new Vector3Msg(0, 0, angularZ);

        ros.Publish(cmdVelTopic, twist);
    }

    public void SendJointPositions(string[] jointNames, double[] positions)
    {
        // Create and publish joint trajectory command
        var trajectory = new Unity.Robotics.ROSTCPConnector.MessageTypes.Control.JointTrajectoryMsg();
        trajectory.joint_names = jointNames;

        var point = new Unity.Robotics.ROSTCPConnector.MessageTypes.Control.JointTrajectoryPointMsg();
        point.positions = positions;
        point.time_from_start = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std.DurationMsg(1, 0);

        trajectory.points = new Unity.Robotics.ROSTCPConnector.MessageTypes.Control.JointTrajectoryPointMsg[] { point };

        string controllerTopic = "/joint_group_position_controller/command";
        ros.Publish(controllerTopic, trajectory);
    }

    void ProcessCameraImage(ImageMsg image)
    {
        // Convert ROS image to Unity texture
        // This would involve converting the image data format
        Debug.Log($"Received camera image: {image.width}x{image.height}");
    }

    // Example of sending sensor data from Unity to ROS
    public void PublishSensorData()
    {
        // Create sensor message
        var sensorMsg = new LaserScanMsg();
        sensorMsg.header = new HeaderMsg();
        sensorMsg.header.stamp = new TimeMsg(System.DateTime.UtcNow.Second, System.DateTime.UtcNow.Millisecond * 1000000);
        sensorMsg.header.frame_id = "laser_frame";

        // Fill in sensor data
        sensorMsg.angle_min = -Mathf.PI / 2;
        sensorMsg.angle_max = Mathf.PI / 2;
        sensorMsg.angle_increment = Mathf.PI / 180; // 1 degree
        sensorMsg.range_min = 0.1f;
        sensorMsg.range_max = 10.0f;

        // Example ranges (in practice, this would come from Unity's raycasting)
        int numRanges = 181; // 181 points for 180 degrees at 1 degree increments
        sensorMsg.ranges = new float[numRanges];
        for (int i = 0; i < numRanges; i++)
        {
            sensorMsg.ranges[i] = Random.Range(0.5f, 5.0f); // Random example values
        }

        ros.Publish("/scan", sensorMsg);
    }

    Transform FindJointByName(string jointName)
    {
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == jointName)
                return child;
        }
        return null;
    }
}
```

## VR/AR Integration for Humanoid Robotics

### VR Teleoperation Interface

Creating immersive VR interfaces for robot control:

```csharp
// VR teleoperation interface
using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

public class VRTeleoperationInterface : MonoBehaviour
{
    [Header("VR Controllers")]
    public Transform leftController;
    public Transform rightController;
    public GameObject leftControllerModel;
    public GameObject rightControllerModel;

    [Header("Robot Control Mapping")]
    public float controllerSensitivity = 1.0f;
    public float maxTeleopVelocity = 0.5f;

    [Header("Haptic Feedback")]
    public bool enableHapticFeedback = true;

    [Header("Safety Boundaries")]
    public float operationRadius = 5.0f;
    public LayerMask environmentLayer;

    [Header("VR Camera")]
    public Camera vrCamera;

    private EnhancedRobotInterface robotInterface;
    private Vector2 lastLeftStick;
    private Vector2 lastRightStick;

    void Start()
    {
        robotInterface = FindObjectOfType<EnhancedRobotInterface>();
        SetupVRControllers();
    }

    void SetupVRControllers()
    {
        // Initialize VR controllers
        if (leftControllerModel != null)
            leftControllerModel.SetActive(true);

        if (rightControllerModel != null)
            rightControllerModel.SetActive(true);
    }

    void Update()
    {
        HandleVRInput();
        UpdateVRVisualization();
    }

    void HandleVRInput()
    {
        // Get controller inputs
        Vector2 leftStick = GetControllerAxis("Left", "Axis 1", "Axis 2");
        Vector2 rightStick = GetControllerAxis("Right", "Axis 1", "Axis 2");

        // Calculate robot movement based on controller input
        float linearVel = 0;
        float angularVel = 0;

        // Left stick for movement
        if (leftStick.magnitude > 0.1f)
        {
            linearVel = leftStick.y * maxTeleopVelocity * controllerSensitivity;
            angularVel = -leftStick.x * maxTeleopVelocity * controllerSensitivity;
        }

        // Right stick for fine control
        if (rightStick.magnitude > 0.1f)
        {
            // Add fine control adjustments
            linearVel += rightStick.y * maxTeleopVelocity * controllerSensitivity * 0.3f;
            angularVel -= rightStick.x * maxTeleopVelocity * controllerSensitivity * 0.3f;
        }

        // Send commands to robot
        if (robotInterface != null)
        {
            robotInterface.SendVelocityCommand(linearVel, angularVel);
        }

        // Store for haptic feedback
        lastLeftStick = leftStick;
        lastRightStick = rightStick;
    }

    Vector2 GetControllerAxis(string controller, string xAxis, string yAxis)
    {
        // Get controller axis input
        float x = Input.GetAxis($"{controller} {xAxis}");
        float y = Input.GetAxis($"{controller} {yAxis}");
        return new Vector2(x, y);
    }

    void UpdateVRVisualization()
    {
        // Update controller models to match actual controller positions
        if (leftController != null && leftControllerModel != null)
        {
            leftControllerModel.transform.position = leftController.position;
            leftControllerModel.transform.rotation = leftController.rotation;
        }

        if (rightController != null && rightControllerModel != null)
        {
            rightControllerModel.transform.position = rightController.position;
            rightControllerModel.transform.rotation = rightController.rotation;
        }

        // Provide haptic feedback based on robot state
        if (enableHapticFeedback)
        {
            ProvideHapticFeedback();
        }
    }

    void ProvideHapticFeedback()
    {
        // Example: Vibrate controller based on robot velocity
        float vibrationIntensity = 0;

        if (lastLeftStick.magnitude > 0.5f || lastRightStick.magnitude > 0.5f)
        {
            vibrationIntensity = Mathf.Min(lastLeftStick.magnitude, lastRightStick.magnitude);
            ApplyControllerHaptics(vibrationIntensity);
        }
    }

    void ApplyControllerHaptics(float intensity)
    {
        // Apply haptic feedback to controllers
        // This would use actual VR SDK methods (Oculus, SteamVR, etc.)
        Debug.Log($"Applying haptic feedback: {intensity}");
    }

    // Safety check methods
    public bool IsWithinOperationalBounds(Vector3 position)
    {
        return Vector3.Distance(vrCamera.transform.position, position) <= operationRadius;
    }

    public void CheckEnvironmentCollision()
    {
        // Check for collisions between VR space and robot environment
        RaycastHit hit;
        if (Physics.Raycast(vrCamera.transform.position, vrCamera.transform.forward, out hit, 10f, environmentLayer))
        {
            Debug.Log($"VR-robot environment collision detected: {hit.collider.name}");
            // Handle collision appropriately
        }
    }

    // VR-specific robot control methods
    public void PointAndMove(Vector3 targetPoint)
    {
        // Calculate direction to target point
        Vector3 direction = (targetPoint - vrCamera.transform.position).normalized;

        // Send movement command to robot
        if (robotInterface != null)
        {
            robotInterface.SendVelocityCommand(direction.z * maxTeleopVelocity, -direction.x * maxTeleopVelocity);
        }
    }

    public void GraspObject(Vector3 objectPosition)
    {
        // Send grasp command to robot
        Debug.Log($"Attempting to grasp object at: {objectPosition}");
        // Implementation would send appropriate ROS messages for manipulation
    }
}
```

### AR Overlay Interface

Creating AR interfaces for robot monitoring:

```csharp
// AR interface for robot monitoring
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ARRobotInterface : MonoBehaviour
{
    [Header("AR Camera")]
    public Camera arCamera;

    [Header("Robot Status Overlays")]
    public GameObject statusPanel;
    public TextMeshProUGUI batteryText;
    public TextMeshProUGUI positionText;
    public TextMeshProUGUI statusText;
    public Slider batterySlider;

    [Header("Navigation Elements")]
    public GameObject waypointIndicator;
    public GameObject pathVisualization;
    public GameObject destinationMarker;

    [Header("Safety Indicators")]
    public GameObject safetyZoneIndicator;
    public GameObject obstacleWarning;
    public Image safetyZoneRing;

    [Header("Interaction Elements")]
    public GameObject commandPanel;
    public Button stopButton;
    public Button pauseButton;
    public Button resumeButton;

    private EnhancedRobotInterface robotInterface;
    private bool isTrackingRobot = false;

    void Start()
    {
        robotInterface = FindObjectOfType<EnhancedRobotInterface>();
        SetupARInterface();
    }

    void SetupARInterface()
    {
        // Initialize AR interface elements
        statusPanel.SetActive(true);
        waypointIndicator.SetActive(false);
        pathVisualization.SetActive(false);
        destinationMarker.SetActive(false);
        safetyZoneIndicator.SetActive(false);
        obstacleWarning.SetActive(false);

        // Setup button events
        if (stopButton != null)
            stopButton.onClick.AddListener(OnStopClicked);
        if (pauseButton != null)
            pauseButton.onClick.AddListener(OnPauseClicked);
        if (resumeButton != null)
            resumeButton.onClick.AddListener(OnResumeClicked);
    }

    void Update()
    {
        UpdateARTracking();
        UpdateRobotStatus();
    }

    void UpdateARTracking()
    {
        // Track robot in AR space
        if (isTrackingRobot)
        {
            // Update AR elements to follow robot position
            UpdateStatusPanelPosition();
            UpdateWaypointIndicators();
            UpdateSafetyIndicators();
        }
    }

    void UpdateStatusPanelPosition()
    {
        // Position status panel relative to robot in AR space
        if (robotInterface != null)
        {
            // This would involve getting robot position from ROS and converting to AR space
            // For now, we'll simulate with a fixed offset
            Vector3 robotPosition = arCamera.transform.position + arCamera.transform.forward * 2.0f;
            statusPanel.transform.position = robotPosition;
        }
    }

    void UpdateRobotStatus()
    {
        // Update status information
        if (batteryText != null)
            batteryText.text = "Battery: 85%";

        if (positionText != null)
            positionText.text = "Position: X:1.2 Y:0.5 Z:0.0";

        if (statusText != null)
            statusText.text = "Status: Moving";

        if (batterySlider != null)
            batterySlider.value = 0.85f;
    }

    void UpdateWaypointIndicators()
    {
        // Update navigation elements
        if (waypointIndicator != null)
        {
            // Show/hide based on robot state
            waypointIndicator.SetActive(true);
        }

        if (pathVisualization != null)
        {
            // Update path visualization
            pathVisualization.SetActive(true);
        }

        if (destinationMarker != null)
        {
            // Update destination marker position
            destinationMarker.SetActive(true);
        }
    }

    void UpdateSafetyIndicators()
    {
        // Update safety elements based on environment
        if (safetyZoneIndicator != null)
        {
            safetyZoneIndicator.SetActive(true);
        }

        if (obstacleWarning != null)
        {
            // Check for obstacles and show warning if needed
            bool hasObstacles = CheckForObstacles();
            obstacleWarning.SetActive(hasObstacles);
        }
    }

    bool CheckForObstacles()
    {
        // Check for obstacles in robot path
        // This would involve processing sensor data from ROS
        return false; // Placeholder
    }

    public void StartTrackingRobot()
    {
        isTrackingRobot = true;
        statusPanel.SetActive(true);
    }

    public void StopTrackingRobot()
    {
        isTrackingRobot = false;
        statusPanel.SetActive(false);
    }

    void OnStopClicked()
    {
        Debug.Log("Stop button clicked");
        if (robotInterface != null)
        {
            robotInterface.SendVelocityCommand(0, 0);
        }
    }

    void OnPauseClicked()
    {
        Debug.Log("Pause button clicked");
        // Send pause command to robot
    }

    void OnResumeClicked()
    {
        Debug.Log("Resume button clicked");
        // Send resume command to robot
    }

    // AR-specific interaction methods
    public void SetDestination(Vector3 destination)
    {
        // Send navigation goal to robot
        Debug.Log($"Setting destination: {destination}");
        // This would send a ROS navigation goal
    }

    public void ShowPathVisualization(bool show)
    {
        if (pathVisualization != null)
        {
            pathVisualization.SetActive(show);
        }
    }

    public void HighlightRobot()
    {
        // Highlight robot in AR view
        Debug.Log("Robot highlighted in AR view");
    }

    public void ToggleSafetyZone(bool visible)
    {
        if (safetyZoneIndicator != null)
        {
            safetyZoneIndicator.SetActive(visible);
        }
    }
}
```

## Performance Optimization for Complex Humanoid Models

### Level of Detail (LOD) System

Optimizing rendering performance for complex humanoid models:

```csharp
// LOD system for humanoid robots
using UnityEngine;

[System.Serializable]
public class RobotLODLevel
{
    public float distance;
    public GameObject[] renderers;
    public int qualityLevel;
    public float physicsComplexity;
}

public class RobotLODManager : MonoBehaviour
{
    [Header("LOD Configuration")]
    public RobotLODLevel[] lodLevels;
    public Transform viewer;
    public float updateInterval = 0.5f;

    [Header("Performance Settings")]
    public bool enableLOD = true;
    public bool enablePhysicsLOD = true;
    public bool enableTextureLOD = true;

    private float lastUpdate;
    private int currentLODLevel = 0;

    void Start()
    {
        if (viewer == null)
            viewer = Camera.main.transform;

        SetupLODLevels();
        lastUpdate = Time.time;
    }

    void Update()
    {
        if (enableLOD && Time.time - lastUpdate > updateInterval)
        {
            UpdateLOD();
            lastUpdate = Time.time;
        }
    }

    void SetupLODLevels()
    {
        // Ensure LOD levels are sorted by distance
        System.Array.Sort(lodLevels, (a, b) => a.distance.CompareTo(b.distance));
    }

    void UpdateLOD()
    {
        if (viewer == null) return;

        float distance = Vector3.Distance(transform.position, viewer.position);
        int newLODLevel = FindLODLevel(distance);

        if (newLODLevel != currentLODLevel)
        {
            SetLODLevel(newLODLevel);
            currentLODLevel = newLODLevel;
        }
    }

    int FindLODLevel(float distance)
    {
        for (int i = lodLevels.Length - 1; i >= 0; i--)
        {
            if (distance >= lodLevels[i].distance)
                return i;
        }
        return 0; // Use highest detail level if within closest distance
    }

    void SetLODLevel(int lodIndex)
    {
        for (int i = 0; i < lodLevels.Length; i++)
        {
            bool isVisible = i == lodIndex;

            if (lodLevels[i].renderers != null)
            {
                foreach (GameObject renderer in lodLevels[i].renderers)
                {
                    if (renderer != null)
                        renderer.SetActive(isVisible);
                }
            }

            // Adjust physics complexity
            if (enablePhysicsLOD)
            {
                AdjustPhysicsComplexity(i, isVisible, lodLevels[i].physicsComplexity);
            }

            // Adjust texture quality
            if (enableTextureLOD && isVisible)
            {
                AdjustTextureQuality(lodLevels[i].qualityLevel);
            }
        }
    }

    void AdjustPhysicsComplexity(int lodIndex, bool isActive, float complexity)
    {
        // Adjust physics properties based on LOD level
        if (!isActive) return;

        // Example: Reduce physics simulation quality for distant models
        if (complexity < 0.5f)
        {
            // Use simplified collision shapes
            // Reduce physics update rate
        }
        else if (complexity > 0.8f)
        {
            // Use detailed collision shapes
            // Increase physics accuracy
        }
    }

    void AdjustTextureQuality(int qualityLevel)
    {
        // Adjust texture quality based on LOD
        // This would involve switching texture atlases or resolution
        QualitySettings.SetQualityLevel(qualityLevel, true);
    }
}
```

### Occlusion Culling and Frustum Culling

Optimizing rendering by only drawing visible objects:

```csharp
// Advanced culling system for robotics environments
using UnityEngine;

public class RoboticsCullingSystem : MonoBehaviour
{
    [Header("Culling Configuration")]
    public Camera mainCamera;
    public float cullingUpdateInterval = 0.1f;
    public float maxCullingDistance = 50.0f;

    [Header("Robot Culling")]
    public Transform[] robotTransforms;
    public Renderer[] robotRenderers;

    [Header("Environment Culling")]
    public GameObject[] environmentObjects;
    public Renderer[] environmentRenderers;

    [Header("Performance Monitoring")]
    public bool enablePerformanceLogging = false;
    public TextMesh performanceText;

    private float lastCullingUpdate;
    private int visibleRobots = 0;
    private int visibleEnvironment = 0;

    void Start()
    {
        if (mainCamera == null)
            mainCamera = Camera.main;

        lastCullingUpdate = Time.time;
    }

    void Update()
    {
        if (Time.time - lastCullingUpdate > cullingUpdateInterval)
        {
            PerformCulling();
            lastCullingUpdate = Time.time;

            if (enablePerformanceLogging)
            {
                LogPerformance();
            }
        }
    }

    void PerformCulling()
    {
        visibleRobots = 0;
        visibleEnvironment = 0;

        // Cull robots
        if (robotRenderers != null)
        {
            foreach (Renderer renderer in robotRenderers)
            {
                if (renderer != null)
                {
                    bool isVisible = IsVisible(renderer.bounds);
                    renderer.enabled = isVisible;
                    if (isVisible) visibleRobots++;
                }
            }
        }

        // Cull environment objects
        if (environmentRenderers != null)
        {
            foreach (Renderer renderer in environmentRenderers)
            {
                if (renderer != null)
                {
                    bool isVisible = IsVisible(renderer.bounds) &&
                                   Vector3.Distance(renderer.transform.position, mainCamera.transform.position) < maxCullingDistance;
                    renderer.enabled = isVisible;
                    if (isVisible) visibleEnvironment++;
                }
            }
        }
    }

    bool IsVisible(Bounds bounds)
    {
        if (mainCamera == null) return true;

        // Check if bounds are within camera frustum
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(mainCamera);
        return GeometryUtility.TestPlanesAABB(planes, bounds);
    }

    void LogPerformance()
    {
        if (performanceText != null)
        {
            performanceText.text = $"Visible: {visibleRobots} robots, {visibleEnvironment} env objects";
        }

        if (enablePerformanceLogging)
        {
            Debug.Log($"Culling: {visibleRobots} robots, {visibleEnvironment} environment objects visible");
        }
    }

    // Dynamic batching optimization
    public void OptimizeForBatching()
    {
        // Combine static environment objects for better batching
        // This would involve mesh combining for static objects
        Debug.Log("Optimizing environment for dynamic batching");
    }

    // Texture atlasing for robot components
    public void CreateTextureAtlas()
    {
        // Create texture atlas for robot components to reduce draw calls
        Debug.Log("Creating texture atlas for robot components");
    }
}
```

## Best Practices and Design Patterns

### Modular Architecture

Creating a modular system for humanoid robot simulation:

```csharp
// Modular robot system architecture
using UnityEngine;
using System.Collections.Generic;

// Base interface for all robot modules
public interface IRobotModule
{
    string ModuleName { get; }
    bool IsOperational { get; }
    void Initialize();
    void UpdateModule();
    void Shutdown();
}

// Robot controller that manages all modules
public class ModularRobotController : MonoBehaviour, IRobotModule
{
    [Header("Robot Modules")]
    public List<IRobotModule> modules = new List<IRobotModule>();

    [Header("Module Configuration")]
    public bool autoInitialize = true;
    public float moduleUpdateInterval = 0.1f;

    private float lastModuleUpdate;
    private bool isInitialized = false;

    public string ModuleName => "Robot Controller";
    public bool IsOperational => isInitialized && modules.TrueForAll(m => m.IsOperational);

    void Start()
    {
        if (autoInitialize)
        {
            Initialize();
        }
    }

    void Update()
    {
        if (Time.time - lastModuleUpdate > moduleUpdateInterval)
        {
            UpdateModule();
            lastModuleUpdate = Time.time;
        }
    }

    public void Initialize()
    {
        foreach (IRobotModule module in modules)
        {
            module.Initialize();
        }
        isInitialized = true;
        Debug.Log("Robot controller initialized with all modules");
    }

    public void UpdateModule()
    {
        foreach (IRobotModule module in modules)
        {
            if (module.IsOperational)
            {
                module.UpdateModule();
            }
        }
    }

    public void Shutdown()
    {
        foreach (IRobotModule module in modules)
        {
            module.Shutdown();
        }
        isInitialized = false;
        Debug.Log("Robot controller shut down");
    }

    // Add module at runtime
    public void AddModule(IRobotModule module)
    {
        modules.Add(module);
        if (isInitialized)
        {
            module.Initialize();
        }
    }

    // Remove module at runtime
    public bool RemoveModule(IRobotModule module)
    {
        if (modules.Remove(module))
        {
            module.Shutdown();
            return true;
        }
        return false;
    }

    // Get module by type
    public T GetModule<T>() where T : class, IRobotModule
    {
        foreach (IRobotModule module in modules)
        {
            if (module is T tModule)
            {
                return tModule;
            }
        }
        return null;
    }
}

// Example: Rendering module
public class RobotRenderingModule : MonoBehaviour, IRobotModule
{
    public string ModuleName => "Rendering Module";
    public bool IsOperational { get; private set; }

    [Header("Rendering Configuration")]
    public Material[] robotMaterials;
    public Light[] robotLights;
    public ParticleSystem[] robotEffects;

    public void Initialize()
    {
        // Setup rendering components
        SetupRendering();
        IsOperational = true;
        Debug.Log($"{ModuleName} initialized");
    }

    public void UpdateModule()
    {
        // Update rendering-specific logic
        UpdateVisualEffects();
    }

    public void Shutdown()
    {
        // Cleanup rendering resources
        CleanupRendering();
        IsOperational = false;
        Debug.Log($"{ModuleName} shut down");
    }

    void SetupRendering()
    {
        // Initialize rendering components
        if (robotMaterials != null)
        {
            foreach (Material mat in robotMaterials)
            {
                if (mat != null)
                {
                    // Configure material properties
                }
            }
        }
    }

    void UpdateVisualEffects()
    {
        // Update particle effects, lighting, etc.
        if (robotEffects != null)
        {
            foreach (ParticleSystem effect in robotEffects)
            {
                if (effect != null && effect.isPlaying)
                {
                    // Update effect parameters
                }
            }
        }
    }

    void CleanupRendering()
    {
        // Cleanup rendering resources
    }
}

// Example: Interaction module
public class RobotInteractionModule : MonoBehaviour, IRobotModule
{
    public string ModuleName => "Interaction Module";
    public bool IsOperational { get; private set; }

    [Header("Interaction Configuration")]
    public LayerMask interactionLayer;
    public float interactionDistance = 3.0f;

    public void Initialize()
    {
        // Setup interaction components
        SetupInteraction();
        IsOperational = true;
        Debug.Log($"{ModuleName} initialized");
    }

    public void UpdateModule()
    {
        // Update interaction-specific logic
        CheckForInteractions();
    }

    public void Shutdown()
    {
        // Cleanup interaction components
        IsOperational = false;
        Debug.Log($"{ModuleName} shut down");
    }

    void SetupInteraction()
    {
        // Initialize interaction components
    }

    void CheckForInteractions()
    {
        // Check for user interactions
        Ray ray = Camera.main.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, interactionDistance, interactionLayer))
        {
            // Handle interaction
            HandleInteraction(hit);
        }
    }

    void HandleInteraction(RaycastHit hit)
    {
        // Process interaction with hit object
        Debug.Log($"Interaction with: {hit.collider.name}");
    }
}
```

## Practical Example: Complete Humanoid Robot Simulation

Here's a complete example of a humanoid robot simulation setup in Unity:

```csharp
// Complete humanoid robot simulation setup
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class CompleteHumanoidSimulation : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotModel;
    public string robotName = "HumanoidRobot";
    public float robotHeight = 1.7f;

    [Header("Simulation Components")]
    public UnityRosManager rosManager;
    public RobotControlInterface controlInterface;
    public AdvancedInteractionSystem interactionSystem;
    public VRTeleoperationInterface vrInterface;
    public ARRobotInterface arInterface;
    public RobotLODManager lodManager;

    [Header("Visual Components")]
    public RoboticsLightingSetup lightingSetup;
    public RobotEffects robotEffects;
    public RoboticsCullingSystem cullingSystem;

    [Header("Simulation Settings")]
    public bool enablePhysics = true;
    public bool enableRealtimeRendering = true;
    public bool enableVRMode = false;
    public bool enableARMode = false;

    [Header("Performance Metrics")]
    public TextMeshProUGUI performanceText;
    public Slider simulationSpeedSlider;

    void Start()
    {
        InitializeSimulation();
    }

    void InitializeSimulation()
    {
        // Initialize all components
        InitializeRobotModel();
        InitializeCommunication();
        InitializeInterfaces();
        InitializeVisualComponents();
        InitializeSimulationSettings();

        Debug.Log($"Humanoid robot simulation initialized: {robotName}");
    }

    void InitializeRobotModel()
    {
        if (robotModel != null)
        {
            robotModel.transform.localScale = Vector3.one * (robotHeight / 1.7f); // Scale to desired height
        }
    }

    void InitializeCommunication()
    {
        if (rosManager != null)
        {
            rosManager.rosBridgeServerUrl = "ws://127.0.0.1:9090";
            rosManager.robotStateTopic = $"/{robotName}/joint_states";
            rosManager.robotCommandTopic = $"/{robotName}/cmd_vel";
        }
    }

    void InitializeInterfaces()
    {
        // Setup control interface
        if (controlInterface != null)
        {
            controlInterface.UpdateStatus("Initializing");
        }

        // Setup interaction system
        if (interactionSystem != null)
        {
            interactionSystem.isVRMode = enableVRMode;
        }

        // Setup VR interface
        if (vrInterface != null)
        {
            vrInterface.enabled = enableVRMode;
        }

        // Setup AR interface
        if (arInterface != null)
        {
            arInterface.enabled = enableARMode;
        }

        // Setup LOD system
        if (lodManager != null)
        {
            lodManager.enableLOD = true;
        }
    }

    void InitializeVisualComponents()
    {
        // Setup lighting
        if (lightingSetup != null)
        {
            lightingSetup.environmentIntensity = 1.0f;
        }

        // Setup effects
        if (robotEffects != null)
        {
            robotEffects.dustIntensity = 1.0f;
        }

        // Setup culling
        if (cullingSystem != null)
        {
            cullingSystem.enablePerformanceLogging = true;
        }
    }

    void InitializeSimulationSettings()
    {
        // Configure physics
        if (enablePhysics)
        {
            Physics.defaultSolverIterations = 10;
            Physics.defaultSolverVelocityIterations = 8;
        }

        // Setup performance monitoring
        if (performanceText != null)
        {
            performanceText.text = "Simulation Running";
        }

        // Setup simulation speed control
        if (simulationSpeedSlider != null)
        {
            simulationSpeedSlider.minValue = 0.1f;
            simulationSpeedSlider.maxValue = 5.0f;
            simulationSpeedSlider.value = 1.0f;
            simulationSpeedSlider.onValueChanged.AddListener(OnSimulationSpeedChanged);
        }
    }

    void OnSimulationSpeedChanged(float speed)
    {
        Time.timeScale = speed;
        Debug.Log($"Simulation speed changed to: {speed}x");
    }

    void Update()
    {
        UpdateSimulationMetrics();
        HandleSimulationInputs();
    }

    void UpdateSimulationMetrics()
    {
        if (performanceText != null)
        {
            float fps = 1.0f / Time.unscaledDeltaTime;
            performanceText.text = $"FPS: {fps:F1} | Robots: 1 | Objects: {GetActiveObjectCount()}";
        }
    }

    int GetActiveObjectCount()
    {
        // Count active objects in the scene
        return GameObject.FindObjectsOfType<Renderer>().Length;
    }

    void HandleSimulationInputs()
    {
        // Handle global simulation inputs
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ToggleSimulationPause();
        }

        if (Input.GetKeyDown(KeyCode.F1))
        {
            ToggleVRMode();
        }

        if (Input.GetKeyDown(KeyCode.F2))
        {
            ToggleARMode();
        }
    }

    void ToggleSimulationPause()
    {
        Time.timeScale = Time.timeScale == 0 ? 1 : 0;
        Debug.Log($"Simulation {(Time.timeScale == 0 ? "paused" : "resumed")}");
    }

    void ToggleVRMode()
    {
        enableVRMode = !enableVRMode;
        if (vrInterface != null)
        {
            vrInterface.enabled = enableVRMode;
        }
        Debug.Log($"VR mode {(enableVRMode ? "enabled" : "disabled")}");
    }

    void ToggleARMode()
    {
        enableARMode = !enableARMode;
        if (arInterface != null)
        {
            arInterface.enabled = enableARMode;
        }
        Debug.Log($"AR mode {(enableARMode ? "enabled" : "disabled")}");
    }

    // Public methods for external control
    public void SetRobotPosition(Vector3 position)
    {
        if (robotModel != null)
        {
            robotModel.transform.position = position;
        }
    }

    public void SetRobotRotation(Quaternion rotation)
    {
        if (robotModel != null)
        {
            robotModel.transform.rotation = rotation;
        }
    }

    public void TriggerRobotAction(string action)
    {
        switch (action)
        {
            case "wave":
                TriggerWaveAnimation();
                break;
            case "point":
                TriggerPointAnimation();
                break;
            case "greet":
                TriggerGreetingSequence();
                break;
            default:
                Debug.Log($"Unknown robot action: {action}");
                break;
        }
    }

    void TriggerWaveAnimation()
    {
        // Trigger wave animation
        Debug.Log("Robot waving");
        if (robotEffects != null)
        {
            robotEffects.TriggerInteractionEffect(robotModel.transform.position + Vector3.up, Color.blue);
        }
    }

    void TriggerPointAnimation()
    {
        // Trigger pointing animation
        Debug.Log("Robot pointing");
    }

    void TriggerGreetingSequence()
    {
        // Trigger greeting sequence
        Debug.Log("Robot greeting");
        if (controlInterface != null)
        {
            controlInterface.UpdateStatus("Greeting");
        }
    }

    void OnDestroy()
    {
        // Cleanup resources
        if (rosManager != null)
        {
            rosManager.OnDestroy();
        }
    }
}
```

## Exercises

1. **Rendering Optimization Challenge**: Create a humanoid robot model with multiple LOD levels and implement a system that dynamically switches between them based on camera distance. Measure and compare performance at different settings.

2. **VR Interaction Design**: Design and implement a VR interface for teleoperating a humanoid robot that includes gesture recognition, voice commands, and haptic feedback. Test the interface with different types of robot movements.

3. **ROS Integration Project**: Create a Unity scene with a humanoid robot that receives joint state data from ROS and updates its visual model in real-time, while simultaneously sending velocity commands back to the ROS system.

## Summary

This chapter covered advanced Unity techniques for humanoid robotics:

- High-fidelity rendering using PBR materials, advanced lighting, and particle effects
- Human-robot interaction design with intuitive interfaces and advanced input methods
- Unity-ROS integration for bidirectional communication
- VR/AR implementation for immersive robot teleoperation and monitoring
- Performance optimization techniques including LOD systems and culling
- Modular architecture patterns for scalable robot systems

Unity provides powerful capabilities for creating visually impressive and interactive robot simulation environments. The combination of advanced rendering, intuitive interfaces, and VR/AR support makes it ideal for human-robot interaction research and development.

## Next Steps

In the next chapter, we'll explore sensor simulation in both Gazebo and Unity environments, covering how to model realistic sensors like LiDAR, cameras, and IMUs for humanoid robot applications. This will complete our coverage of digital twin technologies for robotics simulation.