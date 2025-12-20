---
id: chapter-14-voice-to-action
title: "Chapter 14: Voice-to-Action using OpenAI Whisper"
sidebar_label: "Chapter 14: Voice-to-Action using OpenAI Whisper"
description: "Implementing voice-to-action systems using OpenAI Whisper for humanoid robot interaction"
keywords: [whisper, speech recognition, voice-to-action, nlp, humanoid, ai]
tags: [voice-recognition, ai-integration, robotics]
authors: [book-authors]
difficulty: advanced
estimated_time: "105 minutes"
module: 4
chapter: 14
prerequisites: [python-ai-basics, ros2-foundations, llm-integration, audio-processing]
learning_objectives:
  - Implement speech recognition using OpenAI Whisper
  - Design voice-to-action pipelines for humanoid robots
  - Integrate speech recognition with ROS 2 systems
  - Handle real-time audio processing for robot interaction
  - Optimize voice-to-action systems for humanoid robotics
related:
  - next: chapter-15-cognitive-planning
  - previous: chapter-13-llm-robotics
  - see_also: [chapter-13-llm-robotics, chapter-15-cognitive-planning, chapter-16-autonomous-humanoid]
---

# Chapter 14: Voice-to-Action using OpenAI Whisper

## Learning Objectives

After completing this chapter, you will be able to:
- Implement speech recognition systems using OpenAI Whisper
- Design voice-to-action pipelines for humanoid robot interaction
- Integrate real-time audio processing with ROS 2 systems
- Handle noise filtering and audio quality issues in robotics environments
- Optimize voice-to-action systems for real-time humanoid robot responses

## Introduction

Voice-to-action systems are crucial for natural human-robot interaction, particularly for humanoid robots that are designed to work alongside humans. OpenAI Whisper has revolutionized speech recognition by providing highly accurate, multilingual speech-to-text capabilities that can be leveraged for robotics applications.

This chapter explores how to implement Whisper-based voice-to-action systems for humanoid robots, enabling them to understand spoken commands and convert them into executable robot actions. We'll cover both the technical implementation and the integration challenges specific to robotics environments.

## Understanding OpenAI Whisper

### Whisper Capabilities

OpenAI Whisper is a robust automatic speech recognition (ASR) system with several key features:

- **Multilingual Support**: Supports multiple languages out of the box
- **High Accuracy**: Performs well even in noisy environments
- **Robustness**: Handles various accents, background noise, and audio quality
- **Real-time Capabilities**: Can process audio streams with low latency
- **Open Source**: Available for both cloud and local deployment

### Whisper in Robotics Context

For humanoid robots, Whisper provides:
- Natural language command understanding
- Multilingual interaction capabilities
- Robust performance in varied acoustic environments
- Integration with existing AI systems (like the LLMs from Chapter 13)

## Voice-to-Action Architecture

### Basic Architecture

The voice-to-action system typically follows this flow:

```
Microphone Input → Audio Preprocessing → Speech Recognition → NLP Processing → Robot Action
```

### Advanced Architecture with Context

A more sophisticated system incorporates environmental context:

```
Microphone Input → Audio Preprocessing → Speech Recognition → Intent Recognition → Context Integration → Action Planning → Robot Execution
```

## Implementing Whisper with ROS 2

### Installing Whisper

First, install the required dependencies:

```bash
pip install openai-whisper
# Or for local processing without OpenAI API
pip install git+https://github.com/openai/whisper.git
```

### Basic Whisper Node

Here's a basic implementation of a Whisper-based speech recognition node:

```python
import rclpy
from rclpy.node import Node
import whisper
import numpy as np
import pyaudio
import wave
import threading
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import tempfile
import os

class WhisperSpeechRecognizer(Node):
    def __init__(self):
        super().__init__('whisper_speech_recognizer')
        
        # Load Whisper model (use smaller models for real-time performance)
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")  # Use "tiny" for faster processing
        
        # Publishers and subscribers
        self.command_pub = self.create_publisher(String, '/voice_command', 10)
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )
        
        # Audio processing parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_duration = 5.0  # Process 5-second chunks
        
        # Buffer for audio data
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        self.get_logger().info('Whisper Speech Recognizer initialized')

    def audio_callback(self, msg):
        """Process incoming audio data."""
        with self.buffer_lock:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize
            
            # Add to buffer
            self.audio_buffer.extend(audio_array)
            
            # If buffer has enough data, process it
            if len(self.audio_buffer) >= self.sample_rate * self.chunk_duration:
                # Process in a separate thread to avoid blocking
                buffer_copy = self.audio_buffer.copy()
                self.audio_buffer = []  # Clear buffer
                threading.Thread(target=self.process_audio, args=(buffer_copy,)).start()

    def process_audio(self, audio_data):
        """Process audio chunk with Whisper."""
        try:
            # Convert to the format Whisper expects
            audio_tensor = np.array(audio_data)
            
            # Transcribe with Whisper
            result = self.model.transcribe(audio_tensor, fp16=False)
            text = result['text'].strip()
            
            if text:  # Only publish if we got text
                self.get_logger().info(f'Recognized: {text}')
                
                # Publish recognized text as a command
                cmd_msg = String()
                cmd_msg.data = text
                self.command_pub.publish(cmd_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error in audio processing: {e}')
```

### Real-time Audio Capture Node

For direct microphone access, here's a node that captures audio in real-time:

```python
import rclpy
from rclpy.node import Node
import whisper
import pyaudio
import numpy as np
from std_msgs.msg import String
import threading
import queue

class RealTimeWhisperNode(Node):
    def __init__(self):
        super().__init__('realtime_whisper_node')
        
        # Load Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")
        
        # Publisher for recognized text
        self.text_pub = self.create_publisher(String, '/recognized_text', 10)
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper expects 16kHz
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Start audio capture thread
        self.audio_queue = queue.Queue()
        self.capture_thread = threading.Thread(target=self.capture_audio)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.get_logger().info('Real-time Whisper Node started')

    def capture_audio(self):
        """Capture audio from microphone."""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        try:
            while rclpy.ok():
                data = stream.read(self.chunk)
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                self.audio_queue.put(audio_data)
        except Exception as e:
            self.get_logger().error(f'Audio capture error: {e}')
        finally:
            stream.stop_stream()
            stream.close()

    def process_audio_stream(self):
        """Process audio stream in chunks."""
        buffer = []
        buffer_duration = 5.0  # Process 5-second chunks
        
        while rclpy.ok():
            try:
                # Collect audio data until we have enough
                while len(buffer) < self.rate * buffer_duration:
                    if not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        buffer.extend(chunk)
                    else:
                        # Wait a bit before checking again
                        import time
                        time.sleep(0.01)
                
                # Process the buffered audio
                audio_segment = np.array(buffer[:int(self.rate * buffer_duration)])
                buffer = buffer[int(self.rate * buffer_duration):]  # Keep remaining
                
                # Transcribe with Whisper
                result = self.model.transcribe(audio_segment, fp16=False)
                text = result['text'].strip()
                
                if text:
                    self.get_logger().info(f'Recognized: {text}')
                    
                    # Publish recognized text
                    text_msg = String()
                    text_msg.data = text
                    self.text_pub.publish(text_msg)
                    
            except Exception as e:
                self.get_logger().error(f'Processing error: {e}')
                buffer = []  # Clear buffer on error

    def destroy_node(self):
        """Clean up audio resources."""
        self.audio.terminate()
        super().destroy_node()
```

## Voice-to-Action Pipeline

### Integration with LLM Processing

Now let's create a complete voice-to-action pipeline that combines Whisper with LLM processing:

```python
import rclpy
from rclpy.node import Node
import whisper
import numpy as np
import pyaudio
import threading
import queue
import openai
import json
from std_msgs.msg import String
from sensor_msgs.msg import AudioData

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')
        
        # Load Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.whisper_model = whisper.load_model("base")
        
        # Initialize OpenAI client
        # In production, use environment variables for API keys
        openai.api_key = "YOUR_API_KEY"
        
        # Publishers and subscribers
        self.command_pub = self.create_publisher(String, '/robot_command', 10)
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_duration = 4.0  # Process 4-second chunks
        self.min_silence_duration = 0.5  # Minimum silence to trigger processing
        
        # Audio buffer and state
        self.audio_buffer = []
        self.processing_lock = threading.Lock()
        
        # VAD (Voice Activity Detection) parameters
        self.energy_threshold = 0.01  # Adjust based on environment
        self.silence_counter = 0
        self.voice_active = False
        
        self.get_logger().info('Voice-to-Action Node initialized')

    def audio_callback(self, msg):
        """Process incoming audio data."""
        # Convert audio data to numpy array
        audio_array = np.frombuffer(msg.data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Check for voice activity
        energy = np.mean(np.abs(audio_array))
        is_speech = energy > self.energy_threshold
        
        if is_speech:
            # Voice detected, add to buffer
            self.audio_buffer.extend(audio_array)
            self.voice_active = True
            self.silence_counter = 0
        else:
            # Silence detected
            if self.voice_active:
                self.silence_counter += len(audio_array) / self.sample_rate
                
                # If sufficient silence detected, process the buffer
                if self.silence_counter >= self.min_silence_duration:
                    if len(self.audio_buffer) > self.sample_rate * 0.5:  # At least 0.5s of speech
                        # Process in a separate thread to avoid blocking
                        buffer_copy = self.audio_buffer.copy()
                        self.audio_buffer = []
                        self.voice_active = False
                        threading.Thread(target=self.process_voice_command, args=(buffer_copy,)).start()
                    else:
                        # Too short, clear buffer
                        self.audio_buffer = []
                        self.voice_active = False
                else:
                    # Add silence to buffer for context
                    self.audio_buffer.extend(audio_array)

    def process_voice_command(self, audio_data):
        """Process voice command through Whisper and LLM."""
        with self.processing_lock:
            try:
                # Transcribe with Whisper
                self.get_logger().info('Processing voice command...')
                result = self.whisper_model.transcribe(np.array(audio_data), fp16=False)
                text = result['text'].strip()
                
                if not text:
                    self.get_logger().info('No speech detected')
                    return
                
                self.get_logger().info(f'Whisper recognized: {text}')
                
                # Process with LLM to generate robot command
                llm_command = self.process_with_llm(text)
                
                if llm_command:
                    # Publish robot command
                    cmd_msg = String()
                    cmd_msg.data = llm_command
                    self.command_pub.publish(cmd_msg)
                    self.get_logger().info(f'Published command: {llm_command}')
                
            except Exception as e:
                self.get_logger().error(f'Error processing voice command: {e}')

    def process_with_llm(self, text):
        """Process recognized text with LLM to generate robot command."""
        try:
            prompt = f"""
            Convert the following human command into a robot action:
            "{text}"
            
            Respond with a single, specific robot command that the robot can execute.
            Examples of commands: "move_to_kitchen", "pick_up_red_cup", "greet_visitor", "navigate_to_person"
            
            Only respond with the command, nothing else.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a robot command interpreter. Convert natural language to specific robot commands."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            command = response.choices[0].message.content.strip()
            return command
            
        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            return None
```

## Optimizing for Robotics Environments

### Noise Reduction and Audio Quality

Robotics environments often have significant background noise. Here's how to improve audio quality:

```python
import scipy.signal as signal

class OptimizedWhisperNode(VoiceToActionNode):
    def __init__(self):
        super().__init__()
        
        # Initialize noise reduction parameters
        self.noise_buffer = []
        self.noise_buffer_size = 10000  # Size for noise profile
        self.noise_profile = None
        
        # Audio enhancement parameters
        self.sample_rate = 16000
        self.high_freq_cutoff = 8000  # High-pass filter cutoff
        self.low_freq_cutoff = 100    # Low-pass filter cutoff
        
    def enhance_audio(self, audio_data):
        """Apply noise reduction and audio enhancement."""
        # Apply high-pass filter to remove low-frequency noise
        b, a = signal.butter(4, self.low_freq_cutoff / (self.sample_rate / 2), 'high')
        audio_data = signal.filtfilt(b, a, audio_data)
        
        # Apply low-pass filter to remove high-frequency noise
        b, a = signal.butter(4, self.high_freq_cutoff / (self.sample_rate / 2), 'low')
        audio_data = signal.filtfilt(b, a, audio_data)
        
        # Normalize audio
        audio_data = audio_data / max(np.max(np.abs(audio_data)), 1e-6)
        
        return audio_data
    
    def audio_callback(self, msg):
        """Process incoming audio with enhancement."""
        # Convert and enhance audio
        audio_array = np.frombuffer(msg.data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Apply enhancement
        enhanced_audio = self.enhance_audio(audio_array)
        
        # Continue with voice activity detection
        energy = np.mean(np.abs(enhanced_audio))
        is_speech = energy > self.energy_threshold
        
        # Rest of the processing logic...
        if is_speech:
            self.audio_buffer.extend(enhanced_audio)
            self.voice_active = True
            self.silence_counter = 0
        else:
            if self.voice_active:
                self.silence_counter += len(enhanced_audio) / self.sample_rate
                
                if self.silence_counter >= self.min_silence_duration:
                    if len(self.audio_buffer) > self.sample_rate * 0.5:
                        buffer_copy = self.audio_buffer.copy()
                        self.audio_buffer = []
                        self.voice_active = False
                        threading.Thread(target=self.process_voice_command, args=(buffer_copy,)).start()
                    else:
                        self.audio_buffer = []
                        self.voice_active = False
                else:
                    self.audio_buffer.extend(enhanced_audio)
```

### Performance Optimization

For real-time performance, consider these optimizations:

```python
class OptimizedVoiceToActionNode(OptimizedWhisperNode):
    def __init__(self):
        super().__init__()
        
        # Use faster Whisper model for real-time processing
        self.whisper_model = whisper.load_model("tiny")
        
        # Implement caching for common commands
        self.command_cache = {}
        self.cache_size = 50
        
        # Pre-compile common prompts for LLM
        self.llm_prompts = {
            'navigation': 'Convert navigation command to robot action: ',
            'manipulation': 'Convert manipulation command to robot action: ',
            'interaction': 'Convert interaction command to robot action: '
        }
    
    def process_with_llm(self, text):
        """Process with LLM using caching and optimization."""
        # Check cache first
        if text in self.command_cache:
            self.get_logger().info('Using cached command')
            return self.command_cache[text]
        
        # Classify command type for optimized prompt
        command_type = self.classify_command_type(text)
        prompt = self.llm_prompts.get(command_type, '') + text
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Convert natural language to specific robot commands."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,  # Limit response length
                temperature=0.1  # More deterministic output
            )
            
            command = response.choices[0].message.content.strip()
            
            # Cache the result
            if len(self.command_cache) < self.cache_size:
                self.command_cache[text] = command
            
            return command
            
        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')
            return None
    
    def classify_command_type(self, text):
        """Classify command type for optimized processing."""
        text_lower = text.lower()
        
        navigation_keywords = ['go', 'move', 'navigate', 'to', 'walk', 'drive', 'location']
        manipulation_keywords = ['pick', 'grasp', 'lift', 'put', 'place', 'take', 'grab', 'hold']
        interaction_keywords = ['hello', 'hi', 'greet', 'talk', 'speak', 'introduce', 'meet']
        
        if any(keyword in text_lower for keyword in navigation_keywords):
            return 'navigation'
        elif any(keyword in text_lower for keyword in manipulation_keywords):
            return 'manipulation'
        elif any(keyword in text_lower for keyword in interaction_keywords):
            return 'interaction'
        else:
            return 'general'
```

## Practical Example: Voice-Controlled Humanoid Robot

Here's a complete example that combines all concepts:

```python
import rclpy
from rclpy.node import Node
import whisper
import numpy as np
import threading
import queue
import openai
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import AudioData

class VoiceControlledHumanoid(Node):
    def __init__(self):
        super().__init__('voice_controlled_humanoid')
        
        # Load Whisper model
        self.whisper_model = whisper.load_model("base")
        
        # Publishers for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)
        
        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_input',
            self.audio_callback,
            10
        )
        
        # State variables
        self.audio_buffer = []
        self.voice_active = False
        self.silence_counter = 0
        self.energy_threshold = 0.01
        self.min_silence_duration = 0.5
        
        # Robot state
        self.is_listening = True
        
        self.get_logger().info('Voice Controlled Humanoid initialized')

    def audio_callback(self, msg):
        """Process audio input for voice commands."""
        audio_array = np.frombuffer(msg.data, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        energy = np.mean(np.abs(audio_array))
        is_speech = energy > self.energy_threshold
        
        if is_speech:
            self.audio_buffer.extend(audio_array)
            self.voice_active = True
            self.silence_counter = 0
        else:
            if self.voice_active:
                self.silence_counter += len(audio_array) / 16000
                
                if self.silence_counter >= self.min_silence_duration:
                    if len(self.audio_buffer) > 16000 * 0.5:
                        buffer_copy = self.audio_buffer.copy()
                        self.audio_buffer = []
                        self.voice_active = False
                        threading.Thread(target=self.process_command, args=(buffer_copy,)).start()
                    else:
                        self.audio_buffer = []
                        self.voice_active = False
                else:
                    self.audio_buffer.extend(audio_array)

    def process_command(self, audio_data):
        """Process voice command and execute robot action."""
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(np.array(audio_data), fp16=False)
            text = result['text'].strip().lower()
            
            if not text:
                return
                
            self.get_logger().info(f'Recognized: {text}')
            
            # Execute appropriate action based on command
            if 'forward' in text or 'straight' in text or 'go' in text:
                self.move_robot(0.5, 0.0)  # Move forward
                self.speak_response("Moving forward")
            elif 'backward' in text or 'back' in text:
                self.move_robot(-0.5, 0.0)  # Move backward
                self.speak_response("Moving backward")
            elif 'left' in text or 'turn left' in text:
                self.move_robot(0.0, 0.5)  # Turn left
                self.speak_response("Turning left")
            elif 'right' in text or 'turn right' in text:
                self.move_robot(0.0, -0.5)  # Turn right
                self.speak_response("Turning right")
            elif 'stop' in text:
                self.move_robot(0.0, 0.0)  # Stop
                self.speak_response("Stopping")
            elif 'hello' in text or 'hi' in text:
                self.speak_response("Hello! How can I help you?")
            else:
                self.speak_response(f"I heard: {text}. Command not recognized.")
                
        except Exception as e:
            self.get_logger().error(f'Command processing error: {e}')

    def move_robot(self, linear_vel, angular_vel):
        """Send movement command to robot."""
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)

    def speak_response(self, text):
        """Publish text for TTS."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
```

## Integration Considerations

### ROS 2 Launch File

Create a launch file to run the voice-to-action system:

```xml
<!-- launch/voice_to_action_launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_to_action',
            executable='whisper_speech_recognizer',
            name='whisper_speech_recognizer',
            parameters=[
                {'model_size': 'base'},
                {'sample_rate': 16000}
            ]
        ),
        Node(
            package='voice_to_action',
            executable='voice_to_action_processor',
            name='voice_to_action_processor',
            parameters=[
                {'energy_threshold': 0.01},
                {'min_silence_duration': 0.5}
            ]
        )
    ])
```

### Configuration Parameters

Consider making the system configurable:

```python
from rclpy.parameter import Parameter

class ConfigurableVoiceToActionNode(VoiceToActionNode):
    def __init__(self):
        super().__init__()
        
        # Declare parameters
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('energy_threshold', 0.01)
        self.declare_parameter('min_silence_duration', 0.5)
        self.declare_parameter('chunk_duration', 4.0)
        
        # Get parameter values
        model_size = self.get_parameter('model_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.energy_threshold = self.get_parameter('energy_threshold').value
        self.min_silence_duration = self.get_parameter('min_silence_duration').value
        self.chunk_duration = self.get_parameter('chunk_duration').value
        
        # Load appropriate Whisper model
        self.whisper_model = whisper.load_model(model_size)
```

## Best Practices for Voice-to-Action Systems

### 1. Robust Error Handling

```python
def safe_process_audio(self, audio_data):
    """Safely process audio with comprehensive error handling."""
    try:
        # Validate audio data
        if len(audio_data) == 0:
            return None
            
        # Check for valid audio range
        if np.max(np.abs(audio_data)) > 1.0:
            self.get_logger().warning('Audio may be clipped')
            
        # Process with Whisper
        result = self.whisper_model.transcribe(audio_data, fp16=False)
        return result['text'].strip()
        
    except Exception as e:
        self.get_logger().error(f'Audio processing error: {e}')
        return None
```

### 2. Resource Management

```python
def manage_resources(self):
    """Manage computational resources for optimal performance."""
    import psutil
    import gc
    
    # Check system resources
    memory_percent = psutil.virtual_memory().percent
    
    if memory_percent > 80:
        # Clear Whisper cache
        gc.collect()
        self.get_logger().warning('High memory usage, cleared cache')
```

### 3. Privacy and Security

For production systems, consider privacy and security:

```python
def process_audio_privately(self, audio_data):
    """Process audio without sending to cloud services."""
    # Use local Whisper model only
    result = self.whisper_model.transcribe(audio_data, fp16=False)
    text = result['text'].strip()
    
    # Process locally without sending to external LLMs if privacy required
    command = self.local_command_processing(text)
    return command
```

## Exercises

1. **Implementation Exercise**: Create a complete voice-to-action system that can recognize and execute basic robot commands like "move forward", "turn left", "stop", etc. Test with recorded audio samples.

2. **Optimization Exercise**: Implement the Whisper model with different sizes ("tiny", "base", "small") and measure the trade-off between accuracy and processing time.

3. **Integration Exercise**: Combine the voice-to-action system with the LLM integration from Chapter 13 to create a more sophisticated command interpretation system.

## Summary

This chapter covered the implementation of voice-to-action systems using OpenAI Whisper:

- Whisper provides robust speech recognition capabilities for robotics applications
- Voice-to-action pipelines involve audio capture, speech recognition, and command execution
- Real-time performance requires optimization techniques like model selection and caching
- Robotics environments require special consideration for noise and real-time constraints
- Integration with other AI systems (like LLMs) creates more sophisticated interaction

Voice-to-action systems enable more natural human-robot interaction, making humanoid robots more accessible and useful in human environments.

## Next Steps

In the next chapter, we'll explore cognitive planning systems that translate natural language into complex ROS 2 action sequences, building on both the LLM integration from Chapter 13 and the voice recognition from this chapter.