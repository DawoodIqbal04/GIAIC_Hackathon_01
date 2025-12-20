---
id: chapter-15-cognitive-planning
title: "Chapter 15: Cognitive Planning: Translating Natural Language into ROS 2 Action Sequences"
sidebar_label: "Chapter 15: Cognitive Planning"
description: "Cognitive planning systems that translate natural language into complex ROS 2 action sequences for humanoid robots"
keywords: [cognitive-planning, natural-language, ros2-actions, task-planning, ai, robotics]
tags: [ai-planning, task-execution, robotics]
authors: [book-authors]
difficulty: advanced
estimated_time: "120 minutes"
module: 4
chapter: 15
prerequisites: [python-ai-basics, ros2-foundations, llm-integration, action-systems]
learning_objectives:
  - Design cognitive planning systems for natural language interpretation
  - Implement action sequence generation from natural language commands
  - Integrate cognitive planning with ROS 2 action servers
  - Handle complex task decomposition and execution
  - Validate and verify action sequences for safety and correctness
related:
  - next: chapter-16-autonomous-humanoid
  - previous: chapter-14-voice-to-action
  - see_also: [chapter-13-llm-robotics, chapter-14-voice-to-action, chapter-16-autonomous-humanoid]
---

# Chapter 15: Cognitive Planning: Translating Natural Language into ROS 2 Action Sequences

## Learning Objectives

After completing this chapter, you will be able to:
- Design cognitive planning architectures for natural language interpretation
- Implement action sequence generation from natural language commands
- Integrate cognitive planning with ROS 2 action servers for complex tasks
- Handle task decomposition, execution, and error recovery
- Validate and verify action sequences for safety and correctness
- Create robust cognitive planning systems for humanoid robots

## Introduction

Cognitive planning represents the bridge between high-level natural language commands and low-level robot execution. It involves interpreting human instructions, decomposing complex tasks into executable actions, and managing the execution of these actions in a way that's both effective and safe. This chapter explores how to build cognitive planning systems that can translate natural language into complex ROS 2 action sequences for humanoid robots.

Cognitive planning systems must handle several key challenges:
- **Ambiguity Resolution**: Interpreting imprecise human language
- **Task Decomposition**: Breaking down complex tasks into manageable steps
- **Context Awareness**: Understanding the environment and constraints
- **Execution Monitoring**: Tracking progress and handling failures
- **Safety Assurance**: Ensuring actions are safe to execute

## Cognitive Planning Architecture

### High-Level Architecture

The cognitive planning system typically consists of several interconnected components:

```
Natural Language Command
         ↓
   [Language Understanding]
         ↓
   [Task Decomposition]
         ↓
   [Action Sequencing]
         ↓
   [Execution Management]
         ↓
   [ROS 2 Action Execution]
         ↓
   Physical Robot Actions
```

### Component Architecture

A more detailed architecture includes:

1. **Language Interface**: Processes natural language input
2. **World Model**: Maintains current state and context
3. **Planner**: Generates action sequences
4. **Executor**: Manages action execution
5. **Monitor**: Tracks execution and handles failures
6. **Validator**: Ensures safety and correctness

## Implementing Cognitive Planning with LLMs

### Basic Cognitive Planner

Here's a basic implementation of a cognitive planner that uses an LLM for task decomposition:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
import openai
import json
import time
from typing import List, Dict, Any

class CognitivePlanner(Node):
    def __init__(self):
        super().__init__('cognitive_planner')
        
        # Initialize OpenAI client
        # In production, use environment variables for API keys
        openai.api_key = "YOUR_API_KEY"
        
        # Publishers and subscribers
        self.action_sequence_pub = self.create_publisher(
            String, 
            '/action_sequence', 
            10
        )
        
        self.command_sub = self.create_subscription(
            String,
            '/natural_language_command',
            self.command_callback,
            10
        )
        
        # World model (simplified)
        self.world_state = {
            'robot_location': 'unknown',
            'objects': [],
            'robot_capabilities': ['navigate', 'grasp', 'speak'],
            'environment_constraints': []
        }
        
        self.get_logger().info('Cognitive Planner initialized')

    def command_callback(self, msg):
        """Process natural language command."""
        try:
            # Plan actions based on command and world state
            action_sequence = self.plan_actions(msg.data, self.world_state)
            
            if action_sequence:
                # Publish action sequence
                action_msg = String()
                action_msg.data = json.dumps(action_sequence)
                self.action_sequence_pub.publish(action_msg)
                
                self.get_logger().info(f'Planned sequence: {action_sequence}')
            else:
                self.get_logger().error('Failed to generate action sequence')
                
        except Exception as e:
            self.get_logger().error(f'Planning error: {e}')

    def plan_actions(self, command: str, world_state: Dict) -> List[Dict]:
        """Plan actions using LLM."""
        try:
            prompt = f"""
            Given the following command and world state, generate a sequence of actions to fulfill the command.
            
            Command: "{command}"
            World State: {world_state}
            
            Respond with a JSON array of action objects. Each action should have:
            - "action_type": The type of action (e.g., "navigate", "grasp", "speak", "detect_object")
            - "parameters": Parameters needed for the action
            - "description": Brief description of the action
            
            Example response format:
            [
              {{
                "action_type": "detect_object",
                "parameters": {{"object_type": "red_cup", "location": "kitchen"}},
                "description": "Look for a red cup in the kitchen"
              }},
              {{
                "action_type": "navigate",
                "parameters": {{"target_location": "kitchen"}},
                "description": "Go to the kitchen"
              }}
            ]
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cognitive planner for a robot. Generate executable action sequences from natural language commands."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            action_sequence = json.loads(response.choices[0].message.content)
            return action_sequence
            
        except Exception as e:
            self.get_logger().error(f'LLM planning error: {e}')
            return []

    def update_world_state(self, state_update: Dict):
        """Update world state with new information."""
        self.world_state.update(state_update)
```

### Advanced Cognitive Planner with Context

A more sophisticated planner maintains detailed context and handles complex scenarios:

```python
class AdvancedCognitivePlanner(CognitivePlanner):
    def __init__(self):
        super().__init__()
        
        # Maintain execution history
        self.execution_history = []
        
        # Task queue for multi-step planning
        self.task_queue = []
        
        # Safety validator
        self.safety_validator = SafetyValidator()

    def plan_actions(self, command: str, world_state: Dict) -> List[Dict]:
        """Enhanced planning with context and safety checks."""
        try:
            # Create detailed prompt with context
            prompt = self.create_detailed_prompt(command, world_state)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use more capable model for complex planning
                messages=[
                    {
                        "role": "system", 
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1  # More deterministic for planning
            )
            
            # Parse action sequence
            raw_response = response.choices[0].message.content
            
            # Extract JSON from response (in case of extra text)
            action_sequence = self.extract_json_from_response(raw_response)
            
            # Validate safety
            if self.safety_validator.validate_sequence(action_sequence, world_state):
                return action_sequence
            else:
                self.get_logger().error('Action sequence failed safety validation')
                return []
            
        except Exception as e:
            self.get_logger().error(f'Enhanced planning error: {e}')
            return []

    def create_detailed_prompt(self, command: str, world_state: Dict) -> str:
        """Create detailed prompt with all necessary context."""
        return f"""
        As a cognitive planner for a humanoid robot, generate a sequence of actions to fulfill the given command.
        
        COMMAND: {command}
        
        CURRENT WORLD STATE:
        - Robot Location: {world_state.get('robot_location', 'unknown')}
        - Available Objects: {world_state.get('objects', [])}
        - Robot Capabilities: {world_state.get('robot_capabilities', [])}
        - Environment Constraints: {world_state.get('environment_constraints', [])}
        - Recent Actions: {self.execution_history[-5:]}  # Last 5 actions
        
        REQUIREMENTS:
        1. Actions must be executable by the robot (based on capabilities)
        2. Consider environment constraints and safety
        3. Account for robot's current location
        4. Include error handling where appropriate
        5. Be specific with parameters
        
        RESPONSE FORMAT:
        Return ONLY a JSON array of action objects with these fields:
        - "action_type": The type of action
        - "parameters": Object with action-specific parameters
        - "description": Human-readable description
        - "expected_duration": Estimated time in seconds
        - "success_criteria": How to verify action completion
        
        Example valid response:
        [
          {{
            "action_type": "navigate",
            "parameters": {{"target_location": "kitchen", "avoid_obstacles": true}},
            "description": "Navigate to the kitchen while avoiding obstacles",
            "expected_duration": 30,
            "success_criteria": "Robot reaches kitchen area"
          }},
          {{
            "action_type": "detect_object",
            "parameters": {{"object_type": "red_cup", "search_area": "counter"}},
            "description": "Look for a red cup on the counter",
            "expected_duration": 10,
            "success_criteria": "Red cup detected with confidence > 0.8"
          }}
        ]
        """

    def get_system_prompt(self) -> str:
        """Get system prompt for cognitive planner."""
        return """
        You are an advanced cognitive planner for a humanoid robot. Your role is to:
        1. Interpret natural language commands
        2. Generate executable action sequences
        3. Consider environmental constraints and safety
        4. Account for robot capabilities and limitations
        5. Include error handling and verification steps
        
        Always respond with a valid JSON array of action objects. Each action must be:
        - Feasible given robot capabilities
        - Safe to execute
        - Specific with required parameters
        - Verifiable with clear success criteria
        """

    def extract_json_from_response(self, response: str) -> List[Dict]:
        """Extract JSON from LLM response that might contain extra text."""
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Look for JSON within the response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
```

## ROS 2 Action Integration

### Action Sequence Executor

Now let's implement an executor that runs the planned action sequences:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory
import json
import asyncio
from typing import List, Dict, Any

class ActionSequenceExecutor(Node):
    def __init__(self):
        super().__init__('action_sequence_executor')
        
        # Action clients for different robot capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.arm_client = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        
        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, '/action_status', 10)
        
        self.action_sequence_sub = self.create_subscription(
            String,
            '/action_sequence',
            self.action_sequence_callback,
            10
        )
        
        # Current execution state
        self.current_sequence = []
        self.current_action_index = 0
        self.is_executing = False
        
        self.get_logger().info('Action Sequence Executor initialized')

    def action_sequence_callback(self, msg):
        """Receive and execute action sequence."""
        try:
            # Parse action sequence
            sequence = json.loads(msg.data)
            
            if not self.is_executing:
                # Start execution of the new sequence
                self.current_sequence = sequence
                self.current_action_index = 0
                self.is_executing = True
                
                # Execute the sequence
                self.execute_sequence()
            else:
                # Queue the sequence or handle concurrent execution
                self.get_logger().warning('Already executing a sequence, queuing new one')
                
        except Exception as e:
            self.get_logger().error(f'Error processing action sequence: {e}')

    async def execute_sequence(self):
        """Execute the current action sequence."""
        for i, action in enumerate(self.current_sequence):
            self.current_action_index = i
            
            status_msg = String()
            status_msg.data = f"Executing action {i+1}/{len(self.current_sequence)}: {action.get('description', 'Unknown')}"
            self.status_pub.publish(status_msg)
            
            success = await self.execute_action(action)
            
            if not success:
                self.get_logger().error(f'Action failed: {action}')
                # Handle failure - stop execution or try recovery
                break
        
        # Sequence completed
        self.is_executing = False
        status_msg = String()
        status_msg.data = "Action sequence completed"
        self.status_pub.publish(status_msg)

    async def execute_action(self, action: Dict) -> bool:
        """Execute a single action based on its type."""
        action_type = action.get('action_type')
        
        if action_type == 'navigate':
            return await self.execute_navigation_action(action)
        elif action_type == 'grasp':
            return await self.execute_grasp_action(action)
        elif action_type == 'speak':
            return await self.execute_speak_action(action)
        elif action_type == 'detect_object':
            return await self.execute_detection_action(action)
        else:
            self.get_logger().error(f'Unknown action type: {action_type}')
            return False

    async def execute_navigation_action(self, action: Dict) -> bool:
        """Execute navigation action."""
        try:
            target_location = action['parameters']['target_location']
            
            # For this example, we'll use a simplified navigation
            # In practice, you'd look up coordinates for named locations
            pose = self.get_pose_for_location(target_location)
            
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose
            
            self.get_logger().info(f'Navigating to {target_location}')
            
            # Send navigation goal
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Navigation action server not available')
                return False
            
            future = self.nav_client.send_goal_async(goal_msg)
            goal_handle = await future
            
            if not goal_handle.accepted:
                self.get_logger().error('Navigation goal rejected')
                return False
            
            result_future = goal_handle.get_result_async()
            result = await result_future
            
            return result.result.status == 1  # SUCCESS
            
        except Exception as e:
            self.get_logger().error(f'Navigation error: {e}')
            return False

    def get_pose_for_location(self, location: str) -> PoseStamped:
        """Get pose for a named location (simplified)."""
        # In a real system, this would look up coordinates from a map
        poses = {
            'kitchen': PoseStamped(),
            'living_room': PoseStamped(),
            'bedroom': PoseStamped()
        }
        return poses.get(location, PoseStamped())

    async def execute_grasp_action(self, action: Dict) -> bool:
        """Execute grasp action."""
        try:
            object_name = action['parameters']['object_name']
            
            self.get_logger().info(f'Attempting to grasp {object_name}')
            
            # This would involve complex manipulation planning
            # For now, we'll simulate the action
            await asyncio.sleep(5.0)  # Simulate grasp time
            
            return True  # Simulated success
            
        except Exception as e:
            self.get_logger().error(f'Grasp error: {e}')
            return False

    async def execute_speak_action(self, action: Dict) -> bool:
        """Execute speak action."""
        try:
            text = action['parameters']['text']
            
            self.get_logger().info(f'Speaking: {text}')
            
            # Publish to TTS system
            tts_pub = self.create_publisher(String, '/tts_input', 10)
            msg = String()
            msg.data = text
            tts_pub.publish(msg)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Speak error: {e}')
            return False

    async def execute_detection_action(self, action: Dict) -> bool:
        """Execute object detection action."""
        try:
            object_type = action['parameters']['object_type']
            search_area = action['parameters'].get('search_area', 'all')
            
            self.get_logger().info(f'Detecting {object_type} in {search_area}')
            
            # This would involve calling perception systems
            # For now, we'll simulate detection
            await asyncio.sleep(2.0)  # Simulate detection time
            
            # Simulate detection result
            detected = True  # In real system, this would come from perception
            
            if detected:
                self.get_logger().info(f'{object_type} detected successfully')
                return True
            else:
                self.get_logger().info(f'{object_type} not found')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            return False
```

## Safety and Validation Systems

### Safety Validator

Safety is critical for cognitive planning systems, especially for humanoid robots:

```python
class SafetyValidator:
    def __init__(self):
        # Define safety rules
        self.safety_rules = [
            self.check_navigation_safety,
            self.check_manipulation_safety,
            self.check_human_safety,
            self.check_environment_safety
        ]
    
    def validate_sequence(self, sequence: List[Dict], world_state: Dict) -> bool:
        """Validate an action sequence for safety."""
        for action in sequence:
            for rule in self.safety_rules:
                if not rule(action, world_state):
                    return False
        return True
    
    def check_navigation_safety(self, action: Dict, world_state: Dict) -> bool:
        """Check if navigation action is safe."""
        if action.get('action_type') != 'navigate':
            return True
            
        target = action.get('parameters', {}).get('target_location')
        
        # Check if target location is known to be safe
        unsafe_locations = world_state.get('unsafe_locations', [])
        if target in unsafe_locations:
            return False
            
        return True
    
    def check_manipulation_safety(self, action: Dict, world_state: Dict) -> bool:
        """Check if manipulation action is safe."""
        if action.get('action_type') not in ['grasp', 'place', 'move_object']:
            return True
            
        # Check if target object is safe to manipulate
        obj_name = action.get('parameters', {}).get('object_name')
        fragile_objects = world_state.get('fragile_objects', [])
        
        if obj_name in fragile_objects:
            # Check if action is appropriate for fragile object
            if action.get('action_type') == 'grasp':
                return True  # Assume grasp is safe with proper planning
            else:
                return False  # Other actions might be unsafe
                
        return True
    
    def check_human_safety(self, action: Dict, world_state: Dict) -> bool:
        """Check if action is safe regarding humans."""
        # Check if action involves moving near humans
        if action.get('action_type') == 'navigate':
            target = action.get('parameters', {}).get('target_location')
            humans_nearby = world_state.get('humans_nearby', [])
            
            # For this example, assume navigating near humans needs caution
            if target in humans_nearby:
                # Check if action includes safety measures
                safety_measures = action.get('parameters', {}).get('safety_measures', [])
                if 'approach_slowly' in safety_measures or 'request_permission' in safety_measures:
                    return True
                else:
                    return False
        
        return True
    
    def check_environment_safety(self, action: Dict, world_state: Dict) -> bool:
        """Check if action is safe for the environment."""
        # Check environmental constraints
        constraints = world_state.get('environment_constraints', [])
        
        if action.get('action_type') == 'navigate':
            target = action.get('parameters', {}).get('target_location')
            if f'no_navigation_to_{target}' in constraints:
                return False
                
        return True
```

## Error Handling and Recovery

### Execution Monitor with Recovery

Robust cognitive planning systems need error handling and recovery mechanisms:

```python
import asyncio
from enum import Enum

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RECOVERING = "recovering"

class MonitoredActionExecutor(ActionSequenceExecutor):
    def __init__(self):
        super().__init__()
        
        # Execution monitoring
        self.execution_status = ExecutionStatus.PENDING
        self.failure_count = 0
        self.max_failures = 3
        self.recovery_attempts = 0
        self.max_recovery_attempts = 2
        
        # Action history for learning
        self.action_history = []

    async def execute_sequence(self):
        """Execute sequence with monitoring and recovery."""
        self.execution_status = ExecutionStatus.RUNNING
        
        for i, action in enumerate(self.current_sequence):
            self.current_action_index = i
            
            # Update status
            status_msg = String()
            status_msg.data = f"Executing action {i+1}/{len(self.current_sequence)}: {action.get('description', 'Unknown')}"
            self.status_pub.publish(status_msg)
            
            success = await self.execute_monitored_action(action)
            
            if not success:
                self.get_logger().error(f'Action failed: {action}')
                
                # Attempt recovery
                recovery_success = await self.attempt_recovery(action)
                
                if not recovery_success:
                    self.execution_status = ExecutionStatus.FAILED
                    break
            else:
                # Record successful action
                self.action_history.append({
                    'action': action,
                    'status': 'success',
                    'timestamp': self.get_clock().now().nanoseconds
                })
        
        # Sequence completed
        if self.execution_status != ExecutionStatus.FAILED:
            self.execution_status = ExecutionStatus.SUCCESS
            
        self.is_executing = False
        status_msg = String()
        status_msg.data = f"Action sequence {self.execution_status.value}"
        self.status_pub.publish(status_msg)

    async def execute_monitored_action(self, action: Dict) -> bool:
        """Execute action with monitoring."""
        try:
            # Record action start
            start_time = self.get_clock().now().nanoseconds
            
            success = await self.execute_action(action)
            
            # Record action end and status
            end_time = self.get_clock().now().nanoseconds
            duration = (end_time - start_time) / 1e9  # Convert to seconds
            
            self.action_history.append({
                'action': action,
                'status': 'success' if success else 'failed',
                'duration': duration,
                'timestamp': start_time
            })
            
            return success
            
        except Exception as e:
            self.get_logger().error(f'Exception in action execution: {e}')
            return False

    async def attempt_recovery(self, failed_action: Dict) -> bool:
        """Attempt to recover from action failure."""
        self.failure_count += 1
        
        if self.failure_count > self.max_failures:
            self.get_logger().error('Max failures reached, stopping execution')
            return False
        
        self.get_logger().info(f'Attempting recovery, attempt {self.failure_count}')
        
        # Different recovery strategies based on action type
        recovery_strategy = self.select_recovery_strategy(failed_action)
        
        if recovery_strategy == "retry":
            self.get_logger().info('Retrying failed action')
            return await self.execute_monitored_action(failed_action)
        elif recovery_strategy == "alternative":
            self.get_logger().info('Using alternative approach')
            alternative_action = self.generate_alternative_action(failed_action)
            if alternative_action:
                return await self.execute_monitored_action(alternative_action)
        elif recovery_strategy == "skip":
            self.get_logger().info('Skipping failed action')
            return True  # Consider it successful to continue
        else:
            self.get_logger().info('No recovery strategy available')
            return False

    def select_recovery_strategy(self, failed_action: Dict) -> str:
        """Select appropriate recovery strategy."""
        action_type = failed_action.get('action_type')
        
        # Simple strategy selection - in practice, this would be more sophisticated
        if action_type in ['navigate', 'detect_object']:
            return "alternative"  # Try different approach
        elif action_type in ['speak']:
            return "skip"  # Skip non-critical actions
        else:
            return "retry"  # Retry other actions

    def generate_alternative_action(self, failed_action: Dict) -> Dict:
        """Generate an alternative action to replace the failed one."""
        action_type = failed_action.get('action_type')
        
        if action_type == 'navigate':
            # Try alternative navigation approach
            original_target = failed_action['parameters']['target_location']
            alternative_target = self.find_alternative_location(original_target)
            
            if alternative_target:
                return {
                    'action_type': 'navigate',
                    'parameters': {'target_location': alternative_target},
                    'description': f'Navigate to alternative location {alternative_target}',
                    'expected_duration': 35,  # Slightly longer
                    'success_criteria': f'Robot reaches {alternative_target}'
                }
        
        elif action_type == 'detect_object':
            # Try different detection approach
            return {
                **failed_action,  # Keep most parameters
                'parameters': {
                    **failed_action.get('parameters', {}),
                    'detection_method': 'multi_view'  # Use different method
                },
                'description': failed_action.get('description', '') + ' (alternative method)'
            }
        
        return None

    def find_alternative_location(self, original_location: str) -> str:
        """Find an alternative location for navigation."""
        # In a real system, this would access a map of location relationships
        location_alternatives = {
            'kitchen': 'dining_room',
            'bedroom': 'living_room',
            'office': 'study',
        }
        return location_alternatives.get(original_location, None)
```

## Practical Example: Complete Cognitive Planning System

Here's a complete example that combines all the concepts:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import openai
import json
import asyncio
import threading
from typing import List, Dict, Any

class CompleteCognitivePlanningSystem(Node):
    def __init__(self):
        super().__init__('complete_cognitive_planning_system')
        
        # Initialize components
        self.planner = AdvancedCognitivePlanner()
        self.executor = MonitoredActionExecutor()
        self.validator = SafetyValidator()
        
        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String,
            '/natural_language_command',
            self.command_callback,
            10
        )
        
        self.world_state_sub = self.create_subscription(
            String,
            '/world_state',
            self.world_state_callback,
            10
        )
        
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        
        # System state
        self.current_world_state = {}
        self.command_queue = []
        
        self.get_logger().info('Complete Cognitive Planning System initialized')

    def command_callback(self, msg):
        """Process natural language command."""
        command = msg.data
        
        # Plan and execute in a separate thread to avoid blocking
        thread = threading.Thread(target=self.process_command, args=(command,))
        thread.daemon = True
        thread.start()

    def process_command(self, command: str):
        """Process command with full cognitive planning pipeline."""
        try:
            # Plan actions
            action_sequence = self.planner.plan_actions(command, self.current_world_state)
            
            if not action_sequence:
                self.get_logger().error('Failed to plan actions')
                return
            
            # Validate safety
            if not self.validator.validate_sequence(action_sequence, self.current_world_state):
                self.get_logger().error('Action sequence failed safety validation')
                return
            
            # Execute sequence
            self.executor.current_sequence = action_sequence
            asyncio.run(self.executor.execute_sequence())
            
        except Exception as e:
            self.get_logger().error(f'Command processing error: {e}')

    def world_state_callback(self, msg):
        """Update world state."""
        try:
            state_update = json.loads(msg.data)
            self.current_world_state.update(state_update)
            
            # Update planner's world state
            self.planner.update_world_state(state_update)
            
        except json.JSONDecodeError:
            self.get_logger().error('Invalid world state format')

    def get_system_status(self) -> str:
        """Get current system status."""
        status = {
            'planner_status': 'ready',
            'executor_status': self.executor.execution_status.value if hasattr(self.executor, 'execution_status') else 'unknown',
            'world_state_known_objects': len(self.current_world_state.get('objects', [])),
            'command_queue_size': len(self.command_queue)
        }
        return json.dumps(status)
```

## Integration with ROS 2 Ecosystem

### Launch File for Cognitive Planning System

```xml
<!-- launch/cognitive_planning_launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cognitive_planning',
            executable='cognitive_planner',
            name='cognitive_planner',
            parameters=[
                {'model_name': 'gpt-4'},
                {'max_failures': 3}
            ],
            remappings=[
                ('/natural_language_command', '/voice_command'),
                ('/action_sequence', '/planned_actions')
            ]
        ),
        Node(
            package='cognitive_planning',
            executable='action_executor',
            name='action_executor',
            parameters=[
                {'max_recovery_attempts': 2}
            ]
        ),
        Node(
            package='cognitive_planning',
            executable='world_state_manager',
            name='world_state_manager'
        )
    ])
```

### Parameter Configuration

```yaml
# config/cognitive_planning_params.yaml
cognitive_planner:
  ros__parameters:
    model_name: "gpt-4"
    max_context_length: 4096
    planning_timeout: 30.0
    safety_check_enabled: true
    verbose_logging: false

action_executor:
  ros__parameters:
    max_failures: 3
    max_recovery_attempts: 2
    action_timeout: 60.0
    enable_monitoring: true

world_state_manager:
  ros__parameters:
    update_frequency: 10.0
    state_history_size: 100
    enable_fusion: true
```

## Best Practices for Cognitive Planning

### 1. Hierarchical Planning

Implement multiple levels of planning:

```python
class HierarchicalCognitivePlanner:
    def __init__(self):
        self.high_level_planner = HighLevelPlanner()
        self.mid_level_planner = MidLevelPlanner()
        self.low_level_planner = LowLevelPlanner()
    
    def plan_task(self, high_level_goal: str, world_state: Dict) -> List[Dict]:
        """Plan task using hierarchical approach."""
        # High-level: Decompose into subgoals
        subgoals = self.high_level_planner.decompose_goal(high_level_goal)
        
        # Mid-level: Convert subgoals to action sequences
        action_sequences = []
        for subgoal in subgoals:
            sequence = self.mid_level_planner.plan_subgoal(subgoal, world_state)
            action_sequences.extend(sequence)
        
        # Low-level: Add implementation details
        detailed_sequence = self.low_level_planner.add_details(action_sequences, world_state)
        
        return detailed_sequence
```

### 2. Continuous Learning

Implement learning from execution outcomes:

```python
class LearningCognitivePlanner(AdvancedCognitivePlanner):
    def __init__(self):
        super().__init__()
        
        # Learning data structures
        self.action_success_rate = {}
        self.environment_models = {}
        self.user_preferences = {}
    
    def update_from_execution(self, action: Dict, outcome: str, context: Dict):
        """Update models based on execution outcome."""
        action_type = action.get('action_type')
        
        # Update success rate for this action type
        if action_type not in self.action_success_rate:
            self.action_success_rate[action_type] = {'success': 0, 'total': 0}
        
        stats = self.action_success_rate[action_type]
        stats['total'] += 1
        if outcome == 'success':
            stats['success'] += 1
        
        # Update environment model based on context
        env_key = self.get_environment_key(context)
        if env_key not in self.environment_models:
            self.environment_models[env_key] = {}
        
        # Update user preferences
        user_id = context.get('user_id', 'unknown')
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Store successful action patterns
        self.store_successful_pattern(action, context)
    
    def get_environment_key(self, context: Dict) -> str:
        """Get key for environment context."""
        location = context.get('location', 'unknown')
        time_of_day = context.get('time_of_day', 'unknown')
        return f"{location}_{time_of_day}"
    
    def store_successful_pattern(self, action: Dict, context: Dict):
        """Store patterns of successful actions."""
        # Implementation to store successful action patterns
        pass
```

### 3. Multi-Modal Integration

Combine multiple input modalities:

```python
class MultiModalCognitivePlanner(AdvancedCognitivePlanner):
    def __init__(self):
        super().__init__()
        
        # Additional input handlers
        self.vision_handler = VisionHandler()
        self.tactile_handler = TactileHandler()
        self.audio_handler = AudioHandler()
    
    def plan_with_multimodal_input(self, command: str, vision_data: Dict, 
                                   tactile_data: Dict, audio_data: Dict) -> List[Dict]:
        """Plan using multiple input modalities."""
        # Combine all modalities into world state
        multimodal_state = self.fuse_modalities(
            self.world_state,
            vision_data,
            tactile_data,
            audio_data
        )
        
        # Plan using enhanced world state
        return self.plan_actions(command, multimodal_state)
    
    def fuse_modalities(self, base_state: Dict, vision_data: Dict, 
                        tactile_data: Dict, audio_data: Dict) -> Dict:
        """Fuse information from multiple modalities."""
        fused_state = base_state.copy()
        
        # Integrate vision data
        fused_state['visual_objects'] = vision_data.get('objects', [])
        fused_state['spatial_layout'] = vision_data.get('layout', {})
        
        # Integrate tactile data
        fused_state['contact_info'] = tactile_data.get('contacts', [])
        fused_state['grasp_quality'] = tactile_data.get('grasp_quality', {})
        
        # Integrate audio data
        fused_state['audio_events'] = audio_data.get('events', [])
        fused_state['speaker_locations'] = audio_data.get('speakers', [])
        
        return fused_state
```

## Exercises

1. **Implementation Exercise**: Create a cognitive planning system that can handle a multi-step task like "Go to the kitchen, find the red cup, pick it up, and bring it to me." Implement safety checks and error recovery.

2. **Design Exercise**: Design a cognitive planning architecture that can handle ambiguous commands like "Clean up the room." Consider how the system would decompose this high-level task into specific actions.

3. **Integration Exercise**: Integrate the cognitive planning system with a simulation environment (like Gazebo) to test the planning and execution pipeline with a virtual humanoid robot.

## Summary

This chapter covered cognitive planning systems that translate natural language into ROS 2 action sequences:

- Cognitive planning bridges natural language understanding and robot execution
- Systems must handle ambiguity, decomposition, context, and safety
- Integration with ROS 2 actions enables complex task execution
- Safety validation and error recovery are critical for humanoid robots
- Hierarchical planning and learning improve system performance over time

Cognitive planning systems enable humanoid robots to understand and execute complex natural language commands safely and effectively.

## Next Steps

In the final chapter of this module, we'll implement the complete capstone project that integrates all the concepts learned throughout the book: ROS 2 communication, simulation, perception, and AI integration into a comprehensive autonomous humanoid robot system.