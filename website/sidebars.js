// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 and Robotic Control Foundations',
      items: [
        'module-1-ros-foundations/module-1-intro',
        'module-1-ros-foundations/chapter-1-middleware-control',
        'module-1-ros-foundations/chapter-2-nodes-topics-services',
        'module-1-ros-foundations/chapter-3-bridging-python-agents',
        'module-1-ros-foundations/chapter-4-urdf-humanoids',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twins using Gazebo and Unity',
      items: [
        'module-2-digital-twins/module-2-intro',
        'module-2-digital-twins/chapter-5-physics-simulation',
        'module-2-digital-twins/chapter-6-gazebo-simulations',
        'module-2-digital-twins/chapter-7-unity-interaction',
        'module-2-digital-twins/chapter-8-sensor-simulation',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI Perception and Navigation with NVIDIA Isaac',
      items: [
        'module-3-ai-navigation/module-3-intro',
        'module-3-ai-navigation/chapter-9-advanced-perception',
        'module-3-ai-navigation/chapter-10-isaac-sim-generation',
        'module-3-ai-navigation/chapter-11-isaac-ros-navigation',
        'module-3-ai-navigation/chapter-12-nav2-path-planning',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems and Capstone',
      items: [
        'module-4-vla-systems/module-4-intro',
        'module-4-vla-systems/chapter-13-llm-robotics',
        'module-4-vla-systems/chapter-14-voice-to-action',
        'module-4-vla-systems/chapter-15-cognitive-planning',
        'module-4-vla-systems/chapter-16-autonomous-humanoid',
      ],
    },
  ],
};

export default sidebars;
