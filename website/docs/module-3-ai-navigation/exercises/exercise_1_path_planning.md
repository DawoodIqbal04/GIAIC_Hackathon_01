# Exercise 1: Implement A* Path Planning Algorithm

## Problem Statement
Implement the A* path planning algorithm for a 2D grid map. The algorithm should find the shortest path from a start position to a goal position while avoiding obstacles.

## Learning Objectives
- Understand the fundamentals of graph search algorithms
- Implement heuristic-based path planning
- Apply A* to robotics navigation problems
- Evaluate path optimality and computational efficiency

## Implementation Requirements

### 1. Grid Map Representation
- Create a 2D grid map where each cell can be free (0) or occupied (1)
- Implement a function to load maps from text files or generate random maps
- Include visualization of the map, start position, goal position, and computed path

### 2. A* Algorithm Implementation
- Implement the core A* algorithm with appropriate data structures (priority queue, open/closed sets)
- Use Manhattan distance as the heuristic function
- Handle diagonal movement as an optional enhancement
- Return the complete path as a sequence of coordinates

### 3. Visualization
- Visualize the exploration process (show which cells were visited)
- Highlight the final path on the grid
- Differentiate between free space, obstacles, start, goal, and path

## Starter Code Template

```python
import numpy as np
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class AStarPlanner:
    def __init__(self, grid: np.ndarray):
        """
        Initialize the A* planner with a grid map.
        
        Args:
            grid: 2D numpy array where 0 represents free space and 1 represents obstacles
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate the heuristic distance between two positions (Manhattan distance).
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Heuristic distance between the positions
        """
        # TODO: Implement Manhattan distance heuristic
        pass
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighbors for a given position.
        
        Args:
            pos: Current position (row, col)
            
        Returns:
            List of valid neighbor positions
        """
        # TODO: Implement neighbor generation (4-connected or 8-connected)
        pass
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan a path from start to goal using A* algorithm.
        
        Args:
            start: Start position (row, col)
            goal: Goal position (row, col)
            
        Returns:
            List of positions representing the path, or None if no path exists
        """
        # TODO: Implement A* algorithm
        pass
    
    def visualize(self, path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Visualize the grid, path, start, and goal positions.
        
        Args:
            path: List of positions representing the path
            start: Start position
            goal: Goal position
        """
        # TODO: Implement visualization using matplotlib
        pass

# Example usage:
if __name__ == "__main__":
    # Create a sample grid (0 = free space, 1 = obstacle)
    grid = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    planner = AStarPlanner(grid)
    start = (0, 0)
    goal = (9, 9)
    
    path = planner.plan(start, goal)
    
    if path:
        print(f"Path found with {len(path)} steps")
        planner.visualize(path, start, goal)
    else:
        print("No path found")
```

## Evaluation Criteria
- Correctness: The algorithm should find the optimal path when one exists
- Completeness: The algorithm should handle all possible grid configurations
- Efficiency: The algorithm should run within reasonable time limits
- Code Quality: Well-documented, readable, and maintainable code
- Visualization: Clear visualization of the planning process and results

## Hints and Resources
- Use a priority queue (heapq in Python) for the open set
- Remember to track the cost from start (g-score) and estimated total cost (f-score = g + h)
- Consider using a dictionary to store g-scores and parent pointers
- For 8-connected movement, include diagonal neighbors with appropriate cost

## Extensions
- Implement different heuristic functions (Euclidean, Diagonal distance)
- Add support for weighted grids where different terrains have different traversal costs
- Implement any-angle path planning using Theta* algorithm
- Integrate with ROS/Nav2 for real robot navigation