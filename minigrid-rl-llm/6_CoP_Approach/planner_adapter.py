import numpy as np
from generated_planner import planner as original_planner

def adapted_planner(observation):
    """
    Adapter that converts fully observable MiniGrid observations to the format
    expected by the original generated planner (7x7 partial observation).
    """
    
    # Get the full grid
    full_grid = observation['image']
    direction = observation['direction']
    
    # Find agent position in the full grid
    agent_pos = None
    for y in range(full_grid.shape[0]):
        for x in range(full_grid.shape[1]):
            if full_grid[y][x][0] == 10:  # Agent object type
                agent_pos = (x, y)
                break
        if agent_pos:
            break
    
    if agent_pos is None:
        # If we can't find the agent, use original planner as-is
        return original_planner(observation)
    
    # Create a 7x7 observation window centered on the agent
    agent_x, agent_y = agent_pos
    
    # Create 7x7 grid filled with walls (default)
    adapted_grid = np.full((7, 7, 3), [2, 5, 0], dtype=np.uint8)  # Wall with gray color
    
    # Copy visible portion of the full grid into the 7x7 window
    # Agent will be at position (3, 3) in the adapted grid
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            # Position in full grid
            full_x = agent_x + dx
            full_y = agent_y + dy
            
            # Position in adapted grid
            adapted_x = 3 + dx
            adapted_y = 3 + dy
            
            # Copy cell if it exists in the full grid
            if 0 <= full_x < full_grid.shape[1] and 0 <= full_y < full_grid.shape[0]:
                adapted_grid[adapted_y][adapted_x] = full_grid[full_y][full_x]
    
    # Create adapted observation
    adapted_obs = {
        'image': adapted_grid,
        'direction': direction,
        'mission': observation.get('mission', 'reach goal')
    }
    
    return original_planner(adapted_obs)

def test_adapter():
    """Test the adapter functionality."""
    import gymnasium as gym
    import minigrid
    from minigrid.wrappers import FullyObsWrapper
    
    print("Testing Planner Adapter")
    print("=" * 40)
    
    base_env = gym.make("MiniGrid-Empty-5x5-v0")
    env = FullyObsWrapper(base_env)
    
    obs_tuple = env.reset()
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    
    print(f"Original observation shape: {obs['image'].shape}")
    print(f"Direction: {obs['direction']}")
    
    # Find agent and goal in original
    agent_pos = None
    goal_pos = None
    for y in range(obs['image'].shape[0]):
        for x in range(obs['image'].shape[1]):
            if obs['image'][y][x][0] == 10:
                agent_pos = (x, y)
            elif obs['image'][y][x][0] == 8:
                goal_pos = (x, y)
    
    print(f"Agent at: {agent_pos}, Goal at: {goal_pos}")
    
    # Test adapter
    try:
        action = adapted_planner(obs)
        action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
        print(f"Adapter successful! Chosen action: {action} ({action_names.get(action, 'unknown')})")
        return True
    except Exception as e:
        print(f"Adapter failed: {e}")
        return False
    finally:
        env.close()

if __name__ == "__main__":
    test_adapter() 