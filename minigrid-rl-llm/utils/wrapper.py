import gymnasium as gym
from gymnasium.core import ObservationWrapper, Wrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
import numpy as np
import dspy
import re

import utils


class TextDesciptionWrapper(ObservationWrapper):
     '''
     Wrapper to add image description of the current state into the observation. If ´loc=False´, then location of the objects is ommited and only a list of object is presented.
     Example:
         Current State: 
         Wall at [(0, 0), (0, 1), (0, 2),  (0, 3), (0, 4), (0, 5), (1, 0), (1, 5), (2, 0), (2, 5), (3, 0), (3, 5), (4, 0), (4, 5), (5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (6, 0), (6, 5), (7, 0), (7, 5), (8, 0), (8, 5), (9, 0), (9, 5), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5)]
         Agent at [(1, 2)]
         Blue Key at [(2, 1)]
         Purple Ball at [(4, 1)]
         Locked Blue Door at [(5, 1)]
         Blue Box at [(6, 3)]
     
     '''
     def __init__(self, env, loc=True):
         super().__init__(env)

         self.loc = loc
         
         new_text_space = gym.spaces.Text(max_length=2048)
         
         self.observation_space = gym.spaces.Dict(
             {**self.observation_space.spaces, "text_desc": new_text_space}
         )
         
         
     def observation(self, obs):
         # reverse mappings for easy lookups
         idx_to_object = {v: k for k, v in OBJECT_TO_IDX.items()}
         idx_to_color = {v: k for k, v in COLOR_TO_IDX.items()}
         idx_to_state = {v: k for k, v in STATE_TO_IDX.items()}
         
         # ignore some objects e.g. Wall, Lava, etc. (note: capitalize the first word)
         # ignore_objects = ["Wall", "Lava", "Empty"] 
         ignore_objects = ["Empty"] 
         
         # dict to store result
         object_locations = {}
         
         img = obs['image']
         
         for i in range(len(img)):
             for j in range(len(img[0])):
                 object_idx, color_idx, state_idx = img[i,j]
                 object_name = idx_to_object.get(object_idx, "unknown")
                 color_name = idx_to_color.get(color_idx, "unknown")
                 state_name = idx_to_state.get(state_idx, "unknown")
                 
                 if object_name in {"key", "ball", "box"}:
                     full_name = f'{color_name.capitalize()} {object_name.capitalize()}'
                 elif object_name == "door":
                     full_name = f'{state_name.capitalize()} {color_name.capitalize()} {object_name.capitalize()}'
                 else:
                     full_name = object_name.capitalize()
                 
                 # only add an object if it in not in ignore_locations
                 if full_name not in ignore_objects:
                     if full_name not in object_locations:
                         object_locations[full_name] = []
                     object_locations[full_name].append((i,j))
         
         # format the output as a string
         output_lines = []
         for obj, locations in object_locations.items():
            locations_str = ", ".join(str(loc) for loc in locations)
            if self.loc:
                 output_lines.append(f"{obj} at [{locations_str}]")
            else:
                output_lines.append(f"{obj}")
         
         final_text_desc = "\n".join(output_lines)
         
         return {**obs, "text_desc": final_text_desc}
 


class RoomAbstractionTextWrapper(ObservationWrapper):
    '''
    Wrapper to add room-based abstraction of the current state into the observation.
    Example:
        Current State (Room abstraction): 
        Agent in Room 2
        Blue Key in Room 3
        Locked Red Door between Room 1 and Room 2
        Total number of rooms: 3
    '''
    def __init__(self, env):
        super().__init__(env)
        
        new_text_space = gym.spaces.Text(max_length=2048)
        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, "text_desc": new_text_space}
        )
        self.rooms = None  # To store room assignments
        self.door_connections = []  # To store doors connecting rooms
    
    def assign_rooms(self, walls, doors, grid_width, grid_height):
        """
        Assign a room number to each cell in the grid based on wall and door positions.
        Treat doors as boundaries between rooms.
        """
        room_grid = np.full((grid_height, grid_width), -1, dtype=int)
        current_room = 1  # Start numbering rooms from 1
        
        def flood_fill(x, y):
            """Flood fill to assign room numbers."""
            if (
                x < 0 or x >= grid_width or
                y < 0 or y >= grid_height or
                room_grid[y, x] != -1 or
                (x, y) in walls or
                (x, y) in doors  # Treat doors as boundaries
            ):
                return
            room_grid[y, x] = current_room
            # Check all neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                flood_fill(x + dx, y + dy)
        
        for y in range(grid_height):
            for x in range(grid_width):
                if room_grid[y, x] == -1 and (x, y) not in walls and (x, y) not in doors:
                    flood_fill(x, y)
                    current_room += 1
        
        return room_grid, current_room - 1  # Subtract 1 to get the actual room count
    
    def get_room_number(self, x, y):
        """
        Retrieve the room number for a given coordinate (x, y).
        """
        return self.rooms[y, x] if self.rooms is not None else -1
    
    def find_door_connections(self, doors):
        """
        Find room connections for each door. A door connects two distinct rooms.
        """
        connections = []
        for door_x, door_y in doors:
            adjacent_rooms = set()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor_x, neighbor_y = door_x + dx, door_y + dy
                if 0 <= neighbor_x < self.rooms.shape[1] and 0 <= neighbor_y < self.rooms.shape[0]:
                    room_number = self.get_room_number(neighbor_x, neighbor_y)
                    if room_number != -1:
                        adjacent_rooms.add(room_number)
            if len(adjacent_rooms) == 2:
                connections.append((door_x, door_y, tuple(sorted(adjacent_rooms))))
        return connections
    
    def observation(self, obs):
        # reverse mappings for easy lookups
        idx_to_object = {v: k for k, v in OBJECT_TO_IDX.items()}
        idx_to_color = {v: k for k, v in COLOR_TO_IDX.items()}
        idx_to_state = {v: k for k, v in STATE_TO_IDX.items()}
        
        # ignore some objects e.g., Empty (note: capitalize the first word)
        ignore_objects = ["Empty", "Wall", "Lava"]
        
        img = obs['image']
        grid_height, grid_width = len(img), len(img[0])
        
        # Extract walls and doors
        walls = set()
        doors = set()
        door_details = {}
        for y in range(grid_height):
            for x in range(grid_width):
                object_idx, color_idx, state_idx = img[y, x]
                object_name = idx_to_object.get(object_idx, "unknown")
                color_name = idx_to_color.get(color_idx, "unknown")
                state_name = idx_to_state.get(state_idx, "unknown")
                
                if object_name == "wall":
                    walls.add((x, y))
                elif object_name == "door":
                    doors.add((x, y))
                    full_name = f"{state_name.capitalize()} {color_name.capitalize()} Door"
                    door_details[(x, y)] = full_name
        
        # Assign rooms and find door connections if not already done
        if self.rooms is None:
            self.rooms, num_rooms = self.assign_rooms(walls, doors, grid_width, grid_height)
            self.door_connections = self.find_door_connections(doors)
        
        # Collect object locations
        object_locations = {}
        for y in range(grid_height):
            for x in range(grid_width):
                object_idx, color_idx, state_idx = img[y, x]
                object_name = idx_to_object.get(object_idx, "unknown")
                color_name = idx_to_color.get(color_idx, "unknown")
                state_name = idx_to_state.get(state_idx, "unknown")
                
                if object_name in {"key", "ball", "box"}:
                    full_name = f'{color_name.capitalize()} {object_name.capitalize()}'
                elif object_name == "door":
                    continue  # Doors are handled separately
                else:
                    full_name = object_name.capitalize()
                
                # Only add an object if it's not in ignore_objects
                if full_name not in ignore_objects:
                    room_number = self.get_room_number(x, y)
                    if full_name not in object_locations:
                        object_locations[full_name] = set()
                    object_locations[full_name].add(room_number)     
        
        # Format the output as a string
        output_lines = []
        
        # Add total number of rooms
        total_rooms = len(set(self.rooms.flatten())) - (1 if -1 in self.rooms else 0)
        output_lines.append(f"Total number of rooms: {total_rooms}")
        
        for obj, rooms in object_locations.items():
            rooms_str = ", ".join(f"Room {room}" for room in sorted(rooms))
            output_lines.append(f"{obj} in {rooms_str}")
        
        # Add door connections
        for door_x, door_y, (room1, room2) in sorted(self.door_connections):
            door_name = door_details.get((door_x, door_y), "Door")
            output_lines.append(f"{door_name} between Room {room1} and Room {room2}")
        
        final_text_desc = "\n".join(output_lines)
        
        return {**obs, "text_desc": final_text_desc}


def configure_llm(llm_model_id='llama3-70b-8192', cache_llm_dspy=False, GROQ=True):
    if GROQ:
        lm = dspy.LM(llm_model_id, api_base='https://api.groq.com/openai/v1', api_key=utils.GROQ_API_KEY, cache=cache_llm_dspy)
    else:
        lm = dspy.LM(llm_model_id, api_base='http://localhost:11434', cache=cache_llm_dspy)
    dspy.configure(lm=lm)


class LLMSuggestedMissionWrapper(ObservationWrapper):
    def __init__(self, env,
                 llm_use_prob=1,
                 llm_model_id='llama3-70b-8192',
                 GROQ=True,
                 cache_llm_dspy=False,
                 ):
        super().__init__(env)
        new_text_space = gym.spaces.Text(max_length=2048)
        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, "llm_hint": new_text_space}
        )

        # configure dspy with LLM
        configure_llm(llm_model_id, cache_llm_dspy, GROQ)
        self.llm_agent = utils.RAGAgent()
        
        # initializing config variables
        self.task_desc = utils.task_desc

        self.llm_use_prob = llm_use_prob

        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["llm_hint"] = self._get_llm_suggesstion(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed = None):
        obs, info = self.env.reset()
        obs["llm_hint"] = self._get_llm_suggesstion(obs)
        return obs,info
    
    def _get_llm_suggesstion(self,obs):
        # TODO: imeplement this function
        response = self.llm_agent(task_desc=self.task_desc, current_state=obs["text_desc"], mission=obs["mission"])
        return response.subgoal
