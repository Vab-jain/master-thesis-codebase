import numpy as np
from minigrid.core.constants import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
    STATE_TO_IDX
)
# Invert STATE_TO_IDX
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

class ObservationEncoder:
    def __init__(self, max_radius: int = 10):
        self.max_radius = max_radius
        
    def _get_direction_name(self, direction):
        """Convert direction integer to readable name"""
        direction_names = {0: "east", 1: "south", 2: "west", 3: "north"}
        return direction_names.get(direction, "unknown")

    def _to_array(self, image):
        # coerce list/Box to ndarray
        arr = np.asarray(image, dtype=int)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 array, got shape {arr.shape}")
        return arr

    def encode_natural_language(self, obs):
        img = self._to_array(obs['image'])
        H, W, _ = img.shape  # Standard format: [height, width, channels]
        
        # Find agent position from image if not in obs
        agent_x, agent_y = obs.get('agent_pos', (None, None))
        if agent_x is None or agent_y is None:
            # Look for agent (type 10) in the image
            for y in range(H):  # y = row index  
                for x in range(W):  # x = column index
                    if img[y, x, 0] == 10:  # agent type - img[row, col]
                        agent_x, agent_y = x, y  # agent_x = column, agent_y = row
                        break
                if agent_x is not None:
                    break
            
            # Fallback if agent not found
            if agent_x is None or agent_y is None:
                raise ValueError("Agent position not found in observation image")
                
        direction = obs.get('direction')
        desc = []

        # Add agent direction information
        direction_name = self._get_direction_name(direction)
        desc.append(f"Agent is facing {direction_name}.")

        for y in range(H):  # y = row index
            for x in range(W):  # x = column index
                if (x, y) == (agent_x, agent_y):  # (column, row) comparison
                    continue

                obj_type, color_idx, state_idx = img[y, x]  # img[row, col]
                # skip unseen=0, empty=1, wall=2, agent=10
                if int(obj_type) in (0, 1, 2, 10):
                    continue

                color = IDX_TO_COLOR[int(color_idx)]
                obj = IDX_TO_OBJECT[int(obj_type)]
                state = IDX_TO_STATE.get(int(state_idx))
                
                # Only show state for objects where it's meaningful (doors)
                if obj == "door" and state:
                    name = f"{color} {obj} ({state})"
                else:
                    name = f"{color} {obj}"
                    
                desc.append(f"There is a {name} at position ({x},{y}).")  # (column, row) - this was correct before

        desc.append(f"Mission: {obs['mission']}")
        return " ".join(desc)

    def encode_ascii_grid(self, obs):
        img = self._to_array(obs['image'])
        H, W, _ = img.shape  # Standard format: [height, width, channels]
        
        # Find agent position from image if not in obs
        agent_x, agent_y = obs.get('agent_pos', (None, None))
        if agent_x is None or agent_y is None:
            # Look for agent (type 10) in the image
            for y in range(H):  # y = row index
                for x in range(W):  # x = column index
                    if img[y, x, 0] == 10:  # agent type - img[row, col]
                        agent_x, agent_y = x, y  # agent_x = column, agent_y = row
                        break
                if agent_x is not None:
                    break
            
            # Fallback if agent not found
            if agent_x is None or agent_y is None:
                raise ValueError("Agent position not found in observation image")
                
        direction = obs.get('direction', 0)
        direction_name = self._get_direction_name(direction)

        # Direction symbols for agent
        agent_symbols = {0: ">", 1: "v", 2: "<", 3: "^"}
        agent_symbol = agent_symbols.get(direction, "@")
        
        # First pass: Scan all objects to determine what color-object combinations exist
        object_combinations = {}  # {object_type: {color1, color2, ...}}
        all_objects = []  # [(x, y, obj_type, color_idx, state_idx), ...]
        
        for y in range(H):  # y = row index
            for x in range(W):  # x = column index
                if (x, y) == (agent_x, agent_y):  # Skip agent position
                    continue
                    
                obj_type, color_idx, state_idx = img[y, x]  # img[row, col]
                obj_type = int(obj_type)
                
                # Skip non-objects
                if obj_type in (0, 1, 2, 10):  # unseen, empty, wall, agent
                    continue
                
                obj = IDX_TO_OBJECT[obj_type]
                color = IDX_TO_COLOR[int(color_idx)]
                
                # Track object-color combinations
                if obj not in object_combinations:
                    object_combinations[obj] = set()
                object_combinations[obj].add(color)
                
                all_objects.append((x, y, obj_type, color_idx, state_idx))
        
        # Create smart symbol mapping
        symbol_map = {}
        color_symbols = {
            "red": "R", "green": "G", "blue": "B", 
            "purple": "P", "yellow": "Y", "grey": "Z"
        }
        
        # Default symbols for object types
        default_symbols = {
            "key": "K", "door": "D", "ball": "O",  # Changed ball from "B" to "O" to avoid conflict with blue
            "box": "X", "goal": "E"  # Changed goal from "G" to "E" to avoid conflict with green
        }
        
        for obj_type, colors in object_combinations.items():
            if len(colors) == 1:
                # Only one color for this object type, use default symbol
                symbol_map[(obj_type, list(colors)[0])] = default_symbols.get(obj_type, "?")
            else:
                # Multiple colors for this object type, use color-specific symbols
                for color in colors:
                    if color in color_symbols:
                        # Use color letter + first letter of object type
                        obj_initial = obj_type[0].upper()
                        symbol_map[(obj_type, color)] = color_symbols[color] + obj_initial
                    else:
                        # Fallback for unknown colors
                        symbol_map[(obj_type, color)] = default_symbols.get(obj_type, "?")
        
        # Initialize legend with basic symbols
        legend = {
            agent_symbol: f"agent facing {direction_name}",
            "?": "unseen",
            ".": "empty",
            "W": "wall"
        }
        rows = []
        
        # Keep track of object types and colors at each position for legend
        position_info = {}

        # Build the ASCII grid with transposed coordinates to match the coordinate system
        # We need to transpose so that array img[y,x] appears at visual position (x,y)
        # This means we iterate x (columns) as rows, y (rows) as columns
        for visual_row in range(W):  # visual_row corresponds to array column x
            cols = []
            for visual_col in range(H):  # visual_col corresponds to array row y
                # Get the array position: array[y=visual_col, x=visual_row]
                array_row, array_col = visual_col, visual_row
                obj_type, color_idx, state_idx = img[array_row, array_col]  # img[row, col]
                obj_type = int(obj_type)

                # Check if this is the agent position
                # Agent is at (agent_x, agent_y) which should appear at visual (agent_x, agent_y)
                if (visual_row, visual_col) == (agent_x, agent_y):  # (visual_row=agent_x, visual_col=agent_y)
                    cols.append(agent_symbol)
                    continue

                # unseen → "?"
                if obj_type == 0:
                    cols.append("?")
                    continue
                # empty → "."
                if obj_type == 1:
                    cols.append(".")
                    continue
                # wall → "W"
                if obj_type == 2:
                    cols.append("W")
                    continue
                # agent → should be handled above, but skip if found here
                if obj_type == 10:
                    cols.append(agent_symbol)
                    continue

                # other objects - use smart symbol mapping
                obj = IDX_TO_OBJECT[obj_type]
                color = IDX_TO_COLOR[int(color_idx)]
                sym = symbol_map.get((obj, color), "?")
                cols.append(sym)

                # Store position info for legend
                pos_key = (visual_row, visual_col)
                if pos_key not in position_info:
                    position_info[pos_key] = []
                
                state = IDX_TO_STATE.get(int(state_idx))
                # Only show state for objects where it's meaningful (doors)
                if obj == "door" and state:
                    desc = f"{color} {obj} ({state})"
                else:
                    desc = f"{color} {obj}"
                position_info[pos_key].append((sym, desc))

            rows.append(" ".join(cols))

        # Build legend from symbol mapping and position info
        symbol_descriptions = {}
        
        # Add symbols from the smart mapping
        for (obj_type, color), sym in symbol_map.items():
            if sym not in symbol_descriptions:
                symbol_descriptions[sym] = set()
            
            # Find if any objects of this type have states (for doors)
            has_state = False
            state_variants = set()
            for pos_info_list in position_info.values():
                for pos_sym, desc in pos_info_list:
                    if pos_sym == sym and "(" in desc and ")" in desc:
                        has_state = True
                        state_variants.add(desc)
            
            if has_state:
                # Use the actual state descriptions from position_info
                symbol_descriptions[sym].update(state_variants)
            else:
                symbol_descriptions[sym].add(f"{color} {obj_type}")
        
        # Add any additional symbols from position info (safety net)
        for pos_info_list in position_info.values():
            for sym, desc in pos_info_list:
                if sym not in symbol_descriptions:
                    symbol_descriptions[sym] = set()
                symbol_descriptions[sym].add(desc)
        
        # Create legend entries
        for sym, desc_set in symbol_descriptions.items():
            if len(desc_set) == 1:
                # Only one variant, use simple description
                legend[sym] = list(desc_set)[0]
            else:
                # Multiple variants, list them all
                legend[sym] = ", ".join(sorted(desc_set))

        grid_str = "\n".join(rows)
        legend_str = "\n".join(f"{s} - {d}" for s, d in legend.items())
        return f"Grid:\n{grid_str}\n\nLegend:\n{legend_str}\n\nMission: {obs['mission']}"

    def encode_tuple_list(self, obs):
        img = self._to_array(obs['image'])
        H, W, _ = img.shape  # Standard format: [height, width, channels]
        
        # Find agent position from image if not in obs
        agent_x, agent_y = obs.get('agent_pos', (None, None))
        if agent_x is None or agent_y is None:
            # Look for agent (type 10) in the image
            for y in range(H):  # y = row index
                for x in range(W):  # x = column index
                    if img[y, x, 0] == 10:  # agent type - img[row, col]
                        agent_x, agent_y = x, y  # agent_x = column, agent_y = row
                        break
                if agent_x is not None:
                    break
            
            # Fallback if agent not found
            if agent_x is None or agent_y is None:
                raise ValueError("Agent position not found in observation image")
                
        direction = obs.get('direction', 0)
        direction_name = self._get_direction_name(direction)
        items = []

        for y in range(H):  # y = row index
            for x in range(W):  # x = column index
                if (x, y) == (agent_x, agent_y):  # (column, row) comparison
                    continue

                obj_type, color_idx, state_idx = img[y, x]  # img[row, col]
                # skip unseen=0, empty=1, wall=2, agent=10
                if int(obj_type) in (0, 1, 2, 10):
                    continue

                color = IDX_TO_COLOR[int(color_idx)]
                obj = IDX_TO_OBJECT[int(obj_type)]
                state = IDX_TO_STATE.get(int(state_idx))
                
                # Only show state for objects where it's meaningful (doors)
                if obj == "door" and state:
                    state_str = f", state={state}"
                else:
                    state_str = ""
                items.append(f"({repr(color)} {obj}{state_str}, ({x}, {y}))")  # (column, row) - this was correct before

        agent_info = f"Agent at ({agent_x}, {agent_y}) facing {direction_name}. "
        return agent_info + f"Objects: [{', '.join(items)}]. Mission: {obs['mission']}"

    def encode_relative_description(self, obs):
        img = self._to_array(obs['image'])
        H, W, _ = img.shape  # Standard format: [height, width, channels]
        
        # Find agent position from image if not in obs
        agent_x, agent_y = obs.get('agent_pos', (None, None))
        if agent_x is None or agent_y is None:
            # Look for agent (type 10) in the image
            for y in range(H):  # y = row index
                for x in range(W):  # x = column index
                    if img[y, x, 0] == 10:  # agent type - img[row, col]
                        agent_x, agent_y = x, y  # agent_x = column, agent_y = row
                        break
                if agent_x is not None:
                    break
            
            # Fallback if agent not found
            if agent_x is None or agent_y is None:
                raise ValueError("Agent position not found in observation image")
                
        direction = obs.get('direction', 0)
        direction_name = self._get_direction_name(direction)
        rel_descs = []
        
        # Add agent direction information
        rel_descs.append(f"Agent is facing {direction_name}.")

        for y in range(H):  # y = row index
            for x in range(W):  # x = column index
                if (x, y) == (agent_x, agent_y):  # (column, row) comparison
                    continue

                obj_type, color_idx, state_idx = img[y, x]  # img[row, col]
                # skip unseen=0, empty=1, wall=2, agent=10
                if int(obj_type) in (0, 1, 2, 10):
                    continue

                # Calculate relative position from agent's perspective
                # Note: in our coordinate system (x,y) = (row, col)
                dx, dy = x - agent_x, y - agent_y  # dx = row_diff, dy = col_diff
                
                # Transform coordinates to agent's reference frame
                # MiniGrid directions: 0=east, 1=south, 2=west, 3=north
                # For each direction, we need to map world coordinates to agent's frame:
                # - rel_y > 0: ahead of agent
                # - rel_y < 0: behind agent  
                # - rel_x > 0: to agent's right
                # - rel_x < 0: to agent's left
                
                if direction == 0:  # facing east (right)
                    # ahead = increasing column (positive dy)
                    # behind = decreasing column (negative dy)
                    # left = decreasing row (negative dx) 
                    # right = increasing row (positive dx)
                    rel_x, rel_y = dx, dy
                elif direction == 1:  # facing south (down)
                    # ahead = increasing row (positive dx)
                    # behind = decreasing row (negative dx)
                    # left = increasing column (positive dy)
                    # right = decreasing column (negative dy) 
                    rel_x, rel_y = dy, dx
                elif direction == 2:  # facing west (left)
                    # ahead = decreasing column (negative dy)
                    # behind = increasing column (positive dy)
                    # left = increasing row (positive dx)
                    # right = decreasing row (negative dx)
                    rel_x, rel_y = -dx, -dy
                elif direction == 3:  # facing north (up)
                    # ahead = decreasing row (negative dx)
                    # behind = increasing row (positive dx)
                    # left = decreasing column (negative dy)
                    # right = increasing column (positive dy)
                    rel_x, rel_y = -dy, -dx
                else:
                    rel_x, rel_y = dx, dy  # fallback

                dist = abs(rel_x) + abs(rel_y)
                if dist > self.max_radius:
                    continue

                # Determine relative position description
                if rel_y < 0:
                    pos = "behind"
                elif rel_y > 0:
                    pos = "ahead"
                elif rel_x < 0:
                    pos = "to your left"
                elif rel_x > 0:
                    pos = "to your right"
                else:
                    pos = "at your position"  # shouldn't happen as we skip agent position

                color = IDX_TO_COLOR[int(color_idx)]
                obj = IDX_TO_OBJECT[int(obj_type)]
                state = IDX_TO_STATE.get(int(state_idx))
                
                # Only show state for objects where it's meaningful (doors)
                if obj == "door" and state:
                    state_str = f" ({state})"
                else:
                    state_str = ""
                rel_descs.append(
                    f"There is a {color} {obj}{state_str} {dist} tile(s) {pos}."
                )

        rel_descs.append(f"Mission: {obs['mission']}")
        return " ".join(rel_descs)

    def encode_all(self, obs):
        return {
            "natural":  self.encode_natural_language(obs),
            "ascii":    self.encode_ascii_grid(obs),
            "tuples":   self.encode_tuple_list(obs),
            "relative": self.encode_relative_description(obs),
        }

    def debug_coordinate_system(self, obs):
        """Debug method to verify coordinate system is correct"""
        img = self._to_array(obs['image'])
        H, W, _ = img.shape
        
        print(f"Image shape: H={H} (rows), W={W} (columns)")
        
        # Find agent position
        agent_pos = obs.get('agent_pos', (None, None))
        if agent_pos[0] is None:
            for y in range(H):  # y = row index
                for x in range(W):  # x = column index
                    if img[y, x, 0] == 10:  # agent type
                        agent_pos = (y, x)  # agent_pos = (row, col)
                        break
                if agent_pos[0] is not None:
                    break
        
        print(f"Agent position: ({agent_pos[0]}, {agent_pos[1]}) -> row {agent_pos[0]}, col {agent_pos[1]}")
        print(f"Agent in image array at: img[{agent_pos[0]}, {agent_pos[1]}] = {img[agent_pos[0], agent_pos[1]] if agent_pos[0] is not None else 'NOT FOUND'}")
        
        # Show a small grid around agent for verification
        if agent_pos[0] is not None and agent_pos[1] is not None:
            print("\nSmall grid around agent (showing obj types):")
            for dy in range(-1, 2):
                row_str = ""
                for dx in range(-1, 2):
                    y_pos, x_pos = agent_pos[0] + dy, agent_pos[1] + dx
                    if 0 <= y_pos < H and 0 <= x_pos < W:
                        obj_type = img[y_pos, x_pos, 0]  # img[row, col]
                        if dy == 0 and dx == 0:
                            row_str += f"[{obj_type:2d}]"  # agent position in brackets
                        else:
                            row_str += f" {obj_type:2d} "
                    else:
                        row_str += " -- "
                print(f"  {row_str}")
        
        return "Debug info printed to console"
    
