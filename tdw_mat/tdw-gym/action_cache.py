class ActionCache:
    """
    Cache for storing and retrieving available action plans.
    Contains its own get_available_plans method for generating action plans.
    """
    
    # Action weights based on plan type selection ratios (selected_count / appeared_count)
    # Initial values from episodes 1-4 plan_sequence analysis
    _WEIGHT_SELECTED_INIT = {
        'explore': 25,
        'gograsp_container': 12,
        'gograsp_target': 48,
        'goto': 35,
        'putin': 18,
        'sendmessage': 13,
        'transport': 20,
    }
    _WEIGHT_APPEARED_INIT = {
        'explore': 92,
        'gograsp_container': 47,
        'gograsp_target': 71,
        'goto': 171,
        'putin': 24,
        'sendmessage': 158,
        'transport': 88,
    }
    ACTION_WEIGHTS = {
        'explore': round(25 / 92, 3),
        'gograsp_container': round(12 / 47, 3),
        'gograsp_target': round(48 / 71, 3),
        'goto': round(35 / 171, 3),
        'putin': round(18 / 24, 3),
        'sendmessage': round(13 / 158, 3),
        'transport': round(20 / 88, 3),
    }
    
    def __init__(self, rooms_name=None, goal_objects=None, debug_writer=None, agent_id=None, no_cache_loading=False):
        """
        Initialize ActionCache.
        
        Args:
            no_cache_loading: If True, skip loading 2gram file and build cache at runtime
        """
        # Set debug_writer first before any other initialization
        self.debug_writer = debug_writer
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 1 else "he"
        self.cache = {}
        self._weight_selected = dict(self._WEIGHT_SELECTED_INIT)
        self._weight_appeared = dict(self._WEIGHT_APPEARED_INIT)
        self.ACTION_WEIGHTS = dict(ActionCache.ACTION_WEIGHTS)  # instance-level copy for dynamic updates
        self.last_message = None
        self.last_plans = None
        self.last_num = 0
        self.last_plans_list = []
        
        # Action-specific cache structure
        self.action_transitions = {}
        self.action_2gram_data = None
        self.no_cache_loading = no_cache_loading
        if not no_cache_loading:
            self._load_action_2gram_data()
        else:
            self._debug_print("Cache loading disabled: will build cache at runtime")
        self.single = False
        # State variables needed for plan generation
        self.communication = True
        self.current_room = None
        self.rooms_explored = {}
        self.holding_objects = [{'type': None}, {'type': None}]
        self.object_list = [[], [], []]  # [target_objects, containers, goal_objects]
        self.rooms = rooms_name
        self.goal_objects = goal_objects
    
    def _recompute_action_weights(self):
        """Recompute ACTION_WEIGHTS from current selected/appeared counts."""
        for action_type in self._weight_appeared:
            if self._weight_appeared[action_type] > 0:
                self.ACTION_WEIGHTS[action_type] = round(
                    self._weight_selected.get(action_type, 0) / self._weight_appeared[action_type], 3
                )
    
    def update_weight_appeared(self, available_types):
        """
        Update appeared counts for action types seen in available options.
        Each unique action type counts once per step.
        
        Args:
            available_types: List of action type strings (may contain duplicates)
        """
        for action_type in set(available_types):
            normalized = self._action_to_cache_entry_type(action_type)
            self._weight_appeared[normalized] = self._weight_appeared.get(normalized, 0) + 1
    
    def update_weight_selected(self, selected_type):
        """
        Update selected count for the action type that was actually chosen.
        Call this from hit/miss to record what was actually selected.
        
        Args:
            selected_type: The action type that was selected (raw or cache-entry format)
        """
        normalized = self._action_to_cache_entry_type(selected_type)
        self._weight_selected[normalized] = self._weight_selected.get(normalized, 0) + 1
        self._recompute_action_weights()
    
    def _action_to_cache_entry_type(self, action_type):
        """Normalize an action type string to cache-entry format."""
        type_map = {
            'explore': 'explore',
            'go to': 'goto',
            'goto': 'goto',
            'go grasp target': 'gograsp_target',
            'gograsp_target': 'gograsp_target',
            'go grasp container': 'gograsp_container',
            'gograsp_container': 'gograsp_container',
            'put into container': 'putin',
            'putin': 'putin',
            'transport': 'transport',
            'send message': 'sendmessage',
            'sendmessage': 'sendmessage',
        }
        return type_map.get(action_type, action_type)
    
    def _debug_print(self, message):
        """Print debug message using the provided debug_writer if available"""
        if self.debug_writer:
            self.debug_writer(f"ActionCache: {message}")
        # If debug_writer is None, don't print anything
    
    def _create_prioritized_room_list(self, none_room, part_room, oppo_room):
        """
        Create prioritized room list based on agent parity preference.
        Priority order:
        1. 홀짝이 맞는 none_room
        2. 홀짝이 맞는 part_room  
        3. 안맞는 none_room
        4. 안맞는 part_room
        5. oppo_room
        """
        import re
        import random
        
        def get_room_number(room):
            """Extract room number from room name"""
            # Handle format like "<Livingroom> (1000)" or "room_1" or just "1"
            room_match = re.search(r'\((\d+)\)|room_(\d+)|(\d+)', room)
            if room_match:
                room_id = int(room_match.group(1) or room_match.group(2) or room_match.group(3))
                # Divide by 1000 to get the room number for parity calculation
                return room_id // 1000
            return None
        
        def separate_by_parity(room_list):
            """Separate rooms by parity matching with agent_id"""
            matching_parity = []
            different_parity = []
            
            for room in room_list:
                room_num = get_room_number(room)
                if room_num is not None:
                    agent_parity = self.agent_id % 2
                    room_parity = room_num % 2
                    if agent_parity == room_parity:
                        matching_parity.append(room)
                    else:
                        different_parity.append(room)
                else:
                    # If can't parse room number, put in different_parity (lower priority)
                    different_parity.append(room)
            
            return matching_parity, different_parity
        
        # Separate each room type by parity
        none_matching, none_different = separate_by_parity(none_room)
        part_matching, part_different = separate_by_parity(part_room)
        
        # Shuffle each category separately
        random.shuffle(none_matching)
        random.shuffle(none_different)
        random.shuffle(part_matching)
        random.shuffle(part_different)
        random.shuffle(oppo_room)
        
        # Combine in priority order
        shuffled_room = none_matching + part_matching + none_different + part_different + oppo_room
        
        # Debug logging
        # self._debug_print(f"ActionCache agent_{self.agent_id}: room priority order: "
        #                   f"none_matching={none_matching}, part_matching={part_matching}, "
        #                   f"none_different={none_different}, part_different={part_different}, "
        #                   f"oppo_room={oppo_room}")
        
        return shuffled_room
    
    def _action_to_cache_entry(self, action_string):
        """
        Convert action string to normalized cache entry format based on action_2gram_grouped_by_start.txt.
        
        Args:
            action_string: Raw action string from action history
            
        Returns:
            str: Normalized cache entry key matching the 2-gram data
        """
        if not action_string:
            return "START"
            
        # Remove step information and status suffixes
        action = action_string.split(" at step ")[0]
        
        # Remove status suffixes like " - canceled", " - unreached"
        for suffix in [" - canceled", " - unreached"]:
            if action.endswith(suffix):
                action = action[:-len(suffix)]
                break
        
        # Normalize different action types based on the 2-gram data
        if action.startswith("go to "):
            return "goto"
        elif action.startswith("explore "):
            return "explore"
        elif action.startswith("go grasp target object"):
            return "gograsp_target"
        elif action.startswith("go grasp container"):
            return "gograsp_container"
        elif action.startswith("transport "):
            return "transport"
        elif action.startswith("put "):
            return "putin"
        elif action.startswith("send a message"):
            return "sendmessage"
        elif action == "[wait]":
            return "wait"
        else:
            # Handle any unmatched actions gracefully
            self._debug_print(f"Warning3: Unmatched action type: {action}")
            return "unknown"
    
    def get_available_plans(self, message, opponent_last_room=None, current_step=0, successful_count=0):
        """
        Generate available action plans based on current state.
        
        Args:
            message: The message input for plan generation
            opponent_last_room: Last room where opponent was seen (optional)
            current_step: Current frame number (optional)
            successful_count: Number of successful objects collected (optional)
            
        Returns:
            tuple: (available_plans, num, available_plans_list)
        """
        available_plans = []
        
        if self.communication and message is not None:
            available_plans.append(f"send a message: {message}")
            
        if self.holding_objects[0]['type'] is None or self.holding_objects[1]['type'] is None:
            # Prioritize objects in current room first, then others (maintaining recent discovery order)
            current_room_obj_ids = set()
            if self.current_room in self.obj_per_room:
                current_room_obj_ids = set(obj['id'] for obj in self.obj_per_room[self.current_room][0])
                current_room_obj_ids.update(obj['id'] for obj in self.obj_per_room[self.current_room][1])

            # Add target objects: current room first, then others
            for obj in self.object_list[0]:
                if obj['id'] in current_room_obj_ids:
                    available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")
            for obj in self.object_list[0]:
                if obj['id'] not in current_room_obj_ids:
                    available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")

            # Add containers: current room first, then others (only if successful < 6)
            if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1) and successful_count < 5 and current_step < 1700:
                for obj in self.object_list[1]:
                    if obj['id'] in current_room_obj_ids:
                        available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
                for obj in self.object_list[1]:
                    if obj['id'] not in current_room_obj_ids:
                        available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
        else:
            # Only allow put action if there are still target objects to collect
            if self.object_list[0] and current_step < 2300:
                if self.holding_objects[0]['type'] == 1 and self.holding_objects[0]['contained'][-1] is None and self.holding_objects[1]['type'] == 0:
                    available_plans.append(f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})")
                elif self.holding_objects[1]['type'] == 1 and self.holding_objects[1]['contained'][-1] is None and self.holding_objects[0]['type'] == 0:
                    available_plans.append(f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})")
                
        # Only allow transport if holding target objects (type 0) or containers with target objects
        has_target_objects = False
        holding_container = False
        total_target_objects = 0  # Total count of target objects held (including those in container)
        
        for obj in self.holding_objects:
            if obj['type'] == 0:  # Target object
                has_target_objects = True
                total_target_objects += 1
            elif obj['type'] == 1 and 'contained' in obj:  # Container with objects
                holding_container = True
                container_objects = len([o for o in obj['contained'] if o is not None])
                total_target_objects += container_objects
                if container_objects > 0:
                    has_target_objects = True
        
        # Transport conditions:
        # 1. Not holding container OR frame > 1000: transport anytime if has target objects
        # 2. Holding container AND frame <= 1000: need at least 3 total target objects
        can_transport = False
        if has_target_objects and len(self.object_list[2]) != 0:
            if not holding_container or current_step > 1000:
                can_transport = True
            elif holding_container and current_step <= 1000 and total_target_objects ==4:
                can_transport = True
        
        if can_transport:
            available_plans.append(f"transport objects I'm holding to the bed")
        # self._debug_print(f"rooms_explored: {self.rooms_explored}")
        none_room = []
        part_room = []
        oppo_room = []
        for room in self.rooms:
            # Skip current room, None values, and fully explored rooms (marked as 'all')
            # self._debug_print(f"room: {room}")
            if room == self.current_room or room is None or room == 'None' or (room in self.rooms_explored and self.rooms_explored[room] == 'all'):
                continue
            available_plans.append(f"go to {room}")
            if room == opponent_last_room:
                oppo_room.append(room)
            if(room in self.rooms_explored and self.rooms_explored[room] == 'part'):
                part_room.append(room)
            else:
                none_room.append(room)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            available_plans.append(f"explore current room {self.current_room}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"
        
        # Create shuffled_room with priority order based on agent parity
        shuffled_room = self._create_prioritized_room_list(none_room, part_room, oppo_room)
        return plans, len(available_plans), available_plans, shuffled_room
    
    def _create_cache_key(self, message):
        """
        Create a cache key based on current LLM state and message.
        
        Args:
            message: The message input
            
        Returns:
            str: Cache key representing current state
        """
        # Create key based on relevant LLM state
        key_components = [
            str(message),
            str(getattr(self.llm, 'current_room', None)),
            str(getattr(self.llm, 'holding_objects', None)),
            str(getattr(self.llm, 'rooms_explored', None)),
            str(getattr(self.llm, 'object_list', None))
        ]
        
        return "|".join(key_components)
    
    def clear_cache(self):
        """Clear all cached plans."""
        self.cache.clear()
        self.last_message = None
        self.last_plans = None
        self.last_num = 0
        self.last_plans_list = []
    
    def invalidate_cache_for_message(self, message):
        """
        Invalidate cache entries for a specific message.
        
        Args:
            message: The message to invalidate cache for
        """
        keys_to_remove = [key for key in self.cache.keys() if key.startswith(str(message))]
        for key in keys_to_remove:
            del self.cache[key]
    
    def get_total_cache_lines(self):
        """
        Get total number of cache lines (transitions) stored.
        
        Returns:
            int: Total number of transitions across all actions
        """
        total = 0
        for action, transitions in self.action_transitions.items():
            total += len(transitions)
        return total
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics including size and hit rate
        """
        return {
            'cache_size': len(self.cache),
            'last_num_plans': self.last_num,
            'has_cached_plans': self.last_plans is not None,
            'total_cache_lines': self.get_total_cache_lines()
        }

    def progress2text(self, current_step, satisfied, opponent_grabbed_objects, opponent_last_room,): #important
        lines = []
        lines.append(f"Progress: I've taken {current_step}/3000 steps.")
        
        # Transported objects
        if len(satisfied) > 0:
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            transported = [x for x in unique_satisfied if x['type'] == 0]
            if len(transported) > 0:
                obj_str = ', '.join([f"<{x['name']}> ({x['id']})" for x in transported[:-1]])
                if len(transported) > 1:
                    obj_str += f", and <{transported[-1]['name']}> ({transported[-1]['id']})"
                else:
                    obj_str = f"<{transported[0]['name']}> ({transported[0]['id']})"
                lines.append(f"{'I' if self.single else 'We'}'ve already transported {obj_str} to the bed.")
        elif len(self.object_list[2]) > 0:
            lines.append("No objects have been transported to the bed yet.")
        
        # Currently holding
        hold_parts = []
        for obj in self.holding_objects:
            if obj['type'] == 0:
                hold_parts.append(f"a target object <{obj['name']}> ({obj['id']})")
            elif obj['type'] == 1:
                contained = []
                for j, o in enumerate(obj['contained']):
                    if o is None:
                        break
                    contained.append(f"<{obj['contained_name'][j]}> ({o})")
                if len(contained) == 0:
                    hold_parts.append(f"an empty container <{obj['name']}> ({obj['id']})")
                else:
                    cont_str = ', and '.join([', '.join(contained[:-1]), contained[-1]]) if len(contained) > 1 else contained[0]
                    hold_parts.append(f"a container <{obj['name']}> ({obj['id']}) that contains {cont_str}")
        
        if len(hold_parts) == 0:
            lines.append("I'm currently holding nothing.")
        elif len(hold_parts) == 1:
            lines.append(f"I'm currently holding {hold_parts[0]}.")
        else:
            lines.append(f"I'm currently holding {hold_parts[0]} and {hold_parts[1]}.")
        
        # Current location
        current_room_objs = self.obj_per_room.get(self.current_room, [[], [], []])
        explored_status = self.rooms_explored.get(self.current_room, 'none')
        if explored_status == 'all':
            explored_str = "fully explored"
        elif explored_status == 'none':
            explored_str = "not explored yet"
        else:
            explored_str = f"explored {explored_status} of the area"
        lines.append(f"I'm located in the {self.current_room}, where I've {explored_str}.")
        
        # Exploration summary
        lines.append(f"\n{self.agent_name}'s Exploration summary:")
        for room in self.rooms:
            if room == self.current_room:
                continue
            obj_list = self.obj_per_room.get(room, [[], [], []])
            explored = self.rooms_explored.get(room, 'none')
            
            # Format exploration status
            if explored == 'all':
                exp_str = "Fully explored"
            elif explored == 'none':
                exp_str = "Not explored yet"
            else:
                exp_str = f"Partially explored ({explored})"
            
            # Format found items
            items = []
            for obj in obj_list[0]:  # targets
                items.append(f"a target object <{obj['name']}> ({obj['id']})")
            for obj in obj_list[1]:  # containers
                items.append(f"a container <{obj['name']}> ({obj['id']})")
            if len(obj_list[2]) > 0:  # bed
                items.append("the bed (goal position)")
            
            if len(items) == 0:
                items_str = "found nothing"
            else:
                items_str = "found " + ', '.join(items[:-1]) + (f", and {items[-1]}" if len(items) > 1 else items[0])
            
            lines.append(f"- {room}: {exp_str}; {items_str}.")
        
        return '\n'.join(lines)

    def access(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room, action_history, dialogue_history, opponent_grabbed_objects = None, opponent_last_room = None):
        """
        Cache version of the run method that updates state and gets available plans.
        
        Args:
            current_step: Current step number
            current_room: Current room name
            rooms_explored: Dictionary of explored rooms
            holding_objects: List of objects being held
            satisfied: Satisfaction status
            object_list: List of available objects
            obj_per_room: Objects per room mapping
            action_history: History of actions taken
            dialogue_history: History of dialogue
            opponent_grabbed_objects: Objects grabbed by opponent (optional)
            opponent_last_room: Last room of opponent (optional)
            
        Returns:
            tuple: (plan, info) where plan is None if no valid plans, info contains metadata
        """
        info = {}
        # self._debug_print(f"current_step: {current_step}")
        
        # Update internal state
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        # For now, set message to None (communication logic would go here)
        message = None
        # self._debug_print(f"progress_desc: {self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)}")
        if self.communication and not action_history[-1].startswith('send a message'):
            progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
            message = progress_desc + "\n"
        
        # Get available plans
        # Count only target objects (type 0) in satisfied
        successful_count = sum(1 for obj in satisfied if isinstance(obj, dict) and obj.get('type') == 0)
        available_plans, num, available_plans_list, shuffled_room = self.get_available_plans(message, opponent_last_room, current_step, successful_count)
        
        # Update dynamic ACTION_WEIGHTS: record appeared types from available plans
        self.update_weight_appeared([self._action_to_cache_entry(p) for p in available_plans_list])
        
        # If there's a put action, always prioritize it
        put_actions = [plan for plan in available_plans_list if plan.startswith('put ')]
        if put_actions:
            return put_actions[0], info, False
        
        last_action_raw = action_history[-1]
        last_action = self._action_to_cache_entry(last_action_raw)
        
        frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count = self._calculate_cache_metrics(current_step, holding_objects, rooms_explored, satisfied)
        #TODO real last action
        # TODO3:cache access하고 action selection
        possible_actions, all_actions = self.get_action_cache_lookup(last_action, current_step, current_room, rooms_explored, holding_objects, satisfied)
        
        # self._debug_print(f"Available Plans: {available_plans}, {available_plans_list}")
        # self._debug_print(f"Cache Access: {possible_actions}, {all_actions}")
        # Match possible_actions with available_plans_list to select a plan
        if num > 0:
            import random
            
            # Group possible_actions by count (priority) and randomize within each group
            def randomize_same_priority_actions(actions_with_counts):
                if not actions_with_counts:
                    return []
                
                # Group by count
                priority_groups = {}
                for action, count in actions_with_counts:
                    if count not in priority_groups:
                        priority_groups[count] = []
                    priority_groups[count].append(action)
                
                # Sort by count (descending) and randomize within each group
                randomized_actions = []
                for count in sorted(priority_groups.keys(), reverse=True):
                    group_actions = priority_groups[count].copy()
                    random.shuffle(group_actions)  # Randomize within same priority
                    randomized_actions.extend(group_actions)
                
                return randomized_actions
            
            # Convert possible_actions with randomization for same priorities
            possible_action_types = randomize_same_priority_actions(possible_actions)
            
            # First try to match with possible_actions
            for action_type in possible_action_types:
                # Map action types to plan prefixes
                if action_type == 'explore':
                    prefix = 'explore'
                elif action_type == 'gograsp_container':
                    prefix = 'go grasp container'
                elif action_type == 'gograsp_target':
                    prefix = 'go grasp target'
                elif action_type == 'goto':
                    prefix = 'go to'
                elif action_type == 'putin':
                    prefix = 'put'
                elif action_type == 'sendmessage':
                    prefix = 'send a message'
                elif action_type == 'transport':
                    prefix = 'transport'
                else:
                    continue
                
                # Find all matching plans in available_plans_list
                matching_plans = [plan for plan in available_plans_list if plan.startswith(prefix)]
                if matching_plans:
                    # Use first element (most recently discovered) for grasp actions, random for others
                    if action_type in ['gograsp_target', 'gograsp_container']:
                        selected_plan = matching_plans[0]  # Select first (most recently discovered)
                    elif action_type == 'goto' and shuffled_room:
                        # Use prioritized room selection for goto actions
                        selected_plan = f"go to {shuffled_room[0]}"
                    else:
                        selected_plan = random.choice(matching_plans)  # Random selection for other actions
                    return selected_plan, info, False
            
            # If no match found in possible_actions, try all_actions as fallback
            all_action_types = randomize_same_priority_actions(all_actions)
            for action_type in all_action_types:
                # Map action types to plan prefixes
                if action_type == 'explore':
                    prefix = 'explore'
                elif action_type == 'gograsp_container':
                    prefix = 'go grasp container'
                elif action_type == 'gograsp_target':
                    prefix = 'go grasp target'
                elif action_type == 'goto':
                    prefix = 'go to'
                elif action_type == 'putin':
                    prefix = 'put'
                elif action_type == 'sendmessage':
                    prefix = 'send a message'
                elif action_type == 'transport':
                    prefix = 'transport'
                else:
                    continue
                
                # Find all matching plans in available_plans_list
                matching_plans = [plan for plan in available_plans_list if plan.startswith(prefix)]
                if matching_plans:
                    # Use first element (most recently discovered) for grasp actions, random for others
                    if action_type in ['gograsp_target', 'gograsp_container']:
                        selected_plan = matching_plans[0]  # Select first (most recently discovered)
                    elif action_type == 'goto' and shuffled_room:
                        # Use prioritized room selection for goto actions
                        selected_plan = f"go to {shuffled_room[0]}"
                    else:
                        selected_plan = random.choice(matching_plans)  # Random selection for other actions
                    return selected_plan, info, False

        # No matching plan found in cache - randomly select from available plans
        info.update({
            "num_available_actions": num,
            "available_plans": available_plans,
            "available_plans_list": available_plans_list,
            "frame_number": frame_number,
            "held_objects_count_excluding_containers": held_objects_count,
            "visited_rooms_count": visited_rooms_count,
            "successful_objects_count": successful_objects_count,
            "holding_container": holding_container
        })
        
        # Randomly select from available plans if any exist
        if available_plans_list:
            import random
            selected_plan = random.choice(available_plans_list)
            self._debug_print(f"No plan! random chosen: {selected_plan}")
            return selected_plan, info, False
        
        return None, info, False  # No plans available at all
    
    def access_with_gaussian_decay(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room, action_history, dialogue_history, opponent_grabbed_objects = None, opponent_last_room = None):
        """
        Cache version with Gaussian decay applied to action selection.
        Uses score = count * weight * gaussian_decay for prioritization.
        
        Args:
            Same as access() method
            
        Returns:
            tuple: (plan, info, hit_flag) where plan is selected based on Gaussian decay scores
        """
        info = {}
        # self._debug_print(f"[GAUSSIAN_DECAY] current_step: {current_step}")
        
        # Update internal state
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        message = None
        # self._debug_print(f"progress_desc: {self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)}")
        if self.communication and not action_history[-1].startswith('send a message'):
            progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
            message = progress_desc + "\n"
        
        # Get available plans
        # Count only target objects (type 0) in satisfied
        successful_count = sum(1 for obj in satisfied if isinstance(obj, dict) and obj.get('type') == 0)
        available_plans, num, available_plans_list, shuffled_room = self.get_available_plans(message, opponent_last_room, current_step, successful_count)

        # Update dynamic ACTION_WEIGHTS: record appeared types from available plans
        self.update_weight_appeared([self._action_to_cache_entry(p) for p in available_plans_list])

        # If there's a put action, always prioritize it
        put_actions = [plan for plan in available_plans_list if plan.startswith('put ')]
        if put_actions:
            return put_actions[0], info, False

        last_action = action_history[-1]

        frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count = self._calculate_cache_metrics(current_step, holding_objects, rooms_explored, satisfied)
        
        # Use Gaussian decay version
        possible_actions, all_actions = self.get_action_cache_lookup_with_gaussian_decay(last_action, current_step, current_room, rooms_explored, holding_objects, satisfied)
        
        # self._debug_print(f"[GAUSSIAN_DECAY] Available Plans: {available_plans}, {available_plans_list}")
        # self._debug_print(f"[GAUSSIAN_DECAY] Cache Access: {possible_actions[:3] if len(possible_actions) > 3 else possible_actions}")  # Show top 3 with scores
        
        # Match possible_actions with available_plans_list to select a plan
        if num > 0:
            import random
            
            # Group possible_actions by score (priority) and randomize within each group
            def randomize_same_priority_actions_with_decay(actions_with_scores):
                if not actions_with_scores:
                    return []
                
                # Group by score (rounded to avoid float precision issues)
                priority_groups = {}
                for action, score, count, gaussian_decay in actions_with_scores:
                    score_key = round(score, 6)  # Round to 6 decimal places
                    if score_key not in priority_groups:
                        priority_groups[score_key] = []
                    priority_groups[score_key].append(action)
                
                # Sort by score (descending) and randomize within each group
                randomized_actions = []
                for score_key in sorted(priority_groups.keys(), reverse=True):
                    group_actions = priority_groups[score_key].copy()
                    random.shuffle(group_actions)  # Randomize within same priority
                    randomized_actions.extend(group_actions)
                
                return randomized_actions
            
            # Convert possible_actions with randomization for same priorities
            possible_action_types = randomize_same_priority_actions_with_decay(possible_actions)
            
            # First try to match with possible_actions
            for action_type in possible_action_types:
                # Map action types to plan prefixes
                if action_type == 'explore':
                    prefix = 'explore'
                elif action_type == 'gograsp_container':
                    prefix = 'go grasp container'
                elif action_type == 'gograsp_target':
                    prefix = 'go grasp target'
                elif action_type == 'goto':
                    prefix = 'go to'
                elif action_type == 'putin':
                    prefix = 'put'
                elif action_type == 'sendmessage':
                    prefix = 'send a message'
                elif action_type == 'transport':
                    prefix = 'transport'
                else:
                    continue
                
                # Find all matching plans in available_plans_list
                matching_plans = [plan for plan in available_plans_list if plan.startswith(prefix)]
                if matching_plans:
                    # Use first element (most recently discovered) for grasp actions, random for others
                    if action_type in ['gograsp_target', 'gograsp_container']:
                        selected_plan = matching_plans[0]  # Select first (most recently discovered)
                    elif action_type == 'goto' and shuffled_room:
                        # Use prioritized room selection for goto actions
                        selected_plan = f"go to {shuffled_room[0]}"
                    else:
                        selected_plan = random.choice(matching_plans)  # Random selection for other actions
                    return selected_plan, info, False
            
            # If no match found in possible_actions, try all_actions as fallback
            all_action_types = randomize_same_priority_actions_with_decay(all_actions)
            for action_type in all_action_types:
                # Map action types to plan prefixes
                if action_type == 'explore':
                    prefix = 'explore'
                elif action_type == 'gograsp_container':
                    prefix = 'go grasp container'
                elif action_type == 'gograsp_target':
                    prefix = 'go grasp target'
                elif action_type == 'goto':
                    prefix = 'go to'
                elif action_type == 'putin':
                    prefix = 'put'
                elif action_type == 'sendmessage':
                    prefix = 'send a message'
                elif action_type == 'transport':
                    prefix = 'transport'
                else:
                    continue
                
                # Find all matching plans in available_plans_list
                matching_plans = [plan for plan in available_plans_list if plan.startswith(prefix)]
                if matching_plans:
                    # Use first element (most recently discovered) for grasp actions, random for others
                    if action_type in ['gograsp_target', 'gograsp_container']:
                        selected_plan = matching_plans[0]  # Select first (most recently discovered)
                    elif action_type == 'goto' and shuffled_room:
                        # Use prioritized room selection for goto actions
                        selected_plan = f"go to {shuffled_room[0]}"
                    else:
                        selected_plan = random.choice(matching_plans)  # Random selection for other actions
                    return selected_plan, info, False

        # No matching plan found in cache - randomly select from available plans
        info.update({
            "num_available_actions": num,
            "available_plans": available_plans,
            "available_plans_list": available_plans_list,
            "frame_number": frame_number,
            "held_objects_count_excluding_containers": held_objects_count,
            "visited_rooms_count": visited_rooms_count,
            "successful_objects_count": successful_objects_count,
            "holding_container": holding_container
        })
        
        # Randomly select from available plans if any exist
        if available_plans_list:
            import random
            selected_plan = random.choice(available_plans_list)
            # self._debug_print(f"[GAUSSIAN_DECAY] No plan! random chosen: {selected_plan}")
            return selected_plan, info, False
        
        return None, info, False  # No plans available at all
    
    def _calculate_cache_metrics(self, current_step, holding_objects, rooms_explored, satisfied):
        """
        Calculate cache access metrics for tracking performance.
        
        Args:
            current_step: Current frame number
            holding_objects: List of objects being held
            rooms_explored: Dictionary of explored rooms
            satisfied: List of satisfied/transported objects (can include containers)
            
        Returns:
            tuple: (frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container,
                    known_targets_count, known_containers_count, unexplored_rooms_count)
        """
        # Frame number (current step)
        frame_number = current_step
        
        # Count objects held excluding containers (type 1 = container, type 0 = target object)
        # Also count objects inside containers that are being held
        held_objects_count = 0
        holding_container = 0  # 0 = not holding container, 1 = holding container
        
        for obj in holding_objects:
            if obj.get('type') is not None:
                if obj.get('type') == 0:  # Target object
                    held_objects_count += 1
                elif obj.get('type') == 1:  # Container
                    holding_container = 1  # Agent is holding a container
                    if 'contained' in obj:
                        # Count non-None objects in the container
                        contained_objects = [o for o in obj['contained'] if o is not None]
                        held_objects_count += len(contained_objects)
        
        # Count visited rooms (rooms that have been explored)
        visited_rooms_count = len([room for room, status in rooms_explored.items() 
                                 if status is not None and room is not None])
        
        # Count successful objects from satisfied list (exclude containers from count)
        successful_objects_count = 0
        if isinstance(satisfied, list):
            # Count only target objects (type 0) in satisfied list, excluding containers
            successful_objects_count = len([obj for obj in satisfied if obj.get('type') == 0])
        elif isinstance(satisfied, int):
            successful_objects_count = satisfied
        
        # v4 metrics: known targets, known containers, unexplored rooms
        known_targets_count = len(self.object_list[0]) if self.object_list and len(self.object_list) > 0 else 0
        known_containers_count = len(self.object_list[1]) if self.object_list and len(self.object_list) > 1 else 0
        total_rooms = len(self.rooms) if self.rooms else 0
        unexplored_rooms_count = total_rooms - visited_rooms_count
        if unexplored_rooms_count < 0:
            unexplored_rooms_count = 0
        
        return frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count
    
    def _load_action_2gram_data(self):
        """
        Load action 2-gram data from action_2gram_grouped_by_start.txt or _v2.txt
        """
        import os
        
        # Try to find the file in the current directory or tdw-gym subdirectory
        # Prefer v4 > v3 > v2 > v1 if it exists
        possible_paths = [
            'action_2gram_grouped_by_start_v4.txt',
            'tdw-gym/action_2gram_grouped_by_start_v4.txt',
            'action_2gram_grouped_by_start_v3.txt',
            'tdw-gym/action_2gram_grouped_by_start_v3.txt',
            'action_2gram_grouped_by_start_v2.txt',
            'tdw-gym/action_2gram_grouped_by_start_v2.txt',
            'action_2gram_grouped_by_start.txt',
            'tdw-gym/action_2gram_grouped_by_start.txt'
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            self._debug_print("Warning: action_2gram_grouped_by_start.txt (v1-v4) not found")
            self.action_transitions = {}
            return
        
        self.action_transitions = {}
        
        # Detect file format (v2/v3/v4 use same format; v3 adds std, v4 adds known_targets/known_containers/unexplored_rooms)
        is_v2_or_v3_format = '_v2.txt' in file_path or '_v3.txt' in file_path or '_v4.txt' in file_path
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            current_action = None
            
            for line in lines:
                line = line.strip()
                
                # Detect current action header (different format for v2/v3)
                if is_v2_or_v3_format:
                    if line.startswith('FROM: '):
                        current_action = line.split('FROM: ')[1]
                        # Normalize action name from v2/v3 format
                        current_action = self._normalize_action_name_v2(current_action)
                        self.action_transitions[current_action] = []
                else:
                    if line.startswith('Starting Action: '):
                        current_action = line.split('Starting Action: ')[1]
                        self.action_transitions[current_action] = []
                
                # Parse transition lines
                if current_action:
                    if is_v2_or_v3_format:
                        # v2/v3 format: "  → go to                          [  37] ( 28.0%)"
                        if line.startswith('→ ') and '[' in line:
                            self._parse_v2_transition_line(line, lines, current_action)
                    else:
                        # v1 format: " 1. explore -> goto (count: 42) ... [END: ...]"
                        if ' -> ' in line and 'END:' in line:
                            self._parse_v1_transition_line(line, current_action)
                    
            
            format_version = 'v2/v3/v4' if is_v2_or_v3_format else 'v1'
            self._debug_print(f"Loaded action transitions for {len(self.action_transitions)} actions from {format_version} format (file: {file_path})")
            
        except Exception as e:
            self._debug_print(f"Error loading action 2-gram data: {e}")
            import traceback
            self._debug_print(traceback.format_exc())
            self.action_transitions = {}
    
    def _normalize_action_name_v2(self, action):
        """Normalize v2/v3 action names to v1 format"""
        action_map = {
            'explore': 'explore',
            'go to': 'goto',
            'go grasp target': 'gograsp_target',
            'go grasp container': 'gograsp_container',
            'put into container': 'putin',
            'transport': 'transport',
            'send message': 'sendmessage'
        }
        return action_map.get(action, action)
    
    def _parse_v1_transition_line(self, line, current_action):
        """Parse v1 format transition line"""
        try:
            # Format: " 1. explore -> goto (count: 42) (total avg: 94.0) [explore:23.1, goto:71.0] [END: rooms:2.6(med:2.0,min:1,max:6), ...]"
            
            # Extract next action
            arrow_split = line.split(' -> ')
            if len(arrow_split) < 2:
                return
            
            next_action = arrow_split[1].split(' (count:')[0]
            
            # Extract count
            count_str = line.split('(count: ')[1].split(')')[0]
            count = int(count_str)
            
            # Extract END statistics
            end_part = line.split('[END: ')[1].rstrip(']')
            
            # Parse metrics
            metrics = self._parse_metrics_string(end_part)
            
            # For v1, we need to manually set container to 0 if steps metric exists
            # Since container metric might be present in v1
            if 'steps' in metrics and 'container' not in metrics:
                metrics['container'] = {'avg': 0.0, 'med': 0.0, 'min': 0, 'max': 0}
            
            transition_data = {
                'next_action': next_action,
                'count': count,
                'metrics': metrics
            }
            
            self.action_transitions[current_action].append(transition_data)
        except Exception as e:
            self._debug_print(f"Error parsing v1 line: {line[:50]}... Error: {e}")
    
    def _parse_v2_transition_line(self, line, lines, current_action):
        """Parse v2/v3 format transition line (needs next line for metrics)"""
        try:
            # v2/v3 format: "  → go to                          [  37] ( 28.0%)"
            # Next line: "    [action1_dur:90(...), action2_dur:202(...)] [...] [rooms:2.5(...), held:0.6(...), container:0.00(...), satisfied:4.1(...)]"
            # v3 adds std field: "action1@frame:1300(med:998,std:1112.7,min:0,max:2877)" - std is ignored by _parse_metrics_string
            
            # Extract action name (between → and [)
            action_part = line.split('→')[1].split('[')[0].strip()
            next_action = self._normalize_action_name_v2(action_part)
            
            # Extract count (between [ and ])
            count_str = line.split('[')[1].split(']')[0].strip()
            count = int(count_str)
            
            # Find the metrics line (next non-empty line)
            line_idx = lines.index(line + '\n') if (line + '\n') in lines else -1
            if line_idx == -1:
                # Try without newline
                for i, l in enumerate(lines):
                    if l.strip() == line:
                        line_idx = i
                        break
            
            if line_idx >= 0 and line_idx + 1 < len(lines):
                next_line = lines[line_idx + 1].strip()
                
                # Extract the third bracket group containing metrics
                # Format: [action1_dur...] [action2_dur...] [rooms:..., held:..., container:..., satisfied:..., (steps:missing in v2)]
                brackets = []
                depth = 0
                current_bracket = ""
                for char in next_line:
                    if char == '[':
                        if depth == 0:
                            current_bracket = ""
                        depth += 1
                        current_bracket += char
                    elif char == ']':
                        current_bracket += char
                        depth -= 1
                        if depth == 0:
                            brackets.append(current_bracket)
                    elif depth > 0:
                        current_bracket += char
                
                if len(brackets) >= 3:
                    # Second bracket contains frame info: [action1@frame:..., action2@frame:...]
                    # Third bracket contains state metrics: [rooms:..., held:..., ...]
                    
                    # Parse frame info from second bracket
                    frame_info_str = brackets[1].strip('[]')
                    frame_metrics = self._parse_metrics_string(frame_info_str)
                    
                    # Parse state metrics from third bracket
                    metrics_str = brackets[2].strip('[]')
                    metrics = self._parse_metrics_string(metrics_str)
                    
                    # Use action2@frame as 'steps' metric (frame number at end of transition)
                    if 'action2@frame' in frame_metrics:
                        metrics['steps'] = frame_metrics['action2@frame']
                    else:
                        # Fallback if frame info not found
                        metrics['steps'] = {'avg': 0, 'med': 0, 'min': 0, 'max': 9999}
                    
                    # v2 container data is unreliable (all 0), force to accept both 0 and 1
                    metrics['container'] = {'avg': 0.0, 'med': 0.0, 'min': 0, 'max': 1}
                    
                    transition_data = {
                        'next_action': next_action,
                        'count': count,
                        'metrics': metrics
                    }
                    
                    self.action_transitions[current_action].append(transition_data)
        except Exception as e:
            self._debug_print(f"Error parsing v2 line: {line[:50]}... Error: {e}")
    
    def _parse_metrics_string(self, metrics_str):
        """Parse metrics string into dictionary with avg, med, min, max values"""
        metrics = {}
        
        # Handle parentheses depth for parsing
        parts = []
        current_part = ""
        depth = 0
        
        for char in metrics_str:
            if char == '(':
                depth += 1
                current_part += char
            elif char == ')':
                depth -= 1
                current_part += char
            elif char == ',' and depth == 0:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        for metric_str in parts:
            if ':' in metric_str:
                metric_name = metric_str.split(':')[0]
                metric_values = metric_str.split(':', 1)[1]
                
                # Extract average, median, min and max values
                # Format: "2.6(med:2.0,min:1,max:6)" or "1292(med:1348,min:33,max:2806)"
                if '(' in metric_values and 'min:' in metric_values and 'max:' in metric_values:
                    # Extract average value (before parentheses)
                    avg_val = float(metric_values.split('(')[0])
                    
                    # Extract values inside parentheses
                    paren_content = metric_values.split('(')[1].rstrip(')')
                    
                    # Parse median, std, min, max (v3 has std field)
                    med_val = None
                    std_val = None
                    min_val = None
                    max_val = None
                    
                    for part in paren_content.split(','):
                        part = part.strip()
                        if part.startswith('med:'):
                            med_val = float(part.split('med:')[1])
                        elif part.startswith('std:'):
                            std_val = float(part.split('std:')[1])
                        elif part.startswith('min:'):
                            min_val = float(part.split('min:')[1])
                        elif part.startswith('max:'):
                            max_val = float(part.split('max:')[1])
                    
                    metrics[metric_name] = {
                        'avg': avg_val,
                        'med': med_val,
                        'std': std_val,
                        'min': min_val, 
                        'max': max_val
                    }
        
        return metrics
    
    def get_possible_actions_by_last_action(self, last_action, rooms_count, held_count, container_count, satisfied_count, steps_count, known_targets_count=None, known_containers_count=None, unexplored_rooms_count=None):
        """
        Get possible next actions based on last action and current state values.
        
        Args:
            last_action: The previous action taken (e.g., 'explore', 'goto', etc.)
            rooms_count: Current number of rooms visited
            held_count: Current number of objects held
            container_count: Current container status (0 or 1)
            satisfied_count: Current number of satisfied objects
            steps_count: Current step count
            known_targets_count: Number of known target objects (v4, optional)
            known_containers_count: Number of known containers (v4, optional)
            unexplored_rooms_count: Number of unexplored rooms (v4, optional)
            
        Returns:
            list: List of tuples (next_action, count) that match the state constraints, sorted by count
        """
        if last_action not in self.action_transitions:
            return []
        
        possible_actions = []
        
        for transition in self.action_transitions[last_action]:
            next_action = transition['next_action']
            count = transition['count']
            metrics = transition['metrics']
            
            # Check if current state values fall within the min/max ranges
            valid = True
            
            # Check that all required metrics exist with min/max values
            required_metrics = ['rooms', 'held', 'container', 'satisfied', 'steps']
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    raise ValueError(f"Required metric '{metric_name}' not found in transition data for action {last_action} -> {next_action}")
                if 'min' not in metrics[metric_name] or 'max' not in metrics[metric_name]:
                    raise ValueError(f"Required metric '{metric_name}' missing min/max values for action {last_action} -> {next_action}: {metrics[metric_name]}")
            
            # Check rooms constraint
            min_rooms = metrics['rooms']['min']
            max_rooms = metrics['rooms']['max']
            if not (min_rooms <= rooms_count <= max_rooms):
                valid = False
            
            # Check held objects constraint
            min_held = metrics['held']['min']
            max_held = metrics['held']['max']
            if not (min_held <= held_count <= max_held):
                valid = False
            
            # Check container constraint
            min_container = metrics['container']['min']
            max_container = metrics['container']['max']
            if not (min_container <= container_count <= max_container):
                valid = False
            
            # Check satisfied objects constraint
            min_satisfied = metrics['satisfied']['min']
            max_satisfied = metrics['satisfied']['max']
            if not (min_satisfied <= satisfied_count <= max_satisfied):
                valid = False
            
            # Check steps constraint
            min_steps = metrics['steps']['min']
            max_steps = metrics['steps']['max']
            if not (min_steps <= steps_count <= max_steps):
                valid = False
            
            # v4 metrics: only filter if both data and current value are available
            if known_targets_count is not None and 'known_targets' in metrics and metrics['known_targets'].get('min') is not None:
                if not (metrics['known_targets']['min'] <= known_targets_count <= metrics['known_targets']['max']):
                    valid = False
            
            if known_containers_count is not None and 'known_containers' in metrics and metrics['known_containers'].get('min') is not None:
                if not (metrics['known_containers']['min'] <= known_containers_count <= metrics['known_containers']['max']):
                    valid = False
            
            if unexplored_rooms_count is not None and 'unexplored_rooms' in metrics and metrics['unexplored_rooms'].get('min') is not None:
                if not (metrics['unexplored_rooms']['min'] <= unexplored_rooms_count <= metrics['unexplored_rooms']['max']):
                    valid = False
            
            if valid:
                possible_actions.append((next_action, count))
        
        # Sort by count * weight in descending order (importance)
        def get_weighted_score(action_count_tuple):
            action, count = action_count_tuple
            weight = self.ACTION_WEIGHTS.get(action, 0.1)  # Default weight for unknown actions
            return count * weight
        
        possible_actions.sort(key=get_weighted_score, reverse=True)
        
        return possible_actions
    
    def get_possible_actions_by_last_action_with_gaussian_decay(self, last_action, rooms_count, held_count, container_count, satisfied_count, steps_count, known_targets_count=None, known_containers_count=None, unexplored_rooms_count=None):
        """
        Get possible next actions based on last action and current state values with Gaussian decay.
        Score = count * weight * gaussian_decay
        
        Args:
            last_action: The previous action taken (e.g., 'explore', 'goto', etc.)
            rooms_count: Current number of rooms visited
            held_count: Current number of objects held
            container_count: Current container status (0 or 1)
            satisfied_count: Current number of satisfied objects
            steps_count: Current step count (used for both filtering and Gaussian decay calculation)
            known_targets_count: Number of known target objects (v4, optional)
            known_containers_count: Number of known containers (v4, optional)
            unexplored_rooms_count: Number of unexplored rooms (v4, optional)
            
        Returns:
            list: List of tuples (next_action, score, count, gaussian_decay) that match the state constraints, sorted by score
        """
        if last_action not in self.action_transitions:
            return []
        
        import math
        
        possible_actions = []
        
        for transition in self.action_transitions[last_action]:
            next_action = transition['next_action']
            count = transition['count']
            metrics = transition['metrics']
            
            # Check if current state values fall within the min/max ranges
            valid = True
            
            # Check that all required metrics exist with min/max values
            required_metrics = ['rooms', 'held', 'container', 'satisfied', 'steps']
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    raise ValueError(f"Required metric '{metric_name}' not found in transition data for action {last_action} -> {next_action}")
                if 'min' not in metrics[metric_name] or 'max' not in metrics[metric_name]:
                    raise ValueError(f"Required metric '{metric_name}' missing min/max values for action {last_action} -> {next_action}: {metrics[metric_name]}")
            
            # Check rooms constraint
            if not (metrics['rooms']['min'] <= rooms_count <= metrics['rooms']['max']):
                valid = False
            
            # Check held objects constraint
            if not (metrics['held']['min'] <= held_count <= metrics['held']['max']):
                valid = False
            
            # Check container constraint
            if not (metrics['container']['min'] <= container_count <= metrics['container']['max']):
                valid = False
            
            # Check satisfied objects constraint
            if not (metrics['satisfied']['min'] <= satisfied_count <= metrics['satisfied']['max']):
                valid = False
            
            # Check steps constraint
            if not (metrics['steps']['min'] <= steps_count <= metrics['steps']['max']):
                valid = False
            
            # v4 metrics: only filter if both data and current value are available
            if known_targets_count is not None and 'known_targets' in metrics and metrics['known_targets'].get('min') is not None:
                if not (metrics['known_targets']['min'] <= known_targets_count <= metrics['known_targets']['max']):
                    valid = False
            
            if known_containers_count is not None and 'known_containers' in metrics and metrics['known_containers'].get('min') is not None:
                if not (metrics['known_containers']['min'] <= known_containers_count <= metrics['known_containers']['max']):
                    valid = False
            
            if unexplored_rooms_count is not None and 'unexplored_rooms' in metrics and metrics['unexplored_rooms'].get('min') is not None:
                if not (metrics['unexplored_rooms']['min'] <= unexplored_rooms_count <= metrics['unexplored_rooms']['max']):
                    valid = False
            
            if valid:
                # Calculate Gaussian decay based on frame timing
                frame_stats = metrics.get('steps', {'avg': steps_count, 'std': None, 'med': steps_count})
                mean_frame = frame_stats.get('avg', frame_stats.get('med', steps_count))
                std_frame = frame_stats.get('std', None)
                
                if std_frame is not None and std_frame > 0:
                    # Apply Gaussian decay with normalization constant
                    # Normalization rewards consistency (small std = more consistent = higher peak)
                    normalization = 1.0 / (std_frame * math.sqrt(2 * math.pi))
                    exponential = math.exp(-((steps_count - mean_frame) ** 2) / (2 * (std_frame ** 2)))
                    gaussian_decay = normalization * exponential
                else:
                    # If no std available, use default decay
                    default_std = 500.0
                    normalization = 1.0 / (default_std * math.sqrt(2 * math.pi))
                    exponential = math.exp(-((steps_count - mean_frame) ** 2) / (2 * (default_std ** 2)))
                    gaussian_decay = normalization * exponential
                
                # Calculate weighted score with Gaussian decay
                weight = self.ACTION_WEIGHTS.get(next_action, 0.1)
                score = count * weight * gaussian_decay
                
                possible_actions.append((next_action, score, count, gaussian_decay))
        
        # Sort by score in descending order (importance)
        possible_actions.sort(key=lambda x: x[1], reverse=True)
        
        return possible_actions
    
    def get_action_cache_lookup(self, last_action, current_step, current_room, rooms_explored, holding_objects, satisfied):
        """
        Main cache lookup method that uses line 204 values from lm_agent.py to get possible actions.
        
        Args:
            last_action: The last action taken
            current_step: Current step number (corresponds to 'steps' in data)
            current_room: Current room name
            rooms_explored: Dictionary of explored rooms
            holding_objects: List of objects being held
            satisfied: List of satisfied objects
            
        Returns:
            tuple: (possible_actions, all_actions) where both are lists of (action, count) tuples sorted by count
        """
        # Calculate state values using the same method as line 204 area
        frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count = self._calculate_cache_metrics(
            current_step, holding_objects, rooms_explored, satisfied
        )
        # Get possible actions from cache (state-constrained, sorted by count)
        possible_actions = self.get_possible_actions_by_last_action(
            last_action,
            visited_rooms_count,  # rooms
            held_objects_count,   # held
            holding_container,    # container
            successful_objects_count,  # satisfied
            frame_number,         # steps
            known_targets_count,  # v4: known targets
            known_containers_count,  # v4: known containers
            unexplored_rooms_count   # v4: unexplored rooms
        )
        
        # Get all possible actions (no state constraints, sorted by count)
        all_actions = self.get_all_possible_actions_by_count(last_action)
        
        return possible_actions, all_actions 
    
    def get_action_cache_lookup_with_gaussian_decay(self, last_action, current_step, current_room, rooms_explored, holding_objects, satisfied):
        """
        Cache lookup method with Gaussian decay applied to all actions.
        Returns actions sorted by (count * weight * gaussian_decay).
        
        Args:
            last_action: The last action taken
            current_step: Current step number (corresponds to 'steps' in data)
            current_room: Current room name
            rooms_explored: Dictionary of explored rooms
            holding_objects: List of objects being held
            satisfied: List of satisfied objects
            
        Returns:
            tuple: (possible_actions_with_decay, all_actions_with_decay) where both are lists of 
                   (action, score, count, gaussian_decay) tuples sorted by score
        """
        # Calculate state values using the same method as line 204 area
        frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count = self._calculate_cache_metrics(
            current_step, holding_objects, rooms_explored, satisfied
        )
        
        # Get possible actions with Gaussian decay (state-constrained, sorted by score)
        possible_actions = self.get_possible_actions_by_last_action_with_gaussian_decay(
            last_action,
            visited_rooms_count,  # rooms
            held_objects_count,   # held
            holding_container,    # container
            successful_objects_count,  # satisfied
            frame_number,         # steps - used both for filtering and Gaussian decay
            known_targets_count,  # v4: known targets
            known_containers_count,  # v4: known containers
            unexplored_rooms_count   # v4: unexplored rooms
        )
        
        # Get all possible actions with Gaussian decay (no state constraints, sorted by score)
        all_actions = self.get_all_possible_actions_with_gaussian_decay(last_action, frame_number)
        
        return possible_actions, all_actions
    
    def get_all_possible_actions_by_count(self, last_action):
        """
        Get all possible next actions from a previous action, sorted by count * weight (descending).
        
        Args:
            last_action: The previous action taken (e.g., 'explore', 'goto', etc.)
            
        Returns:
            list: List of tuples (next_action, count) sorted by count * weight in descending order
        """
        if last_action not in self.action_transitions:
            return []
        
        # Extract all transitions with their counts
        action_counts = []
        for transition in self.action_transitions[last_action]:
            next_action = transition['next_action']
            count = transition['count']
            action_counts.append((next_action, count))
        
        # Sort by count * weight in descending order (importance)
        def get_weighted_score(action_count_tuple):
            action, count = action_count_tuple
            weight = self.ACTION_WEIGHTS.get(action, 0.1)  # Default weight for unknown actions
            return count * weight
        
        action_counts.sort(key=get_weighted_score, reverse=True)
        
        return action_counts
    
    def get_all_possible_actions_with_gaussian_decay(self, last_action, current_frame):
        """
        Get all possible next actions with Gaussian decay applied based on frame timing.
        Score = count * weight * gaussian_decay
        Gaussian decay = exp(-((current_frame - mean_frame)^2) / (2 * std_frame^2))
        
        Args:
            last_action: The previous action taken (e.g., 'explore', 'goto', etc.)
            current_frame: Current frame number for Gaussian decay calculation
            
        Returns:
            list: List of tuples (next_action, score, count, gaussian_decay) sorted by score in descending order
        """
        if last_action not in self.action_transitions:
            return []
        
        import math
        
        # Extract all transitions with their counts and frame statistics
        action_scores = []
        for transition in self.action_transitions[last_action]:
            next_action = transition['next_action']
            count = transition['count']
            metrics = transition['metrics']
            
            # Get frame statistics (use action2@frame which is the end frame of transition)
            # For fallback, use 'steps' metric
            frame_stats = metrics.get('steps', {'avg': current_frame, 'std': None, 'med': current_frame})
            
            # Calculate Gaussian decay
            mean_frame = frame_stats.get('avg', frame_stats.get('med', current_frame))
            std_frame = frame_stats.get('std', None)
            
            if std_frame is not None and std_frame > 0:
                # Apply Gaussian decay with normalization constant
                # Normalization rewards consistency (small std = more consistent = higher peak)
                normalization = 1.0 / (std_frame * math.sqrt(2 * math.pi))
                exponential = math.exp(-((current_frame - mean_frame) ** 2) / (2 * (std_frame ** 2)))
                gaussian_decay = normalization * exponential
            else:
                # If no std available, use default decay based on distance from mean
                # Use a default std of 500 frames
                default_std = 500.0
                normalization = 1.0 / (default_std * math.sqrt(2 * math.pi))
                exponential = math.exp(-((current_frame - mean_frame) ** 2) / (2 * (default_std ** 2)))
                gaussian_decay = normalization * exponential
            
            # Calculate weighted score with Gaussian decay
            weight = self.ACTION_WEIGHTS.get(next_action, 0.1)  # Default weight for unknown actions
            score = count * weight * gaussian_decay
            
            action_scores.append((next_action, score, count, gaussian_decay))
        
        # Sort by score in descending order (importance)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        return action_scores
        
    def hit(self, last_action, current_action, metrics=None):
        """
        Increase the count for a successful prediction.
        
        Args:
            last_action: The previous action (raw string)
            current_action: The current action that was correctly predicted (raw string)
            metrics: Dictionary containing metrics at time of hit (rooms, held, container, satisfied, steps)
        """
        # Update dynamic ACTION_WEIGHTS: current_action was selected
        self.update_weight_selected(current_action)
        
        # Convert to cache entry format
        last_action_key = self._action_to_cache_entry(last_action)
        current_action_key = self._action_to_cache_entry(current_action)
        
        if last_action_key in self.action_transitions:
            for transition in self.action_transitions[last_action_key]:
                if transition['next_action'] == current_action_key:
                    transition['count'] += 1
                    if transition['next_action'] in ['gograsp_target', 'transport']:
                        transition['count'] += 1
                    # Update metrics if provided
                    if metrics:
                        self._update_transition_metrics(transition, metrics)
                    self._debug_print(f"HIT: Increased count for transition {last_action_key} -> {current_action_key} to {transition['count']}")
                    return
            # Transition exists for last_action but not for this specific next_action - create it
            new_transition = {
                'next_action': current_action_key,
                'count': 2 if current_action_key in ['gograsp_target', 'transport'] else 1,
                'metrics': {}
            }
            if metrics:
                self._update_transition_metrics(new_transition, metrics)
            self.action_transitions[last_action_key].append(new_transition)
            self._debug_print(f"HIT: Created new transition {last_action_key} -> {current_action_key} with count {new_transition['count']}")
        else:
            # No transitions for this last_action yet - create the first one
            self.action_transitions[last_action_key] = [{
                'next_action': current_action_key,
                'count': 2 if current_action_key in ['gograsp_target', 'transport'] else 1,
                'metrics': {}
            }]
            if metrics:
                self._update_transition_metrics(self.action_transitions[last_action_key][0], metrics)
            self._debug_print(f"HIT: Created first transition for {last_action_key} -> {current_action_key} with count 1")
    
    def miss(self, last_action, current_action, predicted_action, metrics=None):
        """
        Decrease the count for an incorrect prediction and increase count for the actual action taken.
        Also update metrics based on the current state when miss occurred.
        
        Args:
            last_action: The previous action (raw string)
            current_action: The actual current action taken (raw string)
            predicted_action: The action that was incorrectly predicted (raw string)
            metrics: Dictionary containing metrics at time of query (rooms, held, container, satisfied, steps)
        """
        # Update dynamic ACTION_WEIGHTS: current_action was actually selected
        self.update_weight_selected(current_action)
        
        # Convert to cache entry format
        last_action_key = self._action_to_cache_entry(last_action)
        current_action_key = self._action_to_cache_entry(current_action)
        predicted_action_key = self._action_to_cache_entry(predicted_action)
        if predicted_action_key == "sendmessage":
            return
        if last_action_key in self.action_transitions:
            # 하나의 for문으로 predicted_action과 current_action 모두 처리
            found_predicted = False
            decreased_current = False
            # send message면 하지말까
            for transition in self.action_transitions[last_action_key]:
                # current_action 감소 (실제 실행 패널티)
                if transition['next_action'] == current_action_key and not decreased_current:
                    if transition['count'] > 1:
                        transition['count'] -= 1
                        self._debug_print(f"MISS: Decreased count for transition {last_action_key} -> {current_action_key} to {transition['count']}")
                    decreased_current = True
                
                # predicted_action 증가 및 metrics 업데이트 (LLM 예측을 보상)
                if transition['next_action'] == predicted_action_key and not found_predicted:
                    transition['count'] += 1
                    if transition['next_action'] in ['gograsp_target', 'transport']:
                        transition['count'] += 1
                    self._debug_print(f"MISS: Increased count for transition {last_action_key} -> {predicted_action_key} to {transition['count']}")
                    # Update metrics if provided
                    if metrics:
                        self._update_transition_metrics(transition, metrics)
                        
                    found_predicted = True
                
                # 두 작업 모두 완료하면 반복 종료
                if decreased_current and found_predicted:
                    break
            
            # predicted_action에 대한 transition이 존재하지 않으면 생성 (LLM 예측)
            if not found_predicted:
                new_transition = {
                    'next_action': predicted_action_key,
                    'count': 2 if predicted_action_key in ['gograsp_target', 'transport'] else 1,
                    'metrics': {}
                }
                if metrics:
                    self._update_transition_metrics(new_transition, metrics)
                self.action_transitions[last_action_key].append(new_transition)
                self._debug_print(f"MISS: Created new transition for predicted action {last_action_key} -> {predicted_action_key} with count {new_transition['count']}")
        else:
            # No transitions for this last_action yet - create the first one for predicted action
            new_transition = {
                'next_action': predicted_action_key,
                'count': 2 if predicted_action_key in ['gograsp_target', 'transport'] else 1,
                'metrics': {}
            }
            if metrics:
                self._update_transition_metrics(new_transition, metrics)
            self.action_transitions[last_action_key] = [new_transition]
            self._debug_print(f"MISS: Created first transition for {last_action_key} -> {predicted_action_key} with count {new_transition['count']}")
                
    def _update_transition_metrics(self, transition, metrics):
        """
        Update the metrics for a transition based on current state metrics.
        
        Args:
            transition: The transition dictionary to update
            metrics: Dictionary containing current state metrics
        """
        if 'metrics' not in transition:
            transition['metrics'] = {}
            
        # Initialize metrics structure if needed
        for metric_name in ['rooms', 'held', 'container', 'satisfied', 'steps', 'known_targets', 'known_containers', 'unexplored_rooms']:
            if metric_name not in transition['metrics']:
                current_value = metrics.get(metric_name, 0)
                transition['metrics'][metric_name] = {
                    'avg': current_value,
                    'med': current_value,
                    'min': current_value,
                    'max': current_value
                }
            else:
                # Update min and max
                curr_val = metrics.get(metric_name, 0)
                transition['metrics'][metric_name]['min'] = min(transition['metrics'][metric_name]['min'], curr_val)
                transition['metrics'][metric_name]['max'] = max(transition['metrics'][metric_name]['max'], curr_val)
    
    # ==================== NO NORMALIZATION VERSIONS ====================
    
    def get_possible_actions_by_last_action_with_gaussian_decay_no_norm(self, last_action, rooms_count, held_count, container_count, satisfied_count, steps_count, known_targets_count=None, known_containers_count=None, unexplored_rooms_count=None):
        """
        Get possible next actions based on last action and current state values with Gaussian decay (NO NORMALIZATION).
        Score = count * weight * exponential_decay (without normalization constant)
        
        Args:
            last_action: The previous action taken (e.g., 'explore', 'goto', etc.)
            rooms_count: Current number of rooms visited
            held_count: Current number of objects held
            container_count: Current container status (0 or 1)
            satisfied_count: Current number of satisfied objects
            steps_count: Current step count (used for both filtering and Gaussian decay calculation)
            known_targets_count: Number of known target objects (v4, optional)
            known_containers_count: Number of known containers (v4, optional)
            unexplored_rooms_count: Number of unexplored rooms (v4, optional)
            
        Returns:
            list: List of tuples (next_action, score, count, gaussian_decay) that match the state constraints, sorted by score
        """
        if last_action not in self.action_transitions:
            return []
        
        import math
        
        possible_actions = []
        
        for transition in self.action_transitions[last_action]:
            next_action = transition['next_action']
            count = transition['count']
            metrics = transition['metrics']
            
            # Check if current state values fall within the min/max ranges
            valid = True
            
            # Check that all required metrics exist with min/max values
            required_metrics = ['rooms', 'held', 'container', 'satisfied', 'steps']
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    raise ValueError(f"Required metric '{metric_name}' not found in transition data for action {last_action} -> {next_action}")
                if 'min' not in metrics[metric_name] or 'max' not in metrics[metric_name]:
                    raise ValueError(f"Required metric '{metric_name}' missing min/max values for action {last_action} -> {next_action}: {metrics[metric_name]}")
            
            # Check rooms constraint
            if not (metrics['rooms']['min'] <= rooms_count <= metrics['rooms']['max']):
                valid = False
            
            # Check held objects constraint
            if not (metrics['held']['min'] <= held_count <= metrics['held']['max']):
                valid = False
            
            # Check container constraint
            if not (metrics['container']['min'] <= container_count <= metrics['container']['max']):
                valid = False
            
            # Check satisfied objects constraint
            if not (metrics['satisfied']['min'] <= satisfied_count <= metrics['satisfied']['max']):
                valid = False
            
            # Check steps constraint
            if not (metrics['steps']['min'] <= steps_count <= metrics['steps']['max']):
                valid = False
            
            # v4 metrics: only filter if both data and current value are available
            if known_targets_count is not None and 'known_targets' in metrics and metrics['known_targets'].get('min') is not None:
                if not (metrics['known_targets']['min'] <= known_targets_count <= metrics['known_targets']['max']):
                    valid = False
            
            if known_containers_count is not None and 'known_containers' in metrics and metrics['known_containers'].get('min') is not None:
                if not (metrics['known_containers']['min'] <= known_containers_count <= metrics['known_containers']['max']):
                    valid = False
            
            if unexplored_rooms_count is not None and 'unexplored_rooms' in metrics and metrics['unexplored_rooms'].get('min') is not None:
                if not (metrics['unexplored_rooms']['min'] <= unexplored_rooms_count <= metrics['unexplored_rooms']['max']):
                    valid = False
            
            if valid:
                # Calculate Gaussian decay based on frame timing (NO NORMALIZATION)
                frame_stats = metrics.get('steps', {'avg': steps_count, 'std': None, 'med': steps_count})
                mean_frame = frame_stats.get('avg', frame_stats.get('med', steps_count))
                std_frame = frame_stats.get('std', None)
                
                if std_frame is not None and std_frame > 0:
                    # Apply only exponential decay (NO normalization constant)
                    exponential = math.exp(-((steps_count - mean_frame) ** 2) / (2 * (std_frame ** 2)))
                    gaussian_decay = exponential
                else:
                    # If no std available, use default decay
                    default_std = 500.0
                    exponential = math.exp(-((steps_count - mean_frame) ** 2) / (2 * (default_std ** 2)))
                    gaussian_decay = exponential
                
                # Calculate weighted score with Gaussian decay (no normalization)
                weight = self.ACTION_WEIGHTS.get(next_action, 0.1)
                score = count * weight * gaussian_decay
                
                possible_actions.append((next_action, score, count, gaussian_decay))
        
        # Sort by score in descending order (importance)
        possible_actions.sort(key=lambda x: x[1], reverse=True)
        
        return possible_actions
    
    def get_all_possible_actions_with_gaussian_decay_no_norm(self, last_action, current_frame):
        """
        Get all possible next actions with Gaussian decay applied based on frame timing (NO NORMALIZATION).
        Score = count * weight * exponential_decay (without normalization constant)
        
        Args:
            last_action: The previous action taken (e.g., 'explore', 'goto', etc.)
            current_frame: Current frame number for Gaussian decay calculation
            
        Returns:
            list: List of tuples (next_action, score, count, gaussian_decay) sorted by score in descending order
        """
        if last_action not in self.action_transitions:
            return []
        
        import math
        
        # Extract all transitions with their counts and frame statistics
        action_scores = []
        for transition in self.action_transitions[last_action]:
            next_action = transition['next_action']
            count = transition['count']
            metrics = transition['metrics']
            
            # Get frame statistics (use action2@frame which is the end frame of transition)
            # For fallback, use 'steps' metric
            frame_stats = metrics.get('steps', {'avg': current_frame, 'std': None, 'med': current_frame})
            
            # Calculate Gaussian decay
            mean_frame = frame_stats.get('avg', frame_stats.get('med', current_frame))
            std_frame = frame_stats.get('std', None)
            
            if std_frame is not None and std_frame > 0:
                # Apply only exponential decay (NO normalization constant)
                exponential = math.exp(-((current_frame - mean_frame) ** 2) / (2 * (std_frame ** 2)))
                gaussian_decay = exponential
            else:
                # If no std available, use default decay based on distance from mean
                # Use a default std of 500 frames
                default_std = 500.0
                exponential = math.exp(-((current_frame - mean_frame) ** 2) / (2 * (default_std ** 2)))
                gaussian_decay = exponential
            
            # Calculate weighted score with Gaussian decay (no normalization)
            weight = self.ACTION_WEIGHTS.get(next_action, 0.1)  # Default weight for unknown actions
            score = count * weight * gaussian_decay
            
            action_scores.append((next_action, score, count, gaussian_decay))
        
        # Sort by score in descending order (importance)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        return action_scores
    
    def get_action_cache_lookup_with_gaussian_decay_no_norm(self, last_action, current_step, current_room, rooms_explored, holding_objects, satisfied):
        """
        Cache lookup method with Gaussian decay applied to all actions (NO NORMALIZATION).
        Returns actions sorted by (count * weight * exponential_decay).
        
        Args:
            last_action: The last action taken
            current_step: Current step number (corresponds to 'steps' in data)
            current_room: Current room name
            rooms_explored: Dictionary of explored rooms
            holding_objects: List of objects being held
            satisfied: List of satisfied objects
            
        Returns:
            tuple: (possible_actions_with_decay, all_actions_with_decay) where both are lists of 
                   (action, score, count, gaussian_decay) tuples sorted by score
        """
        # Calculate state values using the same method as line 204 area
        frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count = self._calculate_cache_metrics(
            current_step, holding_objects, rooms_explored, satisfied
        )
        
        # Get possible actions with Gaussian decay (state-constrained, sorted by score, no normalization)
        possible_actions = self.get_possible_actions_by_last_action_with_gaussian_decay_no_norm(
            last_action,
            visited_rooms_count,  # rooms
            held_objects_count,   # held
            holding_container,    # container
            successful_objects_count,  # satisfied
            frame_number,         # steps - used both for filtering and Gaussian decay
            known_targets_count,  # v4: known targets
            known_containers_count,  # v4: known containers
            unexplored_rooms_count   # v4: unexplored rooms
        )
        
        # Get all possible actions with Gaussian decay (no state constraints, sorted by score, no normalization)
        all_actions = self.get_all_possible_actions_with_gaussian_decay_no_norm(last_action, frame_number)
        
        return possible_actions, all_actions
    
    def access_with_gaussian_decay_no_norm(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room, action_history, dialogue_history, opponent_grabbed_objects = None, opponent_last_room = None):
        """
        Cache version with Gaussian decay applied to action selection (NO NORMALIZATION).
        Uses score = count * weight * exponential_decay for prioritization.
        
        Args:
            Same as access() method
            
        Returns:
            tuple: (plan, info, hit_flag) where plan is selected based on Gaussian decay scores (no normalization)
        """
        info = {}
        # self._debug_print(f"[GAUSSIAN_DECAY_NO_NORM] current_step: {current_step}")
        
        # Update internal state
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        message = None
        # self._debug_print(f"progress_desc: {self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)}")
        if self.communication and not action_history[-1].startswith('send a message'):
            progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
            message = progress_desc + "\n"
        
        # Get available plans
        # Count only target objects (type 0) in satisfied
        successful_count = sum(1 for obj in satisfied if isinstance(obj, dict) and obj.get('type') == 0)
        available_plans, num, available_plans_list, shuffled_room = self.get_available_plans(message, opponent_last_room, current_step, successful_count)

        # Update dynamic ACTION_WEIGHTS: record appeared types from available plans
        self.update_weight_appeared([self._action_to_cache_entry(p) for p in available_plans_list])

        # If there's a put action, always prioritize it
        put_actions = [plan for plan in available_plans_list if plan.startswith('put ')]
        if put_actions:
            return put_actions[0], info, False

        last_action_raw = action_history[-1]
        
        last_action = self._action_to_cache_entry(last_action_raw)
        
        frame_number, held_objects_count, visited_rooms_count, successful_objects_count, holding_container, known_targets_count, known_containers_count, unexplored_rooms_count = self._calculate_cache_metrics(current_step, holding_objects, rooms_explored, satisfied)
        
        # Use Gaussian decay version WITHOUT normalization
        possible_actions, all_actions = self.get_action_cache_lookup_with_gaussian_decay_no_norm(last_action, current_step, current_room, rooms_explored, holding_objects, satisfied)
        
        # self._debug_print(f"[GAUSSIAN_DECAY_NO_NORM] Available Plans: {available_plans}, {available_plans_list}")
        # self._debug_print(f"[GAUSSIAN_DECAY_NO_NORM] Cache Access: {possible_actions[:3] if len(possible_actions) > 3 else possible_actions}")  # Show top 3 with scores
        
        # Match possible_actions with available_plans_list to select a plan
        if num > 0:
            import random
            
            # Group possible_actions by score (priority) and randomize within each group
            def randomize_same_priority_actions_with_decay(actions_with_scores):
                if not actions_with_scores:
                    return []
                
                # Group by score (rounded to avoid float precision issues)
                priority_groups = {}
                for action, score, count, gaussian_decay in actions_with_scores:
                    score_key = round(score, 6)  # Round to 6 decimal places
                    if score_key not in priority_groups:
                        priority_groups[score_key] = []
                    priority_groups[score_key].append(action)
                
                # Sort by score (descending) and randomize within each group
                randomized_actions = []
                for score_key in sorted(priority_groups.keys(), reverse=True):
                    group_actions = priority_groups[score_key].copy()
                    random.shuffle(group_actions)  # Randomize within same priority
                    randomized_actions.extend(group_actions)
                
                return randomized_actions
            
            # Convert possible_actions with randomization for same priorities
            possible_action_types = randomize_same_priority_actions_with_decay(possible_actions)
            
            # First try to match with possible_actions
            for action_type in possible_action_types:
                # Map action types to plan prefixes
                if action_type == 'explore':
                    prefix = 'explore'
                elif action_type == 'gograsp_container':
                    prefix = 'go grasp container'
                elif action_type == 'gograsp_target':
                    prefix = 'go grasp target'
                elif action_type == 'goto':
                    prefix = 'go to'
                elif action_type == 'putin':
                    prefix = 'put'
                elif action_type == 'sendmessage':
                    prefix = 'send a message'
                elif action_type == 'transport':
                    prefix = 'transport'
                else:
                    continue
                
                # Find all matching plans in available_plans_list
                matching_plans = [plan for plan in available_plans_list if plan.startswith(prefix)]
                if matching_plans:
                    # Use first element (most recently discovered) for grasp actions, random for others
                    if action_type in ['gograsp_target', 'gograsp_container']:
                        selected_plan = matching_plans[0]  # Select first (most recently discovered)
                    elif action_type == 'goto' and shuffled_room:
                        # Use prioritized room selection for goto actions
                        selected_plan = f"go to {shuffled_room[0]}"
                    else:
                        selected_plan = random.choice(matching_plans)  # Random selection for other actions
                    return selected_plan, info, False
            
            # If no match found in possible_actions, try all_actions as fallback
            all_action_types = randomize_same_priority_actions_with_decay(all_actions)
            for action_type in all_action_types:
                # Map action types to plan prefixes
                if action_type == 'explore':
                    prefix = 'explore'
                elif action_type == 'gograsp_container':
                    prefix = 'go grasp container'
                elif action_type == 'gograsp_target':
                    prefix = 'go grasp target'
                elif action_type == 'goto':
                    prefix = 'go to'
                elif action_type == 'putin':
                    prefix = 'put'
                elif action_type == 'sendmessage':
                    prefix = 'send a message'
                elif action_type == 'transport':
                    prefix = 'transport'
                else:
                    continue
                
                # Find all matching plans in available_plans_list
                matching_plans = [plan for plan in available_plans_list if plan.startswith(prefix)]
                if matching_plans:
                    # Use first element (most recently discovered) for grasp actions, random for others
                    if action_type in ['gograsp_target', 'gograsp_container']:
                        selected_plan = matching_plans[0]  # Select first (most recently discovered)
                    elif action_type == 'goto' and shuffled_room:
                        # Use prioritized room selection for goto actions
                        selected_plan = f"go to {shuffled_room[0]}"
                    else:
                        selected_plan = random.choice(matching_plans)  # Random selection for other actions
                    return selected_plan, info, False

        # No matching plan found in cache - randomly select from available plans
        info.update({
            "num_available_actions": num,
            "available_plans": available_plans,
            "available_plans_list": available_plans_list,
            "frame_number": frame_number,
            "held_objects_count_excluding_containers": held_objects_count,
            "visited_rooms_count": visited_rooms_count,
            "successful_objects_count": successful_objects_count,
            "holding_container": holding_container
        })
        
        # Randomly select from available plans if any exist
        if available_plans_list:
            import random
            selected_plan = random.choice(available_plans_list)
            self._debug_print(f"No plan! random chosen: {selected_plan}")
            return selected_plan, info, False
        
        return None, info, False  # No plans available at all
