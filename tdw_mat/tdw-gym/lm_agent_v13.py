import json
import os
import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
import copy
from PIL import Image
from agent_memory import AgentMemory
import multiprocessing as mp
from multiprocessing import Pool, Manager

from LLM.LLM import LLM
from action_cache import ActionCache

CELL_SIZE = 0.125
ANGLE = 15

# Global variable for worker process LLM instance (one per process)
_worker_llm = None

def clean_action(action):
    """Helper method to remove 'at step X' suffix from action strings"""
    return action.split(" at step ")[0] if " at step " in action else action

def _pool_init_llm(llm_args):
    """Initialize LLM instance in worker process"""
    global _worker_llm
    source, lm_id, prompt_template_path, communication, cot, args, agent_id, output_dir = llm_args
    _worker_llm = LLM(source, lm_id, prompt_template_path, communication, cot, args, agent_id, output_dir)

def _pool_reset_llm(rooms_name, goal_objects):
    """Reset LLM instance in worker process"""
    global _worker_llm
    if _worker_llm is not None:
        _worker_llm.reset(rooms_name, goal_objects)

def async_llm_worker(llm_args, done_flag, result_plan, result_info):
    """Worker function that runs LLM planning in a separate process"""
    try:
        global _worker_llm
        
        # Unpack arguments
        (num_frames, current_room, rooms_explored, held_objects, satisfied_objects,
         object_list, object_per_room, action_history, dialogue_history, 
         oppo_held_objects, oppo_last_room, goal_objects) = llm_args
        
        # Use the global LLM instance
        plan, info = _worker_llm.run(num_frames, current_room, rooms_explored, held_objects,
                                    satisfied_objects, object_list, object_per_room, 
                                    action_history, dialogue_history, oppo_held_objects, oppo_last_room)
        
        # Store results in shared memory
        result_plan['plan'] = plan
        result_info.update(info)
        
        # Set done flag
        done_flag.value = True
        
    except Exception as e:
        print(f"Error in async LLM worker: {e}")
        result_plan['plan'] = None
        result_info['error'] = str(e)
        done_flag.value = True

class lm_agent:
    def  __init__(self, agent_id, logger, max_frames, args, output_dir = 'results'):
        self.with_oppo = None
        self.oppo_pos = None
        self.with_character = None
        self.color2id = None
        self.satisfied = None
        self.object_list = None
        self.container_held = None
        self.gt_mask = None
        self.object_info = {} # {id: {id: xx, type: 0/1/2, name: sss, position: x,y,z}}
        self.object_per_room = {} # {room_name: {0/1/2: [{id: xx, type: 0/1/2, name: sss, position: x,y,z}]}}
        self.object_discovery_time = {} # {object_id: discovery_timestamp}
        self.id_map = None
        self.object_map = None
        self.agent_id = agent_id
        self.agent_type = 'lm_agent'
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 120)
        self.save_img = True
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.logger = logger
        random.seed(1024)
        self.debug = args.debug

        self.new_object_list = None
        self.visible_objects = None
        self.num_frames = None
        self.steps = None
        self.obs = None
        self.local_step = 0

        self.last_action = None
        self.pre_action = None

        self.goal_objects = None
        self.dropping_object = None

        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.action_cache = ActionCache(debug_writer=self.debug_write if self.debug else None, agent_id=self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.plan = None
        
        # Store LLM initialization args for pool creation in reset()
        self.llm_init_args = (self.source, self.lm_id, self.prompt_template_path, 
                             self.communication, self.cot, self.args, self.agent_id, self.output_dir)

        # Pool will be created in reset() for each episode
        self.llm_pool = None
        
        # Initialize manager and shared memory objects (used across episodes)
        self.manager = Manager()
        self.async_result = None
        self.async_query_data = None
        self.llm_done_flag = self.manager.Value('b', False)
        self.async_plan = self.manager.dict()
        self.async_info = self.manager.dict()
        self.async_action_history = []  # Store action history length when async query was submitted
        self.async_plan_buffer = None  # Single variable to store one async plan
        self.goal_objects = None  # Will be set in reset()
        self.turn_on_llm = True
        self.llm_running = False
        self.rooms_name = None
        self.rooms_explored = {}
        self.position = None
        self.forward = None
        self.current_room = None
        self.holding_objects_id = None
        self.oppo_holding_objects_id = None
        self.oppo_last_room = None
        self.rotated = None
        self.navigation_threshold = 5
        self.detection_threshold = 5
        self.evicted_plan = None
        self.replace_plan = False
        self.submitted = False
        self.interrupt = False
        self.replacing_plan = None

    def debug_write(self, message):
        """Write debug message to both console and file"""
        print(message)
        if self.debug:
            debug_dir = os.path.join(self.output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"agent_{self.agent_id}_{self.lm_id}_main_debug.log"), "a", encoding="utf-8") as f:
                f.write(f"{message}\n")

    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        depth = self.obs['depth'].copy()
        for i in range(len(self.obs['seg_mask'])):
            for j in range(len(self.obs['seg_mask'][0])):
                if (self.obs['seg_mask'][i][j] != color).any():
                    depth[i][j] = 1e9
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)

        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].copy().reshape(-1)
        yy = yy[index].copy().reshape(-1)
        depth = depth[index].copy().reshape(-1)

        pc = np.stack((xx, yy, depth, np.ones_like(xx)))

        pc = pc.reshape(4, -1)

        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)
        return rpc[:3]

    def cal_object_position(self, o_dict):
        pc = self.get_pc(o_dict['seg_color'])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]


    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def get_object_list(self):
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        for object_type in [0, 1, 2]:
            obj_map_indices = np.where(self.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.id_map[i, j]
                if id in self.satisfied or id in self.holding_objects_id or id in self.oppo_holding_objects_id or self.object_info[id] in object_list[object_type]:
                    continue
                object_list[object_type].append(self.object_info[id])
                room = self.env_api['belongs_to_which_room'](self.object_info[id]['position'])
                if room is None:
                    self.logger.warning(f"obj {self.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                self.object_per_room[room][object_type].append(self.object_info[id])
        
        # Sort objects by discovery time (most recently discovered first)
        for object_type in [0, 1, 2]:
            # Check all objects have discovery time before sorting
            for obj in object_list[object_type]:
                assert obj['id'] in self.object_discovery_time, f"Object {obj['id']} not found in object_discovery_time"
            
            object_list[object_type].sort(
                key=lambda obj: self.object_discovery_time[obj['id']], 
                reverse=True
            )
        self.object_list = object_list


    def get_new_object_list(self):
        self.visible_objects = self.obs['visible_objects']
        self.new_object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict['id'] is None: continue
            self.color2id[o_dict['seg_color']] = o_dict['id']
            if o_dict['id'] is None or o_dict['id'] in self.satisfied or o_dict['id'] in self.with_character or o_dict['type'] == 4:
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            object_id = o_dict['id']
            new_obj = False
            if object_id not in self.object_info:
                self.object_info[object_id] = {}
                new_obj = True
            # Update discovery time for all visible objects (both new and existing)
            self.object_discovery_time[object_id] = self.num_frames
            self.object_info[object_id]['id'] = object_id
            self.object_info[object_id]['type'] = o_dict['type']
            self.object_info[object_id]['name'] = o_dict['name']
            if o_dict['type'] == 3: # the agent
                if o_dict['id'] == self.opponent_agent_id:
                    position = self.cal_object_position(o_dict)
                    self.oppo_pos = position
                    if position is not None:
                        oppo_last_room = self.env_api['belongs_to_which_room'](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                continue
            if object_id in self.satisfied or object_id in self.with_character:
                continue
            self.object_info[object_id]['position'] = position
            if o_dict['type'] == 0:
                x, y, z = self.object_info[object_id]['position']

                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[0].append(object_id)

            elif o_dict['type'] == 1:
                x, y, z = self.object_info[object_id]['position']
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[1].append(object_id)
            elif o_dict['type'] == 2:
                x, y, z = self.object_info[object_id]['position']
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[2].append(object_id)

    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any(): 
                return -100 # wall
            else: return self.agent_id # agent
        else: return self.color2id[color]

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold = 1.0):
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        if self.plan.startswith('transport'):
            if self.env_api['belongs_to_which_room'](np.array([x, 0, z])) != self.env_api['belongs_to_which_room'](np.array([gx, 0, gz])):
                return False
        return d < threshold

    def reset(self, obs, goal_objects = None, output_dir = None, env_api = None, rooms_name = None, agent_color = [-1, -1, -1], agent_id = 0, gt_mask = True, save_img = True):
        
        self.force_ignore = []
        # Keep episode-specific output_dir separate for AgentMemory (images)
        # But preserve run-level self.output_dir for debug/LLM logs
        episode_output_dir = output_dir if output_dir is not None else self.output_dir
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = episode_output_dir, gt_mask=self.gt_mask, gt_behavior=True, env_api=env_api, constraint_type = None, map_size = self.map_size, scene_bounds = self._scene_bounds, debug_writer=self.debug_write if self.debug else None)
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self.oppo_pos = None
        goal_count = sum([v for k, v in goal_objects.items()])
        # Do NOT overwrite self.output_dir - keep run-level output_dir for debug/LLM logs
        self.last_action = None
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.object_discovery_time = {}
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0
        # print(self.obs.keys())
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)
        self.rotated = None
        self.rooms_explored = {}
        
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        
        # Initialize missing variables
        self.target_pos = None
        self.explore_count = 0
        self.async_action_history = []
        self.turn_on_llm = True
        self.llm_running = False
        self.async_query_metrics = {}
        self.async_plan_buffer = None
        self.submitted = False
        self.local_step = 0
        self.replace_plan = False
        # Reset action cache for fresh start each episode
        self.action_cache = ActionCache(self.rooms_name, self.goal_objects, debug_writer=self.debug_write if self.debug else None, agent_id=self.agent_id)
        self.evicted_plan = None
        self.interrupt = False
        self.replacing_plan = None
        self.gt_mask = gt_mask
        if self.gt_mask == True:
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection
            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
        self.navigation_threshold = 5
        # print(self.rooms_name)
        
        self.save_img = save_img
        
        # Reset async query state by closing and recreating the LLM pool
        # This ensures all pending queries from previous episode are completely terminated
        if hasattr(self, 'llm_pool') and self.llm_pool is not None:
            try:
                # Terminate the pool immediately without waiting for pending tasks
                self.llm_pool.terminate()
                self.llm_pool.join()
            except Exception as e:
                print(f"Warning: Error terminating LLM pool: {e}")
        
        # Recreate the pool for the new episode
        self.llm_pool = Pool(
            processes=1,
            initializer=_pool_init_llm,
            initargs=(self.llm_init_args,)
        )
        
        # Reset LLM in the new worker process with current episode info
        try:
            self.llm_pool.apply(_pool_reset_llm, (self.rooms_name, self.goal_objects))
        except Exception as e:
            print(f"Warning: Could not reset worker LLM: {e}")
        
        # Reset all async-related state
        self.async_result = None
        self.llm_done_flag.value = False
        self.async_plan.clear()
        self.async_info.clear()

    def move(self, target_pos):
        self.local_step += 1
        action, path_len = self.agent_memory.move_to_pos(target_pos)
        return action

    def gotoroom(self):
        target_room = ' '.join(self.plan.split(' ')[2: 4])
        if target_room[-1] == ',': target_room = target_room[:-1]

        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # add an interruption if anything new happens
        if len(self.new_object_list[0]) + len(self.new_object_list[1]) + len(self.new_object_list[2]) > 0:
            if self.debug:
                self.debug_write(f"9. Interruption!")

            self.interrupt = True
            self.action_history[-1] = self.action_history[-1].replace(self.plan, f'go to {self.current_room}')
            self.new_object_list = {0: [], 1: [], 2: []} # why reset?
            self.plan = None #make llm decide new action
            return None
        return self.move(target_pos)


    def goexplore(self):
        target_room = ' '.join(self.plan.split(' ')[-2:])
        # assert target_room == self.current_room, f"{target_room} != {self.current_room}"
        target_pos = self.env_api['center_of_room'](target_room)
        self.explore_count += 1
        dis_threshold = 1 + self.explore_count / 50
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)
        if self.rotated is None:
            self.rotated = 0
        if self.rotated == 16:
            self.rotated = 0
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            return None
        self.rotated += 1
        action = {"type": 1}
        return action

    def gograsp(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.object_map[np.where(self.id_map == target_object_id)] = 0
            self.id_map[np.where(self.id_map == target_object_id)] = 0
            self.plan = None
            return None
        
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_info[target_object_id]['position'])
        target_object_pos = self.target_pos

        if target_object_id not in self.object_info or target_object_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)
        action = {"type": 3, "object": target_object_id, "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'}
        return action
    
    def goput(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            self.with_character = [self.agent_id]
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_list[2][0]['position'])
        target_pos = self.target_pos

        if not self.reach_target_pos(target_pos, 1.5):
            return self.move(target_pos)
        if self.obs["held_objects"][0]['type'] is not None:
            self.dropping_object += [self.obs["held_objects"][0]['id']]
            if self.obs["held_objects"][0]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][0]['contained'] if x is not None]
            return {"type": 5, "arm": "left"}
        else:
            self.dropping_object += [self.obs["held_objects"][1]['id']]
            if self.obs["held_objects"][1]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][1]['contained'] if x is not None]
            return {"type": 5, "arm": "right"}

    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action
    
    def detect(self):
        detect_result = self.detection_model(self.obs['rgb'][..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((self.obs['rgb'].shape[0], self.obs['rgb'].shape[1], 3)).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3: continue
            mask = detect_result['masks'][:,:,i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](mask = mask, name = self.detection_model.cls_to_name_map(label)).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.with_character)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask


    def submit_async_llm_query(self):
        """Submit an async LLM query to the process pool"""
        if self.async_result is not None and not self.async_result.ready():
            print("Previous query still running")
            return  # Previous query still running
        
        # Store current action history for later comparison
        self.async_action_history = self.action_history.copy() # query 할때의 history

        # Store metrics at query time for cache updates
        visited_rooms_count = len([room for room, status in self.rooms_explored.items() if status is not None and room is not None])
        
        held_objects_count = 0
        holding_container = 0
        for obj in self.obs['held_objects']:
            if obj.get('type') is not None:
                if obj.get('type') == 0:  # Target object
                    held_objects_count += 1
                elif obj.get('type') == 1:  # Container
                    holding_container = 1
                    if 'contained' in obj:
                        contained_objects = [o for o in obj['contained'] if o is not None]
                        held_objects_count += len(contained_objects)
        
        # Count only type 0 (target objects) in satisfied - consistent with LLM.py
        if isinstance(self.satisfied, list) and all(isinstance(x, dict) for x in self.satisfied):
            satisfied_count = len([obj for obj in self.satisfied if obj.get('type') == 0])
        else:
            # If satisfied contains non-dict items (just IDs), count all of them as they are target objects
            satisfied_count = len(self.satisfied)
            
        # Store metrics in a dictionary
        self.async_query_metrics = {
            'rooms': visited_rooms_count,
            'held': held_objects_count,
            'container': holding_container,
            'satisfied': satisfied_count,
            'steps': self.num_frames
        }
        
        # Reset shared memory
        self.llm_done_flag.value = False
        self.async_plan.clear()
        self.async_info.clear()
        
        # Prepare arguments for the worker
        llm_args = (
            self.num_frames, self.current_room, dict(self.rooms_explored), 
            self.obs['held_objects'], [self.object_info[x] for x in self.satisfied if x in self.object_info],
            self.object_list, self.object_per_room, list(self.async_action_history), 
            list(self.dialogue_history), self.obs['oppo_held_objects'], self.oppo_last_room,
            self.goal_objects
        )
        
        # Submit to process pool
        self.async_result = self.llm_pool.apply_async(
            async_llm_worker, 
            (llm_args, self.llm_done_flag, self.async_plan, self.async_info)
        )
        self.llm_running = True

    def check_async_llm_result(self):
        """Check if async LLM query is complete and return results"""
        if self.llm_running and self.llm_done_flag.value and self.async_result is not None:
            try:
                # Get the result (this should be immediate since done_flag is True)
                self.async_result.get(timeout=0.1)
                plan = self.async_plan.get('plan', None)
                info = dict(self.async_info)
                
                # Clear the async result
                self.async_result = None
                self.llm_running = False
                return plan, info
            except Exception as e:
                print(f"Error retrieving async LLM result: {e}")
                self.async_result = None
                self.llm_running = False
                return None, {}
        
        return None, {}

    def cleanup(self):
        """Clean up multiprocessing resources"""
        if hasattr(self, 'llm_pool') and self.llm_pool is not None:
            self.llm_pool.close()
            self.llm_pool.join()
            self.llm_pool = None

    def act(self, obs):
        self.debug_write(f"############AGENT #{self.agent_id} START FRAME#{self.num_frames}############")
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)
        self.num_frames = obs['current_frames']
        self.steps += 1

        if not self.gt_mask:
            self.obs['visible_objects'], self.obs['seg_mask'] = self.detect()

        if obs['valid'] == False:
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.satisfied.append(self.last_action['object'])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"
    
        if self.communication:
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}")
    
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api['get_room_distance'](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'
        if self.agent_id not in self.with_character: self.with_character.append(self.agent_id) # DWH: buggy env, need to solve later.
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        for x in self.obs['held_objects']:
            if x['type'] == 0:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
            elif x['type'] == 1:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                #self.with_character.append(x['id'])
                for y in x['contained']:
                    if y is None:
                        break
                    if y not in self.with_character: self.with_character.append(y)
                    #self.with_character.append(y)
        oppo_name = {}
        oppo_type = {}
        for x in self.obs['oppo_held_objects']:
            if x['type'] == 0:
                self.oppo_holding_objects_id.append(x['id'])
                self.with_oppo.append(x['id'])
                oppo_name[x['id']] = x['name']
                oppo_type[x['id']] = x['type']
            elif x['type'] == 1:
                self.oppo_holding_objects_id.append(x['id'])
                self.with_oppo.append(x['id'])
                oppo_name[x['id']] = x['name']
                oppo_type[x['id']] = x['type']
                for i, y in enumerate(x['contained']):
                    if y is None:
                        break
                    self.with_oppo.append(y)
                    oppo_name[y] = x['contained_name'][i]
                    oppo_type[y] = 0
        for obj in self.with_oppo:
            if obj not in self.satisfied:
                self.satisfied.append(obj)
                self.object_info[obj] = {
                    "name": oppo_name[obj],
                    "id": obj,
                    "type": oppo_type[obj],
                }
                self.object_map[np.where(self.id_map == obj)] = 0
                self.id_map[np.where(self.id_map == obj)] = 0
        if not self.obs['valid']: # invalid, the object is not there
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
        if len(self.dropping_object) > 0 and self.obs['status'] == 1:
            self.logger.info(f"Drop object: {self.dropping_object}")
            self.satisfied += self.dropping_object
            self.dropping_object = []
            if len(self.holding_objects_id) == 0:
                self.logger.info("successful drop!")
                self.plan = None

        ignore_obstacles = []
        ignore_ids = []
        self.with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x is None or x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        for x in self.force_ignore:
            self.with_character.append(x)

        for x in self.obs["oppo_held_objects"]:
            if x is None or x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        self.agent_memory.update(
            obs, ignore_ids=ignore_ids, ignore_obstacles=ignore_obstacles, save_img = self.save_img
        )

        if self.obs['status'] == 0: # ongoing
            self.debug_write(f"############AGENT #{self.agent_id} END FRAME#{self.num_frames}############")
            return {'type': 'ongoing'}

        self.get_new_object_list()
        print(self.new_object_list)
        self.get_object_list()

        info = {'satisfied': self.satisfied,
                'object_list': self.object_list,
                'new_object_list': self.new_object_list,
                'current_room': self.current_room,
                'visible_objects': self.filtered(self.obs['visible_objects']),
                'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
              }
        
        async_plan, async_info = self.check_async_llm_result() # when to check is important

        if async_plan is not None:
            message = None
            _, _, available_plans_list = self.get_available_plans(message, self.rooms_name, self.obs['held_objects'])
            self.debug_write(f"1. available_plans_list: {available_plans_list}")
            self.debug_write(f"2. async_plan: {async_plan}, current_plan: {self.plan}")
            self.debug_write(f"3. action_history: {self.action_history}")
            self.debug_write(f"4. async_action_history: {self.async_action_history}")    
            # Find actions that were added after async query was submitted
            # Include the last overlapping action from async_action_history
            if len(self.async_action_history) > 0 and len(self.action_history) >= len(self.async_action_history):
                new_actions_since_async = self.action_history[len(self.async_action_history) - 1:]
            else:
                new_actions_since_async = []
            
            # Clean actions for comparison (remove "at step X" suffix)
            new_actions_cleaned = [clean_action(a) for a in new_actions_since_async]
            self.debug_write(f"5. new_actions_cleaned: {new_actions_cleaned}")
            prev_action = new_actions_cleaned[0]
            if(async_plan == clean_action(self.action_history[-1])):
                self.turn_on_llm = False
            else:
                self.turn_on_llm = True
            
            if(len(new_actions_cleaned) == 1):
                if prev_action == async_plan:
                    prev_action = clean_action(self.async_action_history[-2])
                    self.action_cache.hit(prev_action, async_plan)
                elif async_plan in available_plans_list:
                    self.turn_on_llm = False
                    prev_action = new_actions_cleaned[0]
                    self.action_cache.hit(prev_action, async_plan)
                    
                    # Check if last action type is 0, 1, 2 (movement/rotation actions)
                    # Only replace plan during movement, not during critical actions like pick/put
                    if self.last_action is not None and isinstance(self.last_action, dict) and self.last_action.get('type') in [0, 1, 2]:
                        self.target_pos = None
                        self.plan = async_plan

                        # Add "canceled" suffix to the last action
                        # send a message is instant so doesn't need canceled, only ongoing actions
                        if not self.action_history[-1].startswith("send a message"):
                            self.action_history[-1] += " - canceled"
                        # Add the new async_plan with current step number
                        self.action_history.append(f"{async_plan} at step {self.num_frames}")
                    else:
                        # Store in buffer if not in movement state
                        self.async_plan_buffer = async_plan
                    
                else:
                    prev_action = clean_action(self.async_action_history[-2])
                    self.action_cache.miss(prev_action, new_actions_cleaned[0], async_plan, self.async_query_metrics)
            else:
                next_action = new_actions_cleaned[1]
                if (async_plan in new_actions_cleaned) and (next_action == async_plan):
                    self.action_cache.hit(prev_action, async_plan)
                elif (async_plan in new_actions_cleaned):
                    self.action_cache.hit(prev_action, async_plan)
                else:
                    self.action_cache.miss(prev_action, next_action, async_plan, self.async_query_metrics)
                    if async_plan in available_plans_list:
                        # Check if last action type is 0, 1, 2 (movement/rotation actions)
                        # Only replace plan during movement, not during critical actions like pick/put
                        self.turn_on_llm = False
                        if self.last_action is not None and isinstance(self.last_action, dict) and self.last_action.get('type') in [0, 1, 2]:
                            self.target_pos = None
                            self.plan = async_plan

                            # Add "canceled" suffix to the last action
                            # send a message is instant so doesn't need canceled, only ongoing actions
                            if not self.action_history[-1].startswith("send a message"):
                                self.action_history[-1] += " - canceled"
                            
                            # Add the new async_plan with current step number
                            self.action_history.append(f"{async_plan} at step {self.num_frames}")
                        else:
                            # Store in buffer if not in movement state
                            self.async_plan_buffer = async_plan

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                self.target_pos = None
                plan = None
                a_info = {}  # Initialize a_info

                if self.async_plan_buffer is not None:
                    _, _, available_plans_list = self.get_available_plans(message, self.rooms_name, self.obs['held_objects'])
                    if self.async_plan_buffer in available_plans_list:
                        plan = self.async_plan_buffer
                        self.async_plan_buffer = None
                        a_info = {"source": "async_buffer"}  # Mark source
                    else:
                        prev_action = clean_action(self.action_history[-2])
                        next_action = clean_action(self.action_history[-1])
                        self.action_cache.miss(prev_action, next_action, self.async_plan_buffer, self.async_query_metrics)
                        self.async_plan_buffer = None
                if plan is None:
                    self.turn_on_llm = True
                    plan, a_info, async_select = self.action_cache.access( # 이걸 하 async plan을 적용할때
                        current_step=self.num_frames,
                        current_room=self.current_room,
                        rooms_explored=self.rooms_explored,
                        holding_objects=self.obs['held_objects'],
                        satisfied=[self.object_info[x] for x in self.satisfied if x in self.object_info],
                        object_list=self.object_list,
                        obj_per_room=self.object_per_room,
                        action_history=self.action_history,
                        dialogue_history=self.dialogue_history,
                        opponent_grabbed_objects=self.obs['oppo_held_objects'],
                        opponent_last_room=self.oppo_last_room
                    )

                if plan is None:
                    plan = "[wait]"

                self.plan = plan
                self.action_history.append(f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}")

                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                lm_times += 1
            if self.plan.startswith('go to'):
                action = self.gotoroom()
            elif self.plan.startswith('explore'):
                self.explore_count = 0
                action = self.goexplore()
            elif self.plan.startswith('go grasp'):
                action = self.gograsp()
            elif self.plan.startswith('put'):
                action = self.putin()
            elif self.plan.startswith('transport'):
                action = self.goput()
            #    self.with_character = [self.agent_id]
            elif self.plan.startswith('send a message:'):
                action = {"type": 6,
                          "message": ' '.join(self.plan.split(' ')[3:])}
                self.plan = None
            elif self.plan.startswith('wait'):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")
        if (not self.llm_running) and self.turn_on_llm:
            self.submit_async_llm_query()

        info.update({"action": action,
                     "plan": self.plan})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action

        self.debug_write(f"plan: {self.plan}")
        self.debug_write(f"plan: {self.action_history}")
        self.debug_write(f"############AGENT #{self.agent_id} END FRAME#{self.num_frames}############")
        return action

    def get_available_plans(self, message, rooms, holding_objects):
        """
        Generate available action plans based on current state.
        
        Args:
            rooms: List of available rooms
            holding_objects: List of objects currently being held
            
        Returns:
            tuple: (available_plans, num, available_plans_list)
        """
        available_plans = []
        if self.communication and message is not None:
            available_plans.append(f"send a message: {message}")
        if holding_objects[0]['type'] is None or holding_objects[1]['type'] is None:
            for obj in self.object_list[0]:
                available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")
            if not (holding_objects[0]['type'] == 1 or holding_objects[1]['type'] == 1):
                for obj in self.object_list[1]:
                    available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
        else:
            if holding_objects[0]['type'] == 1 and holding_objects[0]['contained'][-1] is None and holding_objects[1]['type'] == 0:
                available_plans.append(f"put <{holding_objects[1]['name']}> ({holding_objects[1]['id']}) into the container <{holding_objects[0]['name']}> ({holding_objects[0]['id']})")
            elif holding_objects[1]['type'] == 1 and holding_objects[1]['contained'][-1] is None and holding_objects[0]['type'] == 0:
                available_plans.append(f"put <{holding_objects[0]['name']}> ({holding_objects[0]['id']}) into the container <{holding_objects[1]['name']}> ({holding_objects[1]['id']})")
                
        # Only allow transport if holding target objects (type 0) or containers with target objects
        has_target_objects = False
        for obj in holding_objects:
            if obj['type'] == 0:  # Target object
                has_target_objects = True
                break
            elif obj['type'] == 1 and 'contained' in obj:  # Container with objects
                if any(contained_obj is not None for contained_obj in obj['contained']):
                    has_target_objects = True
                    break
        
        if has_target_objects and len(self.object_list[2]) != 0:
            available_plans.append(f"transport objects I'm holding to the bed")
        for room in rooms:
            # Skip current room, None values, and fully explored rooms (marked as 'all')
            if room == self.current_room or room is None or room == 'None' or (room in self.rooms_explored and self.rooms_explored[room] == 'all'):
                continue
            available_plans.append(f"go to {room}")
            
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            available_plans.append(f"explore current room {self.current_room}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    def progress2text(self, current_step, satisfied, opponent_grabbed_objects, opponent_last_room,): #important
        s = f"I've taken {current_step}/3000 steps. "

        sss = {}
        for room, obj_list in self.object_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            s_bed = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss
            if len(obj_list[2]) > 0:
                s_bed = 'the goal position bed'
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += 'nothing'
            elif s_obj != "" and s_con != "" and s_bed == "":
                sr += s_obj + ', and ' + s_con
            elif s_obj != "" and s_con == "" and s_bed != "":
                sr += s_obj + ', and ' + s_bed
            elif s_obj == "" and s_con != "" and s_bed != "":
                sr += s_con + ', and ' + s_bed
            elif s_obj != "" and s_con != "" and s_bed != "":
                sr += s_obj + ', ' + s_con + ', and ' + s_bed
            else:
                sr += s_obj + s_con + s_bed
            sss[room] = sr

        if len(satisfied) == 0:
            if len(self.object_list[2]) == 0:
                s += "I haven't found the goal position bed. "
            else:
                s += ""
        else:
            s += f"{'I' if self.single else 'We'}'ve already transported "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x['type'] == 0]) == 0:
                s += 'nothing'
            s += ', '.join([f"<{x['name']}> ({x['id']})" for x in unique_satisfied if x['type'] == 0])
            s += ' to the bed. '

        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj['type'] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj['type'] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj['contained']): # container iteration
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = 'nothing'
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

        if self.holding_objects[0]["type"] == 0 and self.holding_objects[1]['type'] == 0:
            s += f"I'm holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}). "
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "I'm holding nothing. "
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"I'm holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"I'm holding {s_hold[0]}{s_hold[1]}"

        # ## print(self.current_room, self.obj_per_room)
        if self.current_room not in self.rooms_explored: pred_room = 'none'
        else: pred_room = self.rooms_explored[self.current_room]
        if pred_room != 'all' and sss[self.current_room] == 'nothing':
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it. "
        else:
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it and found {sss[self.current_room]}. "
        ### opponent modeling ?? opponent
        if not self.single:
            s_hold = ["", ""]
            for i, obj in enumerate(opponent_grabbed_objects):
                if obj['type'] == 0:
                    s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
                elif obj['type'] == 1:
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj['contained']):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = 'nothing'
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "
            if opponent_grabbed_objects[0]["type"] == 0 and opponent_grabbed_objects[1]['type'] == 0:
                ss = f"two target objects <{opponent_grabbed_objects[0]['name']}> ({opponent_grabbed_objects[0]['id']}) and <{opponent_grabbed_objects[1]['name']}> ({opponent_grabbed_objects[1]['id']}). "
            if s_hold[0] == "" and s_hold[1] == "":
                ss = "nothing. "
            elif s_hold[0] != "" and s_hold[1] != "":
                ss = f"{s_hold[0][:-2]}, and {s_hold[1]}"
            else:
                ss = f"{s_hold[0]}{s_hold[1]}"

            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == self.current_room:
                s += f"I also see {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == self.current_room:
                continue
            #s += f"I've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
            if room not in self.rooms_explored: pred_room = 'none'
            else: pred_room = self.rooms_explored[room]
            if pred_room != 'all' and sss[room] == 'nothing':
                s += f"I've explored {pred_room} of the {room}. "
            else:
                s += f"I've explored {pred_room} of the {room}, and I found {sss[room]} there. "

        return s
