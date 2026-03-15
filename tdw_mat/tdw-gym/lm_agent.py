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
from multiprocessing import Pool, Manager

from LLM.LLM import LLM

CELL_SIZE = 0.125
ANGLE = 15

# Global worker LLM instance for async processing
_worker_llm = None
_worker_rooms_name = None
_worker_goal_objects = None

def _pool_init_llm(llm_args):
    """Initialize LLM instance in worker process"""
    global _worker_llm, _worker_rooms_name, _worker_goal_objects  # 전역변수 추가
    from LLM.LLM import LLM
    
    # Unpack LLM arguments
    source, lm_id, prompt_template_path, communication, cot, args, agent_id, output_dir = llm_args
    
    # Initialize LLM instance
    _worker_llm = LLM(source, lm_id, prompt_template_path, communication, cot, args, agent_id, output_dir)
    _worker_rooms_name = None  # 추가
    _worker_goal_objects = None  # 추가

def _pool_reset_llm(rooms_name, goal_objects):
    """Reset LLM instance in worker process"""
    global _worker_llm, _worker_rooms_name, _worker_goal_objects
    import os
    worker_pid = os.getpid()
    if _worker_llm is not None:
        _worker_rooms_name = rooms_name  # 추가
        _worker_goal_objects = goal_objects  # 추가
        _worker_llm.reset(rooms_name, goal_objects)
    else:
        print(f"[Worker {worker_pid}] WARNING: _worker_llm is None!")

def _pool_reset_llm_wrapper(args):
    """Wrapper for pool.map which only accepts single argument"""
    rooms_name, goal_objects = args
    return _pool_reset_llm(rooms_name, goal_objects)

def async_llm_worker(llm_args, done_flag, result_plan, result_info):
    """Worker function that runs LLM planning in a separate process"""
    try:
        global _worker_llm, _worker_rooms_name, _worker_goal_objects
        
        # Unpack arguments
        num_frames, current_room, rooms_explored, held_objects, satisfied_objects, object_list, object_per_room, action_history, dialogue_history, oppo_held_objects, oppo_last_room, rooms_name, goal_objects = llm_args
        
        # Use the global LLM instance
        if _worker_llm is None:
            raise RuntimeError("Worker LLM not initialized. Call _pool_init_llm first.")
        # 추가: reset이 필요한지 확인
        if _worker_rooms_name != rooms_name or _worker_goal_objects != goal_objects:
            print(f"[Worker {os.getpid()}] Detected new episode, resetting LLM...")
            _pool_reset_llm(rooms_name, goal_objects)

        plan, a_info = _worker_llm.run(
            num_frames, current_room, rooms_explored, held_objects,
            satisfied_objects, object_list, object_per_room, action_history,
            dialogue_history, oppo_held_objects, oppo_last_room
        )

        # Store results in shared memory
        result_plan['plan'] = plan
        result_plan['a_info'] = a_info
        result_info['success'] = True
        
        # Set done flag
        done_flag.value = True
        
    except Exception as e:
        print(f"Error in async LLM worker: {e}")
        import traceback
        traceback.print_exc()
        result_plan['plan'] = None
        result_plan['a_info'] = None
        result_info['error'] = str(e)
        result_info['success'] = False
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
        self.debug = True

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
        self.LLM = LLM(self.source, "gpt-5-nano", self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id, self.output_dir)
        self.action_history = []
        self.dialogue_history = []
        self.plan = None

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
        
        # Initialize async LLM processing components
        self.manager = Manager()
        
        # LLM pool - manage up to 3 async LLM queries
        self.llm_done_flag = [self.manager.Value('b', False) for _ in range(3)]
        self.llm_async_plan = [self.manager.dict() for _ in range(3)]
        self.llm_async_info = [self.manager.dict() for _ in range(3)]
        self.llm_async_result = [None, None, None]
        self.llm_running = [False, False, False]
        
        # Create process pool with initializer
        self.llm_args = (self.source, self.lm_id, self.prompt_template_path, 
                        self.communication, self.cot, self.args, self.agent_id, self.output_dir)
        self.llm_pool = None
        self.target_plan = {}  # Dict for storing async LLM results: {step_num: plan}
        self.llm_issue_step = [None, None, None]  # Track which step each async query was issued for


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
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, gt_behavior=True, env_api=env_api, constraint_type = None, map_size = self.map_size, scene_bounds = self._scene_bounds)
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
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
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
        self.LLM.reset(self.rooms_name, self.goal_objects)
        self.save_img = save_img
        
        # Reset async LLM pool
        if hasattr(self, 'llm_pool') and self.llm_pool is not None:
            try:
                self.llm_pool.terminate()
                self.llm_pool.join()
            except Exception as e:
                print(f"Warning: Error terminating LLM pool: {e}")
        
        # Recreate the pool for the new episode
        self.llm_pool = Pool(
            processes=3,
            initializer=_pool_init_llm,
            initargs=(self.llm_args,)
        )
        
        # Reset LLM in worker process with current episode info
        # Send reset task to all 3 workers and wait for completion
        try:
            reset_results = []
            for i in range(3):
                result = self.llm_pool.apply_async(_pool_reset_llm_wrapper, ((self.rooms_name, self.goal_objects),))
                reset_results.append(result)
            
            # Wait for all resets to complete
            for i, result in enumerate(reset_results):
                result.get(timeout=30)  # Wait up to 30 seconds per worker
                print(f"Worker {i} reset completed")
        except Exception as e:
            print(f"Warning: Could not reset worker LLM: {e}")
        
        # Reset all async-related state
        for i in range(3):
            self.llm_async_result[i] = None
            self.llm_done_flag[i].value = False
            self.llm_running[i] = False
            self.llm_async_plan[i].clear()
            self.llm_async_info[i].clear()
            self.llm_issue_step[i] = None
        self.target_plan.clear()
        self.draft_plan = {}
        self.last_verified_step = -1

    def move(self, target_pos):
        self.local_step += 1
        action, path_len = self.agent_memory.move_to_pos(target_pos)
        return action

    def gotoroom(self):
        target_room = ' '.join(self.plan.split(' ')[2: 4])
        if target_room[-1] == ',': target_room = target_room[:-1]
        if self.debug:
            print(target_room)
        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # add an interruption if anything new happens
        if len(self.new_object_list[0]) + len(self.new_object_list[1]) + len(self.new_object_list[2]) > 0:
            self.action_history[-1] = self.action_history[-1].replace(self.plan, f'go to {self.current_room}')
            self.new_object_list = {0: [], 1: [], 2: []}
            self.plan = None
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

    def LLM_plan(self):
        return self.LLM.run(self.num_frames, self.current_room, self.rooms_explored, self.obs['held_objects'],[self.object_info[x] for x in self.satisfied if x in self.object_info], self.object_list, self.object_per_room, self.action_history, self.dialogue_history, self.obs['oppo_held_objects'], self.oppo_last_room)
    
    def submit_async_llm_query(self, step_num):
        """Submit an async LLM query to the pool
        
        Args:
            step_num: The step number to generate plan for.
        """
        if self.llm_pool is None:
            raise RuntimeError("LLM pool not initialized")
        
        # Find an available slot (up to 3)
        slot_idx = None
        for i in range(3):
            if not self.llm_running[i]:
                slot_idx = i
                break
        
        if slot_idx is None:
            print(f"Warning: All 3 LLM slots are busy. Query not submitted.")
            return False
        
        # Prepare arguments for LLM worker - pass current action_history and dialogue_history
        satisfied_objects = [self.object_info[x] for x in self.satisfied if x in self.object_info]
        llm_args = (
            self.num_frames, self.current_room, self.rooms_explored, 
            self.obs['held_objects'], satisfied_objects, self.object_list, 
            self.object_per_room, self.action_history, self.dialogue_history,
            self.obs['oppo_held_objects'], self.oppo_last_room,
            self.rooms_name, self.goal_objects
        )
        
        # Reset shared state for this slot
        self.llm_done_flag[slot_idx].value = False
        self.llm_async_plan[slot_idx].clear()
        self.llm_async_info[slot_idx].clear()
        
        # Record the step this query was issued for
        self.llm_issue_step[slot_idx] = step_num
        
        # Submit async task
        self.llm_async_result[slot_idx] = self.llm_pool.apply_async(
            async_llm_worker,
            (llm_args, self.llm_done_flag[slot_idx], self.llm_async_plan[slot_idx], self.llm_async_info[slot_idx])
        )
        self.llm_running[slot_idx] = True
        print(f"Submitted LLM query for step {step_num} in slot {slot_idx}")
        return True
    
    def check_async_llm_result(self):
        """Check async LLM queries and write completed results to self.target_plan
        
        Only clears a slot if its result has arrived AND there are no earlier issue_steps still pending.
        This ensures we don't lose track of earlier steps.
        
        Returns:
            List of newly arrived steps
        """
        newly_arrived = []
        
        # Sort slots by issue_step to process in order
        running_slots = [(slot_idx, self.llm_issue_step[slot_idx]) 
                        for slot_idx in range(3) 
                        if self.llm_running[slot_idx] and self.llm_issue_step[slot_idx] is not None]
        running_slots.sort(key=lambda x: x[1])  # Sort by issue_step
        
        for slot_idx, issue_step in running_slots:
            if self.llm_done_flag[slot_idx].value:
                try:
                    self.llm_async_result[slot_idx].get(timeout=0.1)
                    plan = self.llm_async_plan[slot_idx].get('plan', None)
                    a_info = self.llm_async_plan[slot_idx].get('a_info', None)
                    
                    # Store result in target_plan dict with step_num as key (only plan)
                    if issue_step is not None and plan is not None:
                        # Check if this is a new arrival
                        if issue_step not in self.target_plan:
                            newly_arrived.append(issue_step)
                        self.target_plan[issue_step] = plan
                        print(f"Collected LLM result for step {issue_step} from slot {slot_idx}")
                    
                    # Clear the slot
                    self.llm_async_result[slot_idx] = None
                    self.llm_running[slot_idx] = False
                    self.llm_done_flag[slot_idx].value = False
                    self.llm_issue_step[slot_idx] = None
                    self.llm_async_plan[slot_idx].clear()
                    self.llm_async_info[slot_idx].clear()
                    print(f"Cleared slot {slot_idx} (step {issue_step})")
                except Exception as e:
                    print(f"Error retrieving LLM pool result from slot {slot_idx}: {e}")
            else:
                break

        # Clear any invalidated slots (issue_step set to None after rollback)
        for slot_idx in range(3):
            if self.llm_running[slot_idx] and self.llm_issue_step[slot_idx] is None:
                if self.llm_done_flag[slot_idx].value:
                    try:
                        self.llm_async_result[slot_idx].get(timeout=0.1)
                    except Exception:
                        pass
                    self.llm_async_result[slot_idx] = None
                    self.llm_running[slot_idx] = False
                    self.llm_done_flag[slot_idx].value = False
                    self.llm_async_plan[slot_idx].clear()
                    self.llm_async_info[slot_idx].clear()
                    print(f"Cleared invalidated slot {slot_idx}")

        return newly_arrived
    
    def cleanup(self):
        """Clean up multiprocessing resources"""
        from multiprocessing import TimeoutError
        
        # Collect running results with timeout before closing
        if hasattr(self, 'llm_pool') and self.llm_pool is not None:
            for slot_idx in range(3):
                if self.llm_running[slot_idx] and self.llm_async_result[slot_idx] is not None:
                    try:
                        # Wait up to 60 seconds for this result
                        self.llm_async_result[slot_idx].get(timeout=60)
                        # Extract and store the result
                        plan = self.llm_async_plan[slot_idx].get('plan', None)
                        issue_step = self.llm_issue_step[slot_idx]
                        if plan is not None and issue_step is not None:
                            self.target_plan[issue_step] = plan
                            print(f"Collected LLM pool result during cleanup (step {issue_step}, slot {slot_idx})")
                    except TimeoutError:
                        print(f"Warning: Timeout waiting for LLM pool slot {slot_idx}. Force closing.")
                    except Exception as e:
                        print(f"Warning: Error collecting LLM pool result from slot {slot_idx}: {e}")
            
            self.llm_pool.close()
            self.llm_pool.join()
            self.llm_pool = None

    def act(self, obs):
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
              
        newly_arrived = self.check_async_llm_result()

        # If all 3 slots are full, wait until at least one completes
        if all(self.llm_running):
            print("All 3 LLM slots are busy. Waiting for at least one to complete...")
            while all(self.llm_running):
                newly_arrived.extend(self.check_async_llm_result())
                time.sleep(1)  # Short sleep to avoid busy waiting
            print("At least one slot is now available.")
            
        # Only allow plan switching if last action was move (0) or turn (1, 2)
        allow_plan_switching = True
        if self.last_action is not None and isinstance(self.last_action, dict) and 'type' in self.last_action:
            if self.last_action['type'] not in [0, 1, 2]:
                allow_plan_switching = False
                print(f"Last action type is {self.last_action['type']} - skipping plan switching (only allowed for move/turn)")
        
        # Verify all unverified target_plan entries (not just newly_arrived)
        # This ensures results collected during non-switchable steps are still checked later
        unverified_steps = sorted([s for s in self.target_plan if s > self.last_verified_step])
        if allow_plan_switching and unverified_steps and hasattr(self, 'draft_plan') and self.draft_plan:
            for new_step in unverified_steps:

                target = self.target_plan[new_step]

                # Compare target with all draft plans >= new_step
                all_differ = True
                for step in sorted([s for s in self.draft_plan if s >= new_step and s > self.last_verified_step]):
                    draft = self.draft_plan[step]
                    if target == draft:
                        all_differ = False
                        print(f"Verified: target_plan[{new_step}] == draft_plan[{step}]")
                        break

                if not all_differ:
                    self.last_verified_step = new_step
                    continue  # Verified, proceed to next step

                # Mismatch: attempt rollback to target plan
                target_plan_str = self.target_plan[new_step]

                # Check if target plan is still available in current state
                self.LLM.current_room = self.current_room
                self.LLM.rooms_explored = self.rooms_explored
                self.LLM.holding_objects = self.obs['held_objects'] if hasattr(self, 'obs') else []
                self.LLM.object_list = self.object_list if hasattr(self, 'object_list') else {0: [], 1: [], 2: []}
                self.LLM.obj_per_room = self.object_per_room if hasattr(self, 'object_per_room') else {}

                if target_plan_str.startswith('send a message:'):
                    message = target_plan_str
                else:
                    message = None
                _, _, available_plans_list = self.LLM.get_available_plans(message=message)

                if target_plan_str in available_plans_list:
                    # Rollback: switch to target plan
                    self.plan = target_plan_str
                    self.target_pos = None

                    cached_plan_name = 'send a message' if target_plan_str.startswith('send a message:') else target_plan_str
                    self.action_history[-1] = self.action_history[-1].replace(
                        self.action_history[-1].split(' at step ')[0],
                        cached_plan_name
                    )
                    self.last_verified_step = new_step
                    print(f"Rollback: switched to target_plan[{new_step}]: {target_plan_str} (last_verified_step={new_step})")

                    # Invalidate all async queries for steps > new_step
                    for i in range(3):
                        if self.llm_running[i] and self.llm_issue_step[i] is not None:
                            if self.llm_issue_step[i] > new_step:
                                print(f"Invalidating slot {i} (step {self.llm_issue_step[i]})")
                                self.llm_issue_step[i] = None

                    break  # Done - next new_plan will resume draft+target cycle
                else:
                    self.last_verified_step = new_step
                    print(f"Target plan '{target_plan_str}' not available in current state, skipping (last_verified_step={new_step})")

        action = None
        lm_times = 0
        new_plan = False
        while action is None:
            if self.plan is None:
                self.target_pos = None
                if lm_times > 0:
                    print(info)
                if lm_times > 3:
                    raise Exception(f"retrying LM_plan too many times")

                # Call LLM synchronously (draft model)
                plan, a_info = self.LLM_plan()
                new_plan = True
                # Store in draft_plan dict (only plan)
                if not hasattr(self, 'draft_plan'):
                    self.draft_plan = {}
                if plan is not None:
                    self.draft_plan[self.num_frames] = plan
                
                if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
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

        info.update({"action": action,
                     "plan": self.plan})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action
        
        # Submit async LLM query for next step if any slot is available
        if new_plan:
            print(f"[Step {self.num_frames}] Before submit: llm_running={self.llm_running}")
            result = self.submit_async_llm_query(self.num_frames)
            print(f"[Step {self.num_frames}] After submit: llm_running={self.llm_running}")
        
        return action