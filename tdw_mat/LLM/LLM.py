import random
import re
from typing import List
import json
import pandas as pd
import backoff
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from openai import AzureOpenAI
from openai import OpenAIError
import os
from openai import OpenAI
import requests
from torch.profiler import record_function

class LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id,
                 output_dir='results'
                 ):
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 1 else "he"
        self.output_dir = output_dir
        self.debug = sampling_parameters.debug
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        
        # Read prompts from TXT file (CSV format also supported for backward compatibility)
        if self.prompt_template_path.endswith('.txt'):
            with open(self.prompt_template_path, 'r') as f:
                content = f.read().strip()
            
            self.prompt_template = content.replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
            # Load gen_prompt.txt for communication message generation
            gen_prompt_path = self.prompt_template_path.replace('prompt_com.txt', 'gen_prompt.txt')
            if communication and os.path.exists(gen_prompt_path):
                with open(gen_prompt_path, 'r') as f:
                    gen_content = f.read().strip()
                self.generator_prompt_template = gen_content.replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
            else:
                self.generator_prompt_template = None
        else:
            # Backward compatibility for CSV format
            df = pd.read_csv(self.prompt_template_path)
            self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
            if communication:
                self.generator_prompt_template = df['prompt'][1].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
            else:
                self.generator_prompt_template = None

        self.communication = communication
        self.cot = cot
        self.source = source
        self.model = None
        self.tokenizer = None
        self.lm_id = lm_id
        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id or 'gpt-5' in lm_id or 'gpt-4.1' in lm_id or 'chat' in lm_id
        self.OPENAI_KEY = None
        self.total_cost = 0

        if self.source == "openai":
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == 'hf':
            self.tokenizer = LlamaTokenizer.from_pretrained(self.lm_id, use_fast=True)
            self.model = LlamaForCausalLM.from_pretrained(self.lm_id, device_map='auto', load_in_4bit=True)
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                # 'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                # 'early_stopping': True,
            }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):

            @backoff.on_exception(backoff.expo, OpenAIError)
            def openai_generate(prompt, sampling_params):
                usage = 0
                try:
                    if 'gpt-5' in self.lm_id or 'gpt-4.1' in self.lm_id:
                        # Handle new GPT-5/GPT-4.1 models with different parameter requirements
                        # Create a clean parameters dictionary with only supported parameters
                        adapted_params = sampling_params.copy()
                        
                        # Convert max_tokens to max_completion_tokens - this is the only parameter we'll pass
                        # Since other parameters like temperature and top_p appear to have restrictions
                        if 'max_tokens' in sampling_params:
                            adapted_params.pop('max_tokens')
                        if 'temperature' in sampling_params:
                            adapted_params['temperature'] = 1
                        # We're not passing 'n', 'temperature', or 'top_p' to avoid further API errors
                        # The model will use its defaults
                        response = client.chat.completions.create(model=self.lm_id, messages=prompt, **adapted_params)
                        # Always log token usage
                        llm_dir = os.path.join(self.output_dir, 'LLM')
                        os.makedirs(llm_dir, exist_ok=True)
                        with open(os.path.join(llm_dir, f"token_usage_agent_{self.agent_id}.txt"), 'a') as f:
                            f.write(f"LM: {lm_id}\n")
                            f.write(f"Agent: {self.agent_id}\n")
                            f.write(f"Input tokens: {response.usage.prompt_tokens}\n")
                            f.write(f"Output tokens: {response.usage.completion_tokens}\n")
                            f.write(f"Total tokens: {response.usage.total_tokens}\n")
                            f.write("=" * 80 + "\n\n")

                        io_entry = {
                            'input': [{'role': msg.get('role', ''), 'content': msg.get('content', '')} for msg in prompt],
                            'output': response.choices[0].message.content,
                        }
                        with open(os.path.join(llm_dir, f"chat_io_agent_{self.agent_id}.jsonl"), 'a') as f:
                            f.write(json.dumps(io_entry) + '\n')
                        
                        if self.debug:
                            # Additional detailed logging when debug is enabled
                            with open(os.path.join(llm_dir, f"token_usage_agent_{self.agent_id}.txt"), 'a') as f:
                                f.write(f"LM: {lm_id}\n")
                                f.write(f"Agent: {self.agent_id}\n")
                                f.write(f"Input tokens: {response.usage.prompt_tokens}\n")
                                f.write(f"Output tokens: {response.usage.completion_tokens}\n")
                                f.write(f"Total tokens: {response.usage.total_tokens}\n")
                                f.write("=" * 80 + "\n")
                                f.write("INPUT PROMPT:\n")
                                # Extract and write only the content from each message
                                for i, msg in enumerate(prompt):
                                    f.write(f"[{msg.get('role', 'unknown').upper()}]\n")
                                    f.write(msg.get('content', '') + "\n")
                                    if i < len(prompt) - 1:
                                        f.write("\n")
                                f.write("=" * 80 + "\n")
                                f.write("OUTPUT:\n")
                                f.write(response.choices[0].message.content + "\n")
                                f.write("=" * 80 + "\n\n")
                        # Use the actual number of choices returned rather than referencing 'n' parameter
                        # since we're not sure if 'n' is in adapted_params
                        generated_samples = [response.choices[i].message.content for i in
                                             range(len(response.choices))]
                        # Placeholder pricing for newer models
                        usage = response.usage.total_tokens * 0.01 / 1000
                    elif self.chat:
                        # Handle traditional chat models (GPT-3.5, GPT-4)
                        response = client.chat.completions.create(model=self.lm_id, messages=prompt, **sampling_params)
                        # Always log token usage
                        llm_dir = os.path.join(self.output_dir, 'LLM')
                        os.makedirs(llm_dir, exist_ok=True)
                        with open(os.path.join(llm_dir, f"token_usage_agent_{self.agent_id}.txt"), 'a') as f:
                            f.write(f"LM: {lm_id}\n")
                            f.write(f"Agent: {self.agent_id}\n")
                            f.write(f"Input tokens: {response.usage.prompt_tokens}\n")
                            f.write(f"Output tokens: {response.usage.completion_tokens}\n")
                            f.write(f"Total tokens: {response.usage.total_tokens}\n")
                            f.write("=" * 80 + "\n\n")

                        io_entry = {
                            'input': [{'role': msg.get('role', ''), 'content': msg.get('content', '')} for msg in prompt],
                            'output': response.choices[0].message.content,
                        }
                        with open(os.path.join(llm_dir, f"chat_io_agent_{self.agent_id}.jsonl"), 'a') as f:
                            f.write(json.dumps(io_entry) + '\n')
                        
                        if self.debug:
                            # Additional detailed logging when debug is enabled
                            with open(os.path.join(llm_dir, f"token_usage_agent_{self.agent_id}.txt"), 'a') as f:
                                f.write(f"LM: {lm_id}\n")
                                f.write(f"Agent: {self.agent_id}\n")
                                f.write(f"Input tokens: {response.usage.prompt_tokens}\n")
                                f.write(f"Output tokens: {response.usage.completion_tokens}\n")
                                f.write(f"Total tokens: {response.usage.total_tokens}\n")
                                f.write("=" * 80 + "\n")
                                f.write("INPUT PROMPT:\n")
                                # Extract and write only the content from each message
                                for i, msg in enumerate(prompt):
                                    f.write(f"[{msg.get('role', 'unknown').upper()}]\n")
                                    f.write(msg.get('content', '') + "\n")
                                    if i < len(prompt) - 1:
                                        f.write("\n")
                                f.write("=" * 80 + "\n")
                                f.write("OUTPUT:\n")
                                f.write(response.choices[0].message.content + "\n")
                                f.write("=" * 80 + "\n\n")
                        generated_samples = [response.choices[i].message.content for i in
                                             range(sampling_params['n'])]
                        # Handle pricing for different traditional models
                        if ('gpt-4' in self.lm_id) or ('gpt4' in self.lm_id):
                            usage = response.usage.prompt_tokens * 0.03 / 1000 + response.usage.completion_tokens * 0.06 / 1000
                        elif 'gpt-3.5' in self.lm_id:
                            usage = response.usage.total_tokens * 0.002 / 1000
                    # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                    #                   range(sampling_params['n'])]
                    elif "text-" in lm_id:
                        response = client.completions.create(model=lm_id, prompt=prompt, **sampling_params)
                        # ## print(json.dumps(response, indent=4))
                        if self.debug:
                            with open(f"LLM/raw_v2_{lm_id}_agent_{self.agent_id}.json", 'a') as f:
                                f.write(f"LM: {lm_id}\n")
                                f.write(f"Agent: {self.agent_id}\n")
                                f.write(json.dumps(response, indent=4))
                                f.write('\n')
                        generated_samples = [response.choices[i].text for i in range(sampling_params['n'])]
                    # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                    #               range(sampling_params['n'])]
                    else:
                        raise ValueError(f"{lm_id} not available!")
                except OpenAIError as e:
                    ## print(e)
                    raise e
                return generated_samples, usage

            def tokenize_dialog(dialog):
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                prompt_tokens = []
                # ## print(dialog)
                if dialog[0]["role"] == "system":
                    dialog = [
                                 {
                                     "role": dialog[1]["role"],
                                     "content": B_SYS
                                                + dialog[0]["content"]
                                                + E_SYS
                                                + dialog[1]["content"],
                                 }
                             ] + dialog[2:]
                assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                    [msg["role"] == "assistant" for msg in dialog[1::2]]
                ), (
                    "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )
                dialog_tokens: List[int] = sum(
                    [
                        [self.tokenizer.bos_token_id] +
                        self.tokenizer.encode(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                            add_special_tokens=False
                        )
                        + [self.tokenizer.eos_token_id]
                        for prompt, answer in zip(dialog[::2], dialog[1::2], )
                    ],
                    [],
                )
                assert (
                        dialog[-1]["role"] == "user"
                ), f"Last message must be from user, got {dialog[-1]['role']}"
                dialog_tokens += [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}", add_special_tokens=False
                )
                prompt_tokens.append(dialog_tokens)
                return torch.tensor(prompt_tokens).to('cuda')
            @torch.inference_mode()
            def hf_generate(prompt, sampling_params):
                if self.chat:
                    input_ids = tokenize_dialog(prompt)
                else:
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
                prompt_len = input_ids.shape[-1]
                output_dict = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, # max_length=prompt_len + sampling_params['max_new_tokens'],
                                             **sampling_params)
                generated_samples = self.tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                generated_samples = [s.strip() for s in generated_samples]
                generated_samples = [s[:-4] if '</s>' in s[-4:] else s for s in generated_samples]
                if self.debug:
                    pass
                    ## print(generated_samples)
                return generated_samples, 0

            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    return openai_generate(prompt, sampling_params)
                elif self.source == 'hf':
                    return hf_generate(prompt, sampling_params)
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None


    def reset(self, rooms_name, goal_objects):
        self.rooms = rooms_name
        self.goal_objects = goal_objects
        self.goal_desc = self.goal2description(goal_objects)


    def goal2description(self, goals):  # {predicate: count}
        s = "Transport "
        r = None
        for object_name, count in goals.items():
            s += f"{count} {object_name}{'s' if count > 1 else ''}, "

        s = s[:-2] + f" to the bed."
        return s

    def _clean_action(self, action):
        """
        Clean action by removing status suffix and 'keep current action:' prefix.
        
        Examples:
            "go to room_1, explored part" -> "go to room_1"
            "keep current action: go to room_1" -> "go to room_1"
            "go grasp target object <apple> (123) in current room" -> "go grasp target object <apple> (123)"
            "go grasp target object <apple> (123)" -> "go grasp target object <apple> (123)" (unchanged)
        """
        # Remove "keep current action: " prefix
        if action.startswith("keep current action: "):
            action = action.replace("keep current action: ", "")
        
        # Remove ", explored {status}" suffix for "go to room" actions
        if action.startswith("go to") and ", explored " in action:
            action = action.split(", explored ")[0]
        
        # Remove " in current room" suffix for object/container grasp actions
        if " in current room" in action:
            action = action.replace(" in current room", "")
        
        return action

    def parse_answer(self, available_actions, text):
        flags = 'AC'
        
        # First, try to extract answer from "ANSWER: X" format (CoT)
        import re
        answer_match = re.search(r'ANSWER:\s*([A-Z])', text, re.IGNORECASE)
        if answer_match:
            option_letter = answer_match.group(1).upper()
            option_index = ord(option_letter) - ord('A')
            if 0 <= option_index < len(available_actions):
                return self._clean_action(available_actions[option_index]), flags
        
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message:"):
                action = "send a message"
            # For "go to room, status" format, check both full and partial match
            elif action.startswith("go to") and ", " in action:
                # Extract just "go to room" part for matching
                action_without_status = action.split(", ")[0]
                if action_without_status.lower() in text.lower():
                    # Return the action without the status part
                    return self._clean_action(available_actions[i]), flags
            # For "go grasp ... in current room" format, check both with and without suffix
            elif " in current room" in action:
                action_without_room = action.replace(" in current room", "")
                if action_without_room.lower() in text.lower():
                    return self._clean_action(available_actions[i]), flags
            if action.lower() in text.lower():
                return self._clean_action(available_actions[i]), flags
        sents = text.split('\n')  # Split by space
        words = []
        for sent in sents:
            words.extend(sent.split(' '))
        words = list(filter(None, words))  # Remove empty strings from the result

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in words or f"{option}," in words or f"{option}\n" in text.split(" ") or f"Option {option}" in text or f"({option})" in words or f"action {option}" in text or (len(text) <= 2 and option in text) or option in words:
                return self._clean_action(action), flags
        ## print("WARNING! Fuzzy match!")
        flags = "Fuzzy match"
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('go to'):
                # act = 'go to'
                # Remove ", status" suffix if present for "go to room, status" format
                action_base = action.split(", ")[0] if ", " in action else action
                # Extract room name (it's just the room without brackets for "go to")
                name = action_base.replace("go to ", "")
                id = name  # For "go to room", name and id are the same
            elif action.startswith('explore'):
                act = 'explore'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('go grasp'):
                act = 'grasp'
                # Remove " in current room" suffix if present
                action_clean = action.replace(" in current room", "")
                name = action_clean.split(' ')[-2][1:-1]
                id = action_clean.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if name in text and id in text:
                return self._clean_action(action), flags
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act = "None"
            name = "None"
            id = "None"
            if action.startswith('go to'):
                # Remove ", status" suffix if present for "go to room, status" format
                action_base = action.split(", ")[0] if ", " in action else action
                name = action_base.replace("go to ", "")
                id = name
            elif action.startswith('explore'):
                act = 'explore'
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('go grasp'):
                act = 'grasp'
                # Remove " in current room" suffix if present
                action_clean = action.replace(" in current room", "")
                name = action_clean.split(' ')[-2][1:-1]
                id = action_clean.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if act.lower() in text.lower() or name.lower() in text.lower():
                return self._clean_action(action), flags
        ## print("WARNING! No available action parsed!!! Random choose one")
        flags = "failed to parse"
        return self._clean_action(random.choice(available_actions)), flags


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
        explored_status = self.rooms_explored.get(self.current_room, 'none')
        if explored_status == 'all':
            explored_str = "fully explored"
        elif explored_status == 'none':
            explored_str = "not explored yet"
        else:
            explored_str = f"explored {explored_status} of the area"
        lines.append(f"I'm located in the {self.current_room}, where I've {explored_str}.")
        
        # Opponent information
        if not self.single:
            oppo_hold_parts = []
            for obj in opponent_grabbed_objects:
                if obj['type'] == 0:
                    oppo_hold_parts.append(f"a target object <{obj['name']}> ({obj['id']})")
                elif obj['type'] == 1:
                    contained = []
                    for j, o in enumerate(obj['contained']):
                        if o is None:
                            break
                        contained.append(f"<{obj['contained_name'][j]}> ({o})")
                    if len(contained) == 0:
                        oppo_hold_parts.append(f"an empty container <{obj['name']}> ({obj['id']})")
                    else:
                        cont_str = ', and '.join([', '.join(contained[:-1]), contained[-1]]) if len(contained) > 1 else contained[0]
                        oppo_hold_parts.append(f"a container <{obj['name']}> ({obj['id']}) that contains {cont_str}")
            
            if len(oppo_hold_parts) == 0:
                oppo_holding = "nothing"
            elif len(oppo_hold_parts) == 1:
                oppo_holding = oppo_hold_parts[0]
            else:
                oppo_holding = f"{oppo_hold_parts[0]} and {oppo_hold_parts[1]}"
            
            if opponent_last_room is None:
                lines.append(f"\nI don't know where {self.oppo_name} is.")
            elif opponent_last_room == self.current_room:
                lines.append(f"\nI also see {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {oppo_holding}.")
            else:
                lines.append(f"\nLast time I saw {self.oppo_name}, {self.oppo_pronoun} was in the {opponent_last_room} holding {oppo_holding}.")
        
        # Exploration summary
        lines.append(f"\nMy Exploration summary:")
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
        
        return shuffled_room

    def get_available_plans(self, message, opponent_last_room=None, current_action=None, current_step=0, successful_count=0): #grasp selection list
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        send a message: ""
        keep current action
        """
        available_plans = []
        if self.communication and not (current_action and current_action.startswith("send a message")):
            available_plans.append("send a message to my teammate")
        if self.holding_objects[0]['type'] is None or self.holding_objects[1]['type'] is None:
            # Prioritize objects in current room first, then others (maintaining recent discovery order)
            current_room_obj_ids = set()
            if self.current_room in self.obj_per_room:
                current_room_obj_ids = set(obj['id'] for obj in self.obj_per_room[self.current_room][0])
                current_room_obj_ids.update(obj['id'] for obj in self.obj_per_room[self.current_room][1])

            # Add target objects: current room first, then others
            for obj in self.object_list[0]:
                if obj['id'] in current_room_obj_ids:
                    available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']}) in current room")
            for obj in self.object_list[0]:
                if obj['id'] not in current_room_obj_ids:
                    available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")

            # Add containers: current room first, then others (only if successful < 5)
            if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1) and successful_count < 5 and current_step < 1700:
                for obj in self.object_list[1]:
                    if obj['id'] in current_room_obj_ids:
                        available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']}) in current room")
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
            elif holding_container and current_step <= 1000 and total_target_objects == 4:
                can_transport = True
        
        if can_transport:
            available_plans.append(f"transport objects I'm holding to the bed")
        
        none_room = []
        part_room = []
        oppo_room = []
        for room in self.rooms:
            # Skip current room, None values, and fully explored rooms (marked as 'all')
            if room == self.current_room or room is None or room == 'None' or (room in self.rooms_explored and self.rooms_explored[room] == 'all'):
                continue
            # Add exploration status to "go to room" action
            room_status = 'part' if (room in self.rooms_explored and self.rooms_explored[room] == 'part') else 'none'
            available_plans.append(f"go to {room}, explored {room_status}")
            if room == opponent_last_room:
                oppo_room.append(room)
            if(room in self.rooms_explored and self.rooms_explored[room] == 'part'):
                part_room.append(room)
            else:
                none_room.append(room)
        
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            available_plans.append(f"explore current room {self.current_room}")

        # Add 'keep current action' as the last option
        # If current_action is already in available_plans, remove it and add 'keep current action' instead
        if current_action is not None:
            # Clean the current_action to match format in available_plans
            cleaned_current = self._clean_action(current_action)
            
            # Check if any plan in available_plans matches the cleaned current action
            matching_plan = None
            for plan in available_plans:
                if self._clean_action(plan) == cleaned_current:
                    matching_plan = plan
                    break
            
            # Remove the matching plan if found
            if matching_plan:
                available_plans.remove(matching_plan)
            else:
                llm_dir = os.path.join(self.output_dir, 'LLM')
                os.makedirs(llm_dir, exist_ok=True)
                with open(os.path.join(llm_dir, f"token_usage_agent_{self.agent_id}.txt"), 'a') as f:
                    f.write(f"Warning1: {current_action}\n")
            
            available_plans.append(f"keep current action: {cleaned_current}")
        
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"
        
        # Create shuffled_room with priority order based on agent parity
        shuffled_room = self._create_prioritized_room_list(none_room, part_room, oppo_room)
        return plans, len(available_plans), available_plans, shuffled_room


    def run(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room, action_history, dialogue_history, opponent_grabbed_objects = None, opponent_last_room = None):
        info = {}
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        
        # Get only the most recent opponent's dialogue
        dialogue_history_desc = ""
        for msg in reversed(dialogue_history):
            if msg.startswith(f"{self.oppo_name}:"):
                dialogue_history_desc = msg
                break
        
        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        # Count only type 0 (target objects) in satisfied
        satisfied_count = len([obj for obj in satisfied if isinstance(obj, dict) and obj.get('type') == 0])
        prompt = prompt.replace('$COMPLETED_OBJECTS$', str(satisfied_count))
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
        message = None
        llm_dir = os.path.join(self.output_dir, 'LLM')
        os.makedirs(llm_dir, exist_ok=True)
        with open(os.path.join(llm_dir, f"token_usage_agent_{self.agent_id}.txt"), 'a') as f:
            f.write(f"Start: {current_step}\n")
        if self.communication:
            dialogue_text = dialogue_history_desc if dialogue_history_desc else f"No message from {self.oppo_name} yet."
            prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_text)
        
        # Extract current action from action_history
        current_action = action_history[-1].split(" at step ")[0] if len(action_history) > 0 else None

        available_plans, num, available_plans_list, shuffled_room = self.get_available_plans(message, opponent_last_room, current_action, current_step, satisfied_count) 
        # available_plans_list 형식 확인

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        # CoT instruction is now in prompt template
        normal_prompt = prompt
        chat_prompt = [{"role": "user", "content": prompt}]
        if self.debug:
            pass
        outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
        output = outputs[0]
        self.total_cost += usage
        # if self.debug:
        #     with open(f"LLM/token_usage_agent_{self.agent_id}.txt", 'a') as f:
        #         f.write(f"prompt: {prompt}\n")
        #         f.write(f"output: {output}\n")
        #         f.write(f"End: {current_step}\n")
        plan, flags = self.parse_answer(available_plans_list, output) # parsing part
        if self.debug:
            pass

        # If LLM chose "send a message", generate the message content via gen_prompt
        if plan is not None and plan.startswith("send a message") and self.generator_prompt_template is not None:
            gen_prompt = self.generator_prompt_template.replace('$GOAL$', self.goal_desc)
            gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
            gen_prompt = gen_prompt.replace('$COMPLETED_OBJECTS$', str(satisfied_count))
            gen_prompt = gen_prompt.replace('$ACTION_HISTORY$', action_history_desc)
            gen_prompt = gen_prompt.replace('$DIALOGUE_HISTORY$', dialogue_text if self.communication else '')
            gen_chat = [{"role": "user", "content": gen_prompt}]
            gen_outputs, gen_usage = self.generator(gen_chat if self.chat else gen_prompt, self.sampling_params)
            self.total_cost += gen_usage
            gen_message = gen_outputs[0].strip()
            # Extract quoted message if present
            quoted = re.search(r'"([^"]+)"', gen_message)
            if quoted:
                gen_message = '"' + quoted.group(1) + '"'
            plan = f"send a message: {gen_message}"
            info['prompt_comm'] = gen_prompt
            info['output_comm'] = gen_outputs
            info['usage_comm'] = gen_usage
            if self.debug:
                print(f"gen_message: {gen_message}")

        info.update({"num_available_actions": num,
                     "prompt_plan_stage_2": normal_prompt,
                     "output_plan_stage_2": output,
                     "parse_exception": flags,
                     "plan": plan,
                     "total_cost": self.total_cost})
        return plan, info

