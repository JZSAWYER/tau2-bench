#!/usr/bin/env python3
"""
Build τ²-bench Trajectories Dataset

Generates diverse trajectories using multiple model tiers (expert, intermediate, weak)
on τ²-bench benchmark data using the AgentGymEnv gymnasium environment.

For each task in the TRAINING split, generates trajectories by:
1. Loading tasks from registry and filtering by train split
2. Creating AgentGymEnv with LLM user simulator
3. Generating agent actions using different model tiers
4. Capturing full conversation history and final rewards
5. Outputting records in ShareGPT (Glaive) format

Usage:
    python build_tau2_trajectories.py \
        --domain retail \
        --expert-model-path /path/to/qwen2.5-72b \
        --intermediate-model-path /path/to/qwen2.5-7b \
        --weak-model-path /path/to/qwen2.5-1.5b \
        --output-path data/tau2/sft/retail_trajectories.json \
        --user-llm gpt-4.1 \
        --num-gpus 4
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
from copy import deepcopy
from multiprocessing import Pool, Manager
from pathlib import Path
from threading import Thread, Event
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add tau2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import gymnasium as gym
from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
from tau2.registry import registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory for tau2
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tau2" / "domains"


class ModelGenerator:
    """
    Handles model loading and inference for HuggingFace models.
    
    Generates actions from observation prompts using chat templates.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        quantization: Optional[str] = None,
        max_new_tokens: int = 1024,
        use_auto_device_map: bool = True
    ):
        """
        Initialize model generator.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to use (cuda/cpu)
            quantization: Quantization mode (4bit/8bit/none)
            max_new_tokens: Maximum tokens to generate
            use_auto_device_map: Use automatic device mapping for multi-GPU
        """
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.use_auto_device_map = use_auto_device_map
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optional quantization
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info(f"Model loaded with 4-bit quantization")
            elif self.quantization == "8bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    load_in_8bit=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info(f"Model loaded with 8-bit quantization")
            else:
                if self.use_auto_device_map and self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    logger.info(f"Model loaded with device_map='auto'")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    if self.device == "cuda" or self.device.startswith("cuda:"):
                        self.model = self.model.to(self.device)
                    logger.info(f"Model loaded on device: {self.device}")
                    
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def generate(
        self, 
        observation: str, 
        tools: List[Any], 
        policy: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate action from observation using chat template.
        
        Args:
            observation: Current observation string from environment
            tools: List of available tools
            policy: Domain policy string
            system_prompt: Optional system prompt override
            
        Returns:
            Generated action string (tool call or message)
        """
        try:
            # Build messages for chat template
            messages = []
            
            # System message with policy and tools
            if system_prompt:
                system_content = system_prompt
            else:
                # Format tools for system prompt
                tools_desc = self._format_tools_for_prompt(tools)
                system_content = f"""You are a helpful customer service agent. Follow the policy guidelines below.

## Policy
{policy}

## Available Tools
{tools_desc}

When you need to use a tool, respond with a function call in this format:
tool_name(param1=value1, param2=value2)

When you want to send a message to the user, just write the message directly.
When the task is complete, call: done()
"""
            
            messages.append({"role": "system", "content": system_content})
            
            # Parse observation to extract conversation history
            if observation:
                # The observation contains the conversation history
                messages.append({"role": "user", "content": observation})
            
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                try:
                    # Try with tools parameter
                    tools_list = [self._tool_to_dict(t) for t in tools]
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tools=tools_list,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                except TypeError:
                    # Template doesn't support tools
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
            else:
                # Fallback
                prompt = "\n\n".join([m.get("content", "") for m in messages])
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=8192
            )
            
            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            else:
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def _format_tools_for_prompt(self, tools: List[Any]) -> str:
        """Format tools for inclusion in prompt."""
        tool_strings = []
        for tool in tools:
            tool_dict = self._tool_to_dict(tool)
            name = tool_dict.get("name", "unknown")
            desc = tool_dict.get("description", "")
            params = tool_dict.get("parameters", {}).get("properties", {})
            
            param_str = ", ".join([
                f"{k}: {v.get('type', 'any')}" 
                for k, v in params.items()
            ])
            tool_strings.append(f"- {name}({param_str}): {desc}")
        
        return "\n".join(tool_strings)
    
    def _tool_to_dict(self, tool: Any) -> Dict:
        """Convert tool object to dictionary."""
        if hasattr(tool, 'model_dump'):
            return tool.model_dump()
        elif isinstance(tool, dict):
            return tool
        else:
            return {"name": str(tool), "description": "", "parameters": {}}
    
    def parse_action(self, output: str) -> str:
        """
        Parse model output to extract action.
        
        Handles:
        - Function calls: func_name(arg1=val1, ...)
        - JSON tool calls: {"name": "...", "arguments": {...}}
        - Plain text messages
        
        Args:
            output: Raw model output
            
        Returns:
            Cleaned action string suitable for env.step()
        """
        output = output.strip()
        
        # Try to extract tool_call tags
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', output, re.DOTALL)
        if tool_call_match:
            try:
                parsed = json.loads(tool_call_match.group(1))
                if "name" in parsed:
                    return self._json_to_func_call(parsed)
            except json.JSONDecodeError:
                pass
        
        # Try to extract function-style calls
        func_match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)', output)
        if func_match:
            return func_match.group(0)
        
        # Try JSON format
        json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', output)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if "name" in parsed:
                    return self._json_to_func_call(parsed)
            except json.JSONDecodeError:
                pass
        
        # Return as plain text message
        return output
    
    def _json_to_func_call(self, json_obj: Dict) -> str:
        """Convert JSON tool call to function-style string."""
        name = json_obj.get("name", "")
        args = json_obj.get("arguments", {})
        
        if not args:
            return f"{name}()"
        
        arg_strs = []
        for k, v in args.items():
            if isinstance(v, str):
                arg_strs.append(f"{k}='{v}'")
            else:
                arg_strs.append(f"{k}={v}")
        
        return f"{name}({', '.join(arg_strs)})"


def load_train_task_ids(domain: str) -> set:
    """Load ONLY training task IDs from split_tasks.json."""
    split_path = DATA_DIR / domain / "split_tasks.json"
    
    if not split_path.exists():
        logger.warning(f"split_tasks.json not found for domain {domain}, using all tasks")
        return None
    
    with open(split_path) as f:
        splits = json.load(f)
    
    # Check for "train" key first, then fallback to "base" for mock domain
    if "train" in splits:
        train_ids = set(splits["train"])
        logger.info(f"Loaded {len(train_ids)} training task IDs for domain {domain}")
    elif "base" in splits:
        # Mock domain uses "base" instead of train/test split
        train_ids = set(splits["base"])
        logger.info(f"Loaded {len(train_ids)} base task IDs for domain {domain} (mock mode)")
    else:
        logger.warning(f"No 'train' or 'base' key in split_tasks.json for {domain}, using all tasks")
        return None
    
    return train_ids


def load_train_tasks(domain: str) -> List[Any]:
    """Load tasks filtered by train split."""
    all_tasks = registry.get_tasks_loader(domain)()
    train_ids = load_train_task_ids(domain)
    
    if train_ids is None:
        logger.warning(f"Using all {len(all_tasks)} tasks (no split file)")
        return all_tasks
    
    train_tasks = [t for t in all_tasks if t.id in train_ids]
    logger.info(f"Domain {domain}: {len(train_tasks)} train tasks (filtered from {len(all_tasks)} total)")
    return train_tasks


def serialize_tool(tool: Any) -> Dict:
    """Safely serialize a tool object to dictionary."""
    try:
        # Handle Pydantic models
        if hasattr(tool, 'model_dump'):
            result = tool.model_dump()
            # Recursively clean any non-serializable objects
            return json.loads(json.dumps(result, default=str))
        elif isinstance(tool, dict):
            # Clean dict of non-serializable values
            return json.loads(json.dumps(tool, default=str))
        elif hasattr(tool, '__dict__'):
            d = {k: v for k, v in tool.__dict__.items() if not k.startswith('_')}
            return json.loads(json.dumps(d, default=str))
        elif hasattr(tool, '__name__'):
            # It's a class or function
            return {"name": tool.__name__, "description": ""}
        else:
            return {"name": str(tool), "description": ""}
    except Exception as e:
        logger.warning(f"Failed to serialize tool {tool}: {e}")
        return {"name": str(tool), "description": ""}


def convert_to_sharegpt(
    simulation_run: Dict, 
    tools: List, 
    final_reward: float,
    reward_breakdown: Optional[Dict] = None
) -> Dict:
    """Convert τ²-bench simulation to ShareGPT format.
    
    Args:
        simulation_run: The simulation run data
        tools: List of available tools
        final_reward: The final reward (product of components, typically 0 or 1)
        reward_breakdown: Optional dict with component rewards (DB, COMMUNICATE, etc.)
    """
    conversations = []
    
    messages = simulation_run.get("messages", [])
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        
        if role == "system":
            conversations.append({"from": "system", "value": content or ""})
        elif role == "user":
            conversations.append({"from": "human", "value": content or ""})
        elif role == "assistant":
            if tool_calls:
                for tc in tool_calls:
                    tc_dict = tc if isinstance(tc, dict) else tc.model_dump() if hasattr(tc, 'model_dump') else {"name": str(tc)}
                    conversations.append({
                        "from": "function_call",
                        "value": json.dumps({
                            "name": tc_dict.get("name"),
                            "arguments": tc_dict.get("arguments", {})
                        })
                    })
            if content:
                conversations.append({"from": "gpt", "value": content})
        elif role == "tool":
            conversations.append({"from": "observation", "value": content or ""})
    
    # Format tools - handle various types safely
    tools_list = []
    for t in tools:
        tool_dict = serialize_tool(t)
        tools_list.append(tool_dict)
    
    # Compute component-based reward (average instead of product)
    # This gives more granular reward signal for RC-GRPO
    component_reward = final_reward
    if reward_breakdown:
        components = [v for v in reward_breakdown.values() if isinstance(v, (int, float))]
        if components:
            component_reward = sum(components) / len(components)
    
    return {
        "conversations": conversations,
        "tools": json.dumps(tools_list),
        "final_reward": final_reward,  # Original binary (0/1) reward
        "component_reward": component_reward,  # Averaged component reward (more granular)
        "reward_breakdown": reward_breakdown,  # Raw component scores
        "domain": simulation_run.get("domain"),
        "task_id": simulation_run.get("task_id"),
    }


def build_trajectory(
    task: Any,
    model_generator: ModelGenerator,
    domain: str,
    user_llm: str,
    user_llm_args: Dict,
    max_steps: int = 50,
    solo_mode: bool = False
) -> Optional[Dict]:
    """
    Generate single trajectory using gym environment.
    
    Args:
        task: Task object
        model_generator: ModelGenerator instance
        domain: Domain name
        user_llm: LLM for user simulator
        user_llm_args: Arguments for user simulator LLM
        max_steps: Maximum steps per episode
        solo_mode: If True, use DummyUser (no LLM required for user)
        
    Returns:
        ShareGPT format trajectory dict or None if failed
    """
    try:
        register_gym_agent()
        env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=domain,
            task_id=task.id,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            max_steps=max_steps,
            all_messages_as_observation=True,
            solo_mode=solo_mode,
        )
        
        obs, info = env.reset()
        tools = info.get("tools", [])
        policy = info.get("policy", "")
        
        final_reward = 0.0
        step_count = 0
        
        while step_count < max_steps:
            # Generate action from model
            raw_output = model_generator.generate(obs, tools, policy)
            action = model_generator.parse_action(raw_output)
            
            if not action:
                action = "I apologize, but I'm having trouble understanding. Could you please clarify?"
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            final_reward = reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Get simulation run
        simulation_run_str = info.get("simulation_run", "{}")
        if isinstance(simulation_run_str, str):
            simulation_run = json.loads(simulation_run_str)
        else:
            simulation_run = simulation_run_str
        
        # Extract reward breakdown from reward_info
        reward_breakdown = None
        reward_info_str = info.get("reward_info", "{}")
        if reward_info_str:
            try:
                reward_info = json.loads(reward_info_str) if isinstance(reward_info_str, str) else reward_info_str
                reward_breakdown = reward_info.get("reward_breakdown", {})
                # Convert enum keys to strings if needed
                if reward_breakdown:
                    reward_breakdown = {str(k): v for k, v in reward_breakdown.items()}
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Add domain and task_id if not present
        simulation_run["domain"] = domain
        simulation_run["task_id"] = task.id
        
        return convert_to_sharegpt(simulation_run, tools, final_reward, reward_breakdown)
        
    except Exception as e:
        logger.error(f"Error building trajectory for task {task.id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_shard(args_tuple):
    """
    Worker function to process a shard of tasks on a specific GPU.
    
    Args:
        args_tuple: Tuple containing worker configuration
        
    Returns:
        Tuple of (trajectories_list, success_count, failed_count)
    """
    (gpu_id, model_config, tasks_shard, model_tier, domain, 
     user_llm, user_llm_args, trajectories_per_task, max_steps,
     progress_counter, use_progress_bar, solo_mode) = args_tuple
    
    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch as torch_worker
    
    trajectories = []
    success_count = 0
    failed_count = 0
    
    worker_logger = logging.getLogger(f"worker_{gpu_id}")
    
    # Load model
    try:
        worker_logger.info(f"[GPU {gpu_id}] Loading {model_tier} model...")
        generator = ModelGenerator(
            model_path=model_config["path"],
            device="cuda:0",
            quantization=model_config["quantization"],
            max_new_tokens=model_config["max_new_tokens"],
            use_auto_device_map=False
        )
        worker_logger.info(f"[GPU {gpu_id}] Model loaded!")
    except Exception as e:
        worker_logger.error(f"[GPU {gpu_id}] Failed to load model: {e}")
        total_failed = len(tasks_shard) * trajectories_per_task
        return ([], 0, total_failed)
    
    # Process tasks
    for task in tasks_shard:
        for traj_idx in range(trajectories_per_task):
            try:
                trajectory = build_trajectory(
                    task=task,
                    model_generator=generator,
                    domain=domain,
                    user_llm=user_llm,
                    user_llm_args=user_llm_args,
                    max_steps=max_steps,
                    solo_mode=solo_mode
                )
                
                if trajectory:
                    trajectory["model_tier"] = model_tier
                    trajectories.append(trajectory)
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                if not use_progress_bar:
                    worker_logger.error(f"[GPU {gpu_id}] Error: {e}")
            
            if progress_counter is not None:
                progress_counter.value += 1
    
    # Cleanup
    del generator
    if torch_worker.cuda.is_available():
        torch_worker.cuda.empty_cache()
    
    return (trajectories, success_count, failed_count)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate τ²-bench trajectories using multiple model tiers"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["retail", "airline", "telecom", "mock", "all"],
        help="Domain to process (retail, airline, telecom, mock, or all)"
    )
    parser.add_argument(
        "--expert-model-path",
        type=str,
        required=True,
        help="HuggingFace model path for expert tier"
    )
    parser.add_argument(
        "--intermediate-model-path",
        type=str,
        required=True,
        help="HuggingFace model path for intermediate tier"
    )
    parser.add_argument(
        "--weak-model-path",
        type=str,
        required=True,
        help="HuggingFace model path for weak tier"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for trajectory dataset JSON file"
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default="gpt-4.1",
        help="LLM for user simulator (default: gpt-4.1)"
    )
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=0.7,
        help="Temperature for user simulator (default: 0.7)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for data parallelism (default: 1)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Quantization mode (default: none)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per model call (default: 1024)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50)"
    )
    parser.add_argument(
        "--trajectories-per-task",
        type=int,
        default=1,
        help="Number of trajectories per task per model (default: 1)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit number of tasks (0 = all training tasks)"
    )
    parser.add_argument(
        "--solo-mode",
        action="store_true",
        help="Use solo mode (DummyUser, no LLM required for user simulator). "
             "Useful for testing without API keys."
    )
    
    args = parser.parse_args()
    
    # Determine domains to process
    if args.domain == "all":
        domains = ["retail", "airline", "telecom"]
    else:
        domains = [args.domain]
    
    # Model configurations
    model_configs = {
        "expert": {
            "path": args.expert_model_path,
            "quantization": args.quantization if args.quantization != "none" else None,
            "max_new_tokens": args.max_new_tokens,
        },
        "intermediate": {
            "path": args.intermediate_model_path,
            "quantization": args.quantization if args.quantization != "none" else None,
            "max_new_tokens": args.max_new_tokens,
        },
        "weak": {
            "path": args.weak_model_path,
            "quantization": args.quantization if args.quantization != "none" else None,
            "max_new_tokens": args.max_new_tokens,
        }
    }
    
    user_llm_args = {"temperature": args.user_temperature}
    model_tiers = ["expert", "intermediate", "weak"]
    
    # Multi-GPU setup
    if args.num_gpus > 1:
        logger.info(f"Multi-GPU mode: {args.num_gpus} GPUs")
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    all_trajectories = []
    stats = {tier: {"success": 0, "failed": 0} for tier in model_tiers}
    
    for domain in domains:
        logger.info("=" * 60)
        logger.info(f"Processing domain: {domain}")
        logger.info("=" * 60)
        
        # Load training tasks
        tasks = load_train_tasks(domain)
        
        if args.max_tasks > 0:
            tasks = tasks[:args.max_tasks]
            logger.info(f"Limited to {len(tasks)} tasks")
        
        if not tasks:
            logger.warning(f"No tasks found for domain {domain}")
            continue
        
        # Process each model tier
        for model_tier in model_tiers:
            logger.info("-" * 40)
            logger.info(f"Processing with {model_tier.upper()} model")
            logger.info("-" * 40)
            
            config = model_configs[model_tier]
            total_iterations = len(tasks) * args.trajectories_per_task
            
            if args.num_gpus > 1:
                # Multi-GPU: distribute tasks
                shard_size = len(tasks) // args.num_gpus
                remainder = len(tasks) % args.num_gpus
                
                shards = []
                start_idx = 0
                for gpu_id in range(args.num_gpus):
                    end_idx = start_idx + shard_size + (1 if gpu_id < remainder else 0)
                    shards.append(tasks[start_idx:end_idx])
                    start_idx = end_idx
                
                # Progress tracking
                manager = Manager()
                progress_counter = manager.Value('i', 0)
                stop_event = Event()
                
                pbar = tqdm(total=total_iterations, desc=f"[{domain}/{model_tier}]", unit="traj")
                
                def update_progress(pbar, counter, total, stop_event):
                    import time
                    while not stop_event.is_set() and counter.value < total:
                        pbar.n = counter.value
                        pbar.refresh()
                        time.sleep(0.5)
                    pbar.n = min(counter.value, total)
                    pbar.refresh()
                
                progress_thread = Thread(
                    target=update_progress, 
                    args=(pbar, progress_counter, total_iterations, stop_event)
                )
                progress_thread.daemon = True
                progress_thread.start()
                
                # Prepare worker args
                worker_args = [
                    (
                        gpu_id, config, list(shard), model_tier, domain,
                        args.user_llm, user_llm_args, args.trajectories_per_task,
                        args.max_steps, progress_counter, True, args.solo_mode
                    )
                    for gpu_id, shard in enumerate(shards)
                ]
                
                # Process in parallel
                with Pool(processes=args.num_gpus) as pool:
                    results = pool.map(process_shard, worker_args)
                
                stop_event.set()
                progress_thread.join(timeout=2.0)
                pbar.close()
                manager.shutdown()
                
                # Merge results
                for trajectories, success, failed in results:
                    all_trajectories.extend(trajectories)
                    stats[model_tier]["success"] += success
                    stats[model_tier]["failed"] += failed
                    
            else:
                # Single GPU
                try:
                    logger.info(f"Loading {model_tier} model...")
                    generator = ModelGenerator(
                        model_path=config["path"],
                        device="cuda",
                        quantization=config["quantization"],
                        max_new_tokens=config["max_new_tokens"],
                        use_auto_device_map=True
                    )
                    logger.info(f"Model loaded!")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    stats[model_tier]["failed"] = total_iterations
                    continue
                
                # Process tasks
                for task in tqdm(tasks, desc=f"[{domain}/{model_tier}]"):
                    for traj_idx in range(args.trajectories_per_task):
                        try:
                            trajectory = build_trajectory(
                                task=task,
                                model_generator=generator,
                                domain=domain,
                                user_llm=args.user_llm,
                                user_llm_args=user_llm_args,
                                max_steps=args.max_steps,
                                solo_mode=args.solo_mode
                            )
                            
                            if trajectory:
                                trajectory["model_tier"] = model_tier
                                all_trajectories.append(trajectory)
                                stats[model_tier]["success"] += 1
                            else:
                                stats[model_tier]["failed"] += 1
                                
                        except Exception as e:
                            stats[model_tier]["failed"] += 1
                            logger.error(f"Error processing task {task.id}: {e}")
                
                # Cleanup
                del generator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Save output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Trajectory Generation Complete!")
    logger.info("=" * 60)
    logger.info(f"Domains: {', '.join(domains)}")
    for tier in model_tiers:
        logger.info(f"{tier.capitalize()}: {stats[tier]['success']} success, {stats[tier]['failed']} failed")
    logger.info(f"Total trajectories: {len(all_trajectories)}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
