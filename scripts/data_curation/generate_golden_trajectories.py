#!/usr/bin/env python3
"""
Generate TRUE Golden Trajectories from τ²-bench

This script replays the golden actions through the environment to capture:
1. The actual tool call responses (observations)
2. The resulting database states

This creates TRUE expert trajectories where every action is correct.

Usage:
    python generate_golden_trajectories.py \
        --domain retail \
        --output data/tau2/sft/retail_golden.json \
        --split train
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tau2.registry import registry
from tau2.run import load_tasks
from tau2.data_model.tasks import Task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tasks_by_split(domain: str, split: str = "train") -> List[Task]:
    """Load tasks filtered by split."""
    # Use tau2.run.load_tasks which handles splits properly
    tasks = load_tasks(domain, split)
    logger.info(f"Loaded {len(tasks)} tasks from {domain}/{split}")
    return tasks


def replay_golden_actions(domain: str, task: Task) -> Optional[Dict]:
    """
    Replay golden actions through the environment to capture observations.
    
    Returns a trajectory with the actual tool responses.
    """
    try:
        # Get environment constructor
        env_constructor = registry.get_env_constructor(domain)
        env = env_constructor()
        
        # Set initial state if exists
        if task.initial_state:
            env.set_state(
                initialization_data=task.initial_state.initialization_data,
                initialization_actions=task.initial_state.initialization_actions,
                message_history=task.initial_state.message_history or []
            )
        
        # Get golden actions
        if not task.evaluation_criteria or not task.evaluation_criteria.actions:
            logger.warning(f"Task {task.id} has no golden actions")
            return None
        
        golden_actions = task.evaluation_criteria.actions
        
        # Build trajectory by replaying actions
        conversations = []
        
        # Skip system message - not needed for SFT data
        
        # Add initial user message from task
        user_instructions = task.user_scenario.instructions if task.user_scenario else {}
        user_request = user_instructions.get('reason_for_call', '') if isinstance(user_instructions, dict) else str(user_instructions)
        
        if user_request:
            conversations.append({
                "from": "human",
                "value": user_request
            })
        
        # Replay each golden action
        for action in golden_actions:
            # Add function call
            conversations.append({
                "from": "function_call",
                "value": json.dumps({
                    "name": action.name,
                    "arguments": action.arguments or {}
                })
            })
            
            # Execute action and capture observation
            try:
                requestor = action.requestor if hasattr(action, 'requestor') and action.requestor else 'assistant'
                result = env.make_tool_call(
                    tool_name=action.name,
                    requestor=requestor,
                    **action.arguments
                )
                observation = str(result) if result else "Success"
            except Exception as e:
                observation = f"Error: {str(e)}"
            
            # Add observation
            conversations.append({
                "from": "observation",
                "value": observation
            })
        
        # Get tools from environment
        tools_list = []
        try:
            tool_defs = env.get_tools() if hasattr(env, 'get_tools') else []
            for t in tool_defs:
                tool_dict = {
                    "name": t.name,
                    "description": t.short_desc or t.long_desc or "",
                }
                # Get parameter schema from Pydantic model class
                if t.params and hasattr(t.params, 'model_json_schema'):
                    tool_dict["parameters"] = t.params.model_json_schema()
                else:
                    tool_dict["parameters"] = {}
                tools_list.append(tool_dict)
        except Exception as e:
            logger.warning(f"Could not get tools: {e}")
        
        return {
            "conversations": conversations,
            "tools": json.dumps(tools_list),
            "final_reward": 1.0,  # Golden trajectory = perfect
            "component_reward": 1.0,
            "domain": domain,
            "task_id": str(task.id),
            "is_golden": True,
            "golden_actions_count": len(golden_actions)
        }
        
    except Exception as e:
        logger.error(f"Error replaying task {task.id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate TRUE golden trajectories by replaying golden actions"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to process (retail, airline, telecom)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Task split to use (train, test, base). Use 'both' to generate train and test separately."
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to process"
    )
    
    args = parser.parse_args()
    
    # Load tasks
    tasks = load_tasks_by_split(args.domain, args.split)
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    logger.info(f"Processing {len(tasks)} tasks from {args.domain} domain")
    
    # Generate golden trajectories
    trajectories = []
    success_count = 0
    
    for task in tasks:
        traj = replay_golden_actions(args.domain, task)
        if traj:
            trajectories.append(traj)
            success_count += 1
        
        if len(trajectories) % 10 == 0:
            logger.info(f"Processed {len(trajectories)} trajectories...")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generated {len(trajectories)} golden trajectories")
    logger.info(f"Success rate: {success_count}/{len(tasks)} ({success_count/len(tasks)*100:.1f}%)")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
