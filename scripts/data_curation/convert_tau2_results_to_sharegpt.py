#!/usr/bin/env python3
"""
Convert τ²-bench Results to ShareGPT Format for World Model SFT

Converts the original τ²-bench benchmark results (expert trajectories from GPT-4.1,
Claude, etc.) to ShareGPT format compatible with LLaMA-Factory.

This provides high-quality expert trajectories for World Model Fine-Tuning (WMFT).

Usage:
    # Convert all GPT-4.1 results from all domains
    python convert_tau2_results_to_sharegpt.py \
        --results-dir data/tau2/results/final \
        --output data/tau2/sft/all_domains_expert.json \
        --agent-filter gpt-4.1

    # Convert specific domains
    python convert_tau2_results_to_sharegpt.py \
        --results-dir data/tau2/results/final \
        --output data/tau2/sft/retail_airline_expert.json \
        --domains retail airline

    # Filter by minimum reward
    python convert_tau2_results_to_sharegpt.py \
        --results-dir data/tau2/results/final \
        --output data/tau2/sft/high_reward_only.json \
        --min-reward 0.5
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_message_to_sharegpt(msg: Dict) -> List[Dict]:
    """
    Convert a τ²-bench message to ShareGPT format conversation turns.
    
    τ²-bench format:
        - role: assistant/user/tool
        - content: text content
        - tool_calls: list of tool calls
        
    ShareGPT format:
        - from: gpt/human/function_call/observation/system
        - value: content string
    """
    role = msg.get('role', '')
    content = msg.get('content', '')
    tool_calls = msg.get('tool_calls', [])
    
    turns = []
    
    if role == 'assistant':
        # Handle tool calls first
        if tool_calls:
            for tc in tool_calls:
                tc_data = {
                    "name": tc.get('name', tc.get('function', {}).get('name', '')),
                    "arguments": tc.get('arguments', tc.get('function', {}).get('arguments', {}))
                }
                # Parse arguments if string
                if isinstance(tc_data['arguments'], str):
                    try:
                        tc_data['arguments'] = json.loads(tc_data['arguments'])
                    except:
                        pass
                turns.append({
                    "from": "function_call",
                    "value": json.dumps(tc_data)
                })
        
        # Add text content if present
        if content:
            turns.append({
                "from": "gpt",
                "value": content
            })
            
    elif role == 'user':
        if content:
            turns.append({
                "from": "human",
                "value": content
            })
            
    elif role == 'tool':
        if content:
            turns.append({
                "from": "observation",
                "value": content
            })
            
    elif role == 'system':
        if content:
            turns.append({
                "from": "system",
                "value": content
            })
    
    return turns


def convert_simulation_to_sharegpt(
    simulation: Dict,
    domain: str,
    agent_model: str,
    tools: Optional[List] = None
) -> Optional[Dict]:
    """
    Convert a single τ²-bench simulation to ShareGPT format.
    """
    messages = simulation.get('messages', [])
    reward_info = simulation.get('reward_info', {})
    
    if not messages:
        return None
    
    # Convert messages
    conversations = []
    for msg in messages:
        turns = convert_message_to_sharegpt(msg)
        conversations.extend(turns)
    
    if not conversations:
        return None
    
    # Extract reward information
    final_reward = reward_info.get('reward', 0.0) if reward_info else 0.0
    reward_breakdown = reward_info.get('reward_breakdown', {}) if reward_info else {}
    
    # Convert reward breakdown keys to strings
    if reward_breakdown:
        reward_breakdown = {str(k): v for k, v in reward_breakdown.items()}
    
    # Compute component reward (average)
    component_reward = final_reward
    if reward_breakdown:
        components = [v for v in reward_breakdown.values() if isinstance(v, (int, float))]
        if components:
            component_reward = sum(components) / len(components)
    
    return {
        "conversations": conversations,
        "tools": json.dumps(tools) if tools else "[]",
        "final_reward": final_reward,
        "component_reward": component_reward,
        "reward_breakdown": reward_breakdown,
        "domain": domain,
        "task_id": simulation.get('task_id', ''),
        "agent_model": agent_model,
        "trial": simulation.get('trial', 0),
        "termination_reason": simulation.get('termination_reason', '')
    }


def load_result_file(filepath: Path) -> Optional[Dict]:
    """Load a τ²-bench result file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def extract_domain_from_filename(filename: str) -> str:
    """Extract domain name from result filename."""
    # Pattern: agent_domain_config_user_trials.json
    parts = filename.replace('.json', '').split('_')
    
    # Known domains
    domains = ['airline', 'retail', 'telecom', 'telecom-workflow', 'mock']
    
    for domain in domains:
        if domain in filename:
            return domain
    
    # Fallback: second part usually is domain
    if len(parts) >= 2:
        return parts[1]
    
    return 'unknown'


def extract_agent_from_filename(filename: str) -> str:
    """Extract agent model name from result filename."""
    # Pattern: agent_domain_...
    parts = filename.replace('.json', '').split('_')
    if parts:
        return parts[0]
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description="Convert τ²-bench results to ShareGPT format for World Model SFT"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/tau2/results/final",
        help="Directory containing τ²-bench result files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Filter by specific domains (e.g., retail airline telecom)"
    )
    parser.add_argument(
        "--agent-filter",
        type=str,
        default=None,
        help="Filter by agent model name (e.g., gpt-4.1, claude)"
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Minimum reward threshold for filtering trajectories"
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Maximum trajectories per task (to avoid duplicates from multiple trials)"
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only include successful trajectories (reward > 0)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    logger.info(f"Found {len(result_files)} result files")
    
    # Filter by agent if specified
    if args.agent_filter:
        result_files = [f for f in result_files if args.agent_filter in f.name]
        logger.info(f"After agent filter '{args.agent_filter}': {len(result_files)} files")
    
    # Filter by domain if specified
    if args.domains:
        filtered = []
        for f in result_files:
            domain = extract_domain_from_filename(f.name)
            if domain in args.domains:
                filtered.append(f)
        result_files = filtered
        logger.info(f"After domain filter {args.domains}: {len(result_files)} files")
    
    # Process all files
    all_trajectories = []
    stats = defaultdict(lambda: {"total": 0, "included": 0})
    task_counts = defaultdict(int)
    
    for filepath in result_files:
        logger.info(f"Processing: {filepath.name}")
        
        data = load_result_file(filepath)
        if not data:
            continue
        
        domain = extract_domain_from_filename(filepath.name)
        agent = extract_agent_from_filename(filepath.name)
        simulations = data.get('simulations', [])
        
        # Get tools from environment info
        tools = None
        info = data.get('info', {})
        env_info = info.get('environment_info', {})
        if 'tool_defs' in env_info and env_info['tool_defs']:
            tools = env_info['tool_defs']
        
        for sim in simulations:
            stats[domain]["total"] += 1
            task_id = sim.get('task_id', '')
            task_key = f"{domain}_{task_id}"
            
            # Check max per task
            if args.max_per_task and task_counts[task_key] >= args.max_per_task:
                continue
            
            # Convert to ShareGPT
            traj = convert_simulation_to_sharegpt(sim, domain, agent, tools)
            if not traj:
                continue
            
            # Filter by reward
            reward = traj.get('final_reward', 0)
            if args.min_reward is not None and reward < args.min_reward:
                continue
            if args.success_only and reward <= 0:
                continue
            
            all_trajectories.append(traj)
            stats[domain]["included"] += 1
            task_counts[task_key] += 1
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nSaved {len(all_trajectories)} trajectories to: {output_path}")
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("CONVERSION STATISTICS")
    logger.info("="*60)
    for domain, s in sorted(stats.items()):
        logger.info(f"  {domain:20s}: {s['included']:5d} / {s['total']:5d} included")
    logger.info("-"*60)
    logger.info(f"  {'TOTAL':20s}: {len(all_trajectories):5d}")
    
    # Reward distribution
    rewards = [t.get('final_reward', 0) for t in all_trajectories]
    if rewards:
        success = sum(1 for r in rewards if r > 0)
        logger.info(f"\nReward distribution:")
        logger.info(f"  Success (reward > 0): {success} ({success/len(rewards)*100:.1f}%)")
        logger.info(f"  Failure (reward = 0): {len(rewards) - success} ({(len(rewards)-success)/len(rewards)*100:.1f}%)")
    
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
