#!/usr/bin/env python3
"""
Build Expanded SFT and WMFT Datasets for τ²-bench

Converts ALL τ²-bench benchmark results (GPT-4.1, Claude, O4-mini) to OpenAI format
for Supervised Fine-Tuning (SFT) and World Model Fine-Tuning (WMFT).

KEY DESIGN: Both SFT and WM formats have IDENTICAL fields:
  - messages, tools, final_reward, reward_breakdown
  - domain, task_id, trial, termination_reason

The ONLY difference between SFT and WM is:
  - SFT: No reward token in messages (pure imitation learning)
  - WM:  [Reward Goal: <|high_reward|>] or <|low_reward|> injected in first user message

Features:
- Includes ALL trials (not just first successful one)
- Includes BOTH success (reward=1) AND failure (reward=0) trajectories
- Binary reward token mapping: 1.0 → <|high_reward|>, 0.0 → <|low_reward|>
- Task-based train/test split matching original τ²-bench
- OpenAI format with strict alternation fix for LLaMA-Factory

Usage:
    # Generate both SFT and WM formats (default)
    python build_wmft_expanded.py --output-dir data/tau2/sft
    
    # Generate only SFT format
    python build_wmft_expanded.py --output-dir data/tau2/sft --format sft
    
    # Generate only WM format
    python build_wmft_expanded.py --output-dir data/tau2/sft --format wm
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add tau2 to path for registry access
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tau2"
RESULTS_DIR = DATA_DIR / "results" / "final"
DOMAINS_DIR = DATA_DIR / "domains"


def load_train_test_splits() -> Dict[str, Dict[str, set]]:
    """Load train/test task ID splits for all domains."""
    splits = {}
    for domain in ['airline', 'retail', 'telecom']:
        split_path = DOMAINS_DIR / domain / "split_tasks.json"
        if split_path.exists():
            with open(split_path) as f:
                data = json.load(f)
            splits[domain] = {
                'train': set(data.get('train', [])),
                'test': set(data.get('test', []))
            }
            logger.info(f"Loaded splits for {domain}: train={len(splits[domain]['train'])}, test={len(splits[domain]['test'])}")
        else:
            logger.warning(f"No split file for {domain}")
            splits[domain] = {'train': set(), 'test': set()}
    return splits


def load_domain_policy(domain: str) -> str:
    """Load the policy document for a domain."""
    # Try different policy file names
    policy_names = ["policy.md", "main_policy.md"]
    for name in policy_names:
        policy_path = DOMAINS_DIR / domain / name
        if policy_path.exists():
            with open(policy_path, 'r') as f:
                return f.read()
    return ""


def load_domain_tools_from_results(results_dir: Path, domain: str) -> List[Dict]:
    """Load tool definitions from result files for a domain."""
    # Find a result file for this domain
    for filepath in results_dir.glob(f"*_{domain}_*.json"):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            info = data.get('info', {})
            env_info = info.get('environment_info', {})
            tool_defs = env_info.get('tool_defs', [])
            if tool_defs:
                # Convert to OpenAI function format if needed
                formatted_tools = []
                for tool in tool_defs:
                    if isinstance(tool, dict):
                        if 'type' in tool and tool['type'] == 'function':
                            formatted_tools.append(tool)
                        else:
                            formatted_tools.append({
                                "type": "function",
                                "function": {
                                    "name": tool.get("name", ""),
                                    "description": tool.get("description", ""),
                                    "parameters": tool.get("parameters", {})
                                }
                            })
                return formatted_tools
        except Exception as e:
            continue
    return []


def extract_domain_from_filename(filename: str) -> str:
    """Extract domain name from result filename."""
    for domain in ['airline', 'retail', 'telecom']:
        if domain in filename:
            return domain
    return 'unknown'


def get_reward_token(reward: float) -> str:
    """Map reward to binary token."""
    if reward >= 1.0:
        return "<|high_reward|>"
    else:
        return "<|low_reward|>"


def convert_message_to_openai(msg: Dict) -> List[Dict]:
    """Convert a τ²-bench message to OpenAI format messages."""
    role = msg.get('role', '')
    content = msg.get('content', '') or ''
    tool_calls = msg.get('tool_calls', [])
    
    messages = []
    
    if role == 'assistant':
        assistant_msg = {
            'role': 'assistant',
            'content': content,
            'tool_calls': []
        }
        
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                # Handle different tool_call formats
                if isinstance(tc, dict):
                    name = tc.get('name', tc.get('function', {}).get('name', ''))
                    args = tc.get('arguments', tc.get('function', {}).get('arguments', {}))
                else:
                    name = str(tc)
                    args = {}
                
                # Parse string arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {"raw": args}
                
                assistant_msg['tool_calls'].append({
                    'id': f"call_{i}",
                    'type': 'function',
                    'function': {
                        'name': name,
                        'arguments': json.dumps(args) if isinstance(args, dict) else str(args)
                    }
                })
        
        messages.append(assistant_msg)
        
    elif role == 'user':
        messages.append({
            'role': 'user',
            'content': content
        })
        
    elif role == 'tool':
        messages.append({
            'role': 'tool',
            'content': content,
            'tool_call_id': msg.get('tool_call_id', 'call_0')
        })
        
    elif role == 'system':
        messages.append({
            'role': 'system',
            'content': content
        })
    
    return messages


def get_effective_role(msg: Dict) -> str:
    """Get effective role for alternation checking."""
    role = msg['role']
    if role == 'assistant':
        if msg.get('tool_calls') and len(msg.get('tool_calls', [])) > 0:
            return 'function'
        return 'assistant'
    elif role == 'tool':
        return 'observation'
    return role


def fix_alternation(messages: List[Dict]) -> List[Dict]:
    """Fix conversation to match strict alternating pattern for LLaMA-Factory."""
    if not messages:
        return messages
    
    result = []
    
    # Extract system message
    if messages[0]['role'] == 'system':
        result.append(messages[0])
        messages = messages[1:]
    
    # First pass: merge consecutive assistant+tool sequences
    merged = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg['role']
        
        if role == 'assistant' and msg.get('tool_calls') and len(msg.get('tool_calls', [])) > 0:
            # Start of tool-calling sequence
            combined_assistant = {
                'role': 'assistant',
                'content': msg.get('content', '') or '',
                'tool_calls': list(msg.get('tool_calls', []))
            }
            combined_tools = []
            i += 1
            
            # Collect consecutive tool responses
            while i < len(messages) and messages[i]['role'] == 'tool':
                combined_tools.append(messages[i])
                i += 1
            
            # Check for more assistant(tool)+tool sequences
            while i < len(messages):
                next_msg = messages[i]
                if next_msg['role'] == 'assistant' and next_msg.get('tool_calls') and len(next_msg.get('tool_calls', [])) > 0:
                    if next_msg.get('content'):
                        combined_assistant['content'] += ('\n' if combined_assistant['content'] else '') + next_msg['content']
                    combined_assistant['tool_calls'].extend(next_msg.get('tool_calls', []))
                    i += 1
                    while i < len(messages) and messages[i]['role'] == 'tool':
                        combined_tools.append(messages[i])
                        i += 1
                else:
                    break
            
            merged.append(combined_assistant)
            if combined_tools:
                merged.append({
                    'role': 'tool',
                    'content': '\n---\n'.join([t.get('content', '') for t in combined_tools]),
                    'tool_call_id': combined_tools[0].get('tool_call_id', 'merged')
                })
        else:
            merged.append(msg)
            i += 1
    
    # Second pass: fix alternation by inserting dummy messages
    fixed = []
    for msg in merged:
        effective = get_effective_role(msg)
        
        if not fixed:
            if effective in ('user', 'observation'):
                fixed.append(msg)
            else:
                fixed.append({'role': 'user', 'content': '[Start conversation]'})
                fixed.append(msg)
        else:
            expected_odd = len(fixed) % 2 == 1
            
            if expected_odd:
                if effective in ('assistant', 'function'):
                    fixed.append(msg)
                else:
                    fixed.append({
                        'role': 'assistant',
                        'content': 'I understand. Let me help you with that.',
                        'tool_calls': []
                    })
                    fixed.append(msg)
            else:
                if effective in ('user', 'observation'):
                    fixed.append(msg)
                else:
                    fixed.append({'role': 'user', 'content': '[Continue]'})
                    fixed.append(msg)
    
    # Ensure even count
    if len(fixed) % 2 != 0:
        last_effective = get_effective_role(fixed[-1])
        if last_effective in ('user', 'observation'):
            fixed.append({
                'role': 'assistant',
                'content': 'Is there anything else I can help you with?',
                'tool_calls': []
            })
        else:
            fixed.append({'role': 'user', 'content': '[End of conversation]'})
            fixed.append({
                'role': 'assistant',
                'content': 'Thank you for contacting us. Have a great day!',
                'tool_calls': []
            })
    
    return result + fixed


def convert_simulation_to_openai(
    simulation: Dict,
    domain: str,
    policy: str,
    tools: List[Dict],
    add_reward_token: bool = True
) -> Optional[Dict]:
    """
    Convert a single τ²-bench simulation to OpenAI format.
    
    Args:
        simulation: Raw simulation data
        domain: Domain name
        policy: Policy document
        tools: Tool definitions
        add_reward_token: If True, inject reward token into messages (WM format)
                         If False, no reward token (SFT format)
    
    Returns:
        Dict with identical fields for both SFT and WM formats.
        The ONLY difference is whether reward token is in messages.
    """
    messages = simulation.get('messages', [])
    reward_info = simulation.get('reward_info', {})
    
    if not messages:
        return None
    
    # Extract reward
    final_reward = reward_info.get('reward', 0.0) if reward_info else 0.0
    reward_breakdown = reward_info.get('reward_breakdown', {}) if reward_info else {}
    
    # Build system prompt with policy
    system_content = f"""<instructions>
You are a customer service agent for a {domain} company. Follow the policy guidelines below.

{policy}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>"""
    
    # Convert messages to OpenAI format
    openai_messages = [{'role': 'system', 'content': system_content}]
    
    for msg in messages:
        converted = convert_message_to_openai(msg)
        openai_messages.extend(converted)
    
    # Fix alternation for LLaMA-Factory
    openai_messages = fix_alternation(openai_messages)
    
    if not openai_messages or len(openai_messages) < 2:
        return None
    
    # Add reward token to first user message (after system) - ONLY difference between SFT and WM
    if add_reward_token:
        reward_token = get_reward_token(final_reward)
        for i, msg in enumerate(openai_messages):
            if msg['role'] == 'user':
                msg['content'] = msg['content'] + f"\n\n[Reward Goal: {reward_token}]"
                break
    
    # Return IDENTICAL structure for both SFT and WM
    # The ONLY difference is whether reward token is injected into messages above
    return {
        'messages': openai_messages,
        'tools': tools,
        'final_reward': final_reward,
        'reward_breakdown': {str(k): v for k, v in reward_breakdown.items()} if reward_breakdown else {},
        'domain': domain,
        'task_id': simulation.get('task_id', ''),
        'trial': simulation.get('trial', 0),
        'termination_reason': simulation.get('termination_reason', '')
    }


def validate_conversation(messages: List[Dict]) -> Tuple[bool, str]:
    """Validate conversation matches LLaMA-Factory expected pattern."""
    if not messages:
        return False, "Empty"
    
    content = messages[1:] if messages[0]['role'] == 'system' else messages
    
    if len(content) % 2 != 0:
        return False, f"Odd count: {len(content)}"
    
    for idx, msg in enumerate(content):
        effective = get_effective_role(msg)
        
        if idx % 2 == 0:
            if effective not in ('user', 'observation'):
                return False, f"Idx {idx}: expected user/obs, got {effective}"
        else:
            if effective not in ('assistant', 'function'):
                return False, f"Idx {idx}: expected asst/func, got {effective}"
    
    return True, "OK"


def process_result_file(
    filepath: Path,
    splits: Dict[str, Dict[str, set]],
    policies: Dict[str, str],
    tools_cache: Dict[str, List[Dict]],
    add_reward_token: bool = True
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process a single result file and return train/test trajectories.
    
    Args:
        filepath: Path to result file
        splits: Train/test task ID splits
        policies: Domain policies
        tools_cache: Domain tool definitions
        add_reward_token: If True, generate WM format (with reward token)
                         If False, generate SFT format (no reward token)
    
    Returns:
        Tuple of (train_trajectories, test_trajectories, stats)
        Both formats have IDENTICAL fields - only messages content differs.
    """
    train_trajectories = []
    test_trajectories = []
    stats = {'total': 0, 'train': 0, 'test': 0, 'skipped': 0, 'invalid': 0}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return [], [], stats
    
    domain = extract_domain_from_filename(filepath.name)
    if domain == 'unknown':
        logger.warning(f"Unknown domain for {filepath.name}")
        return [], [], stats
    
    simulations = data.get('simulations', [])
    policy = policies.get(domain, '')
    tools = tools_cache.get(domain, [])
    domain_splits = splits.get(domain, {'train': set(), 'test': set()})
    
    for sim in simulations:
        stats['total'] += 1
        task_id = sim.get('task_id')
        
        # Determine train/test split
        if task_id in domain_splits['train']:
            split = 'train'
        elif task_id in domain_splits['test']:
            split = 'test'
        else:
            # Task not in either split - skip
            stats['skipped'] += 1
            continue
        
        # Convert to OpenAI format (SFT or WM based on add_reward_token)
        trajectory = convert_simulation_to_openai(
            sim, domain, policy, tools, add_reward_token=add_reward_token
        )
        
        if not trajectory:
            stats['invalid'] += 1
            continue
        
        # Validate
        is_valid, reason = validate_conversation(trajectory['messages'])
        if not is_valid:
            stats['invalid'] += 1
            continue
        
        if split == 'train':
            train_trajectories.append(trajectory)
            stats['train'] += 1
        else:
            test_trajectories.append(trajectory)
            stats['test'] += 1
    
    return train_trajectories, test_trajectories, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build expanded SFT and WMFT datasets from τ²-bench results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tau2/sft",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: data/tau2/results/final)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sft", "wm", "both"],
        default="both",
        help="Output format: sft (no reward token), wm (with reward token), or both"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train/test splits
    logger.info("Loading train/test splits...")
    splits = load_train_test_splits()
    
    # Load policies
    logger.info("Loading domain policies...")
    policies = {}
    for domain in ['airline', 'retail', 'telecom']:
        policies[domain] = load_domain_policy(domain)
        logger.info(f"  {domain}: {len(policies[domain])} chars")
    
    # Load tools from result files
    logger.info("Loading domain tools from result files...")
    tools_cache = {}
    for domain in ['airline', 'retail', 'telecom']:
        tools_cache[domain] = load_domain_tools_from_results(results_dir, domain)
        logger.info(f"  {domain}: {len(tools_cache[domain])} tools")
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    logger.info(f"Found {len(result_files)} result files")
    
    # Determine which formats to generate
    formats_to_generate = []
    if args.format in ["sft", "both"]:
        formats_to_generate.append(("sft", False))  # (name, add_reward_token)
    if args.format in ["wm", "both"]:
        formats_to_generate.append(("wm", True))
    
    for format_name, add_reward_token in formats_to_generate:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {format_name.upper()} format (reward_token={add_reward_token})")
        logger.info(f"{'='*60}")
        
        # Process all files for this format
        all_train = []
        all_test = []
        total_stats = defaultdict(int)
        domain_stats = defaultdict(lambda: defaultdict(int))
        reward_stats = {'train': defaultdict(int), 'test': defaultdict(int)}
        
        for filepath in result_files:
            logger.info(f"Processing: {filepath.name}")
            train, test, stats = process_result_file(
                filepath, splits, policies, tools_cache,
                add_reward_token=add_reward_token
            )
            
            all_train.extend(train)
            all_test.extend(test)
            
            domain = extract_domain_from_filename(filepath.name)
            for key, val in stats.items():
                total_stats[key] += val
                domain_stats[domain][key] += val
            
            # Track reward distribution
            for t in train:
                r = t.get('final_reward', 0)
                reward_stats['train']['success' if r >= 1.0 else 'failure'] += 1
            for t in test:
                r = t.get('final_reward', 0)
                reward_stats['test']['success' if r >= 1.0 else 'failure'] += 1
        
        # Save outputs
        train_path = output_dir / f"expanded_{format_name}_train.json"
        test_path = output_dir / f"expanded_{format_name}_test.json"
        
        logger.info(f"\nSaving {len(all_train)} train trajectories to {train_path}")
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(all_train, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saving {len(all_test)} test trajectories to {test_path}")
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(all_test, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        logger.info(f"\n{format_name.upper()} Statistics:")
        logger.info(f"  Processed: {total_stats['total']}")
        logger.info(f"  Train: {total_stats['train']}")
        logger.info(f"  Test: {total_stats['test']}")
        logger.info(f"  Skipped (not in split): {total_stats['skipped']}")
        logger.info(f"  Invalid: {total_stats['invalid']}")
        
        logger.info(f"\nBy Domain:")
        for domain in ['airline', 'retail', 'telecom']:
            ds = domain_stats[domain]
            logger.info(f"  {domain}: train={ds['train']}, test={ds['test']}, skipped={ds['skipped']}")
        
        logger.info(f"\nReward Distribution:")
        logger.info(f"  Train: {reward_stats['train']['success']} success, {reward_stats['train']['failure']} failure")
        logger.info(f"  Test: {reward_stats['test']['success']} success, {reward_stats['test']['failure']} failure")
    
    logger.info("\n" + "=" * 60)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBoth SFT and WM formats have IDENTICAL fields:")
    logger.info(f"  - messages, tools, final_reward, reward_breakdown")
    logger.info(f"  - domain, task_id, trial, termination_reason")
    logger.info(f"\nThe ONLY difference:")
    logger.info(f"  - SFT: No reward token in messages")
    logger.info(f"  - WM:  [Reward Goal: <|high_reward|>] or <|low_reward|> in first user message")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
