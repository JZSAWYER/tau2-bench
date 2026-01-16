#!/usr/bin/env python3
"""
Build Reward-Conditioned Data for τ²-bench World Model Training

Adds reward conditioning tokens to trajectories based on configurable thresholds.
The thresholds should be determined by analyzing the reward distribution from
evaluate_tau2_rewards.py.

Token placement follows RC-GRPO paper:
- Append "[Reward Goal: <|xxx_reward|>]" to the first human message

Usage:
    python build_tau2_reward_conditioned.py \
        --input trajectories.json \
        --output reward_conditioned.json \
        --low-threshold 0.33 \
        --high-threshold 0.67
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reward conditioning tokens
LOW_REWARD_TOKEN = "<|low_reward|>"
MID_REWARD_TOKEN = "<|mid_reward|>"
HIGH_REWARD_TOKEN = "<|high_reward|>"


def get_reward_token(
    reward: float, 
    low_thresh: float, 
    high_thresh: float,
    model_tier: str = None,
    tier_based: bool = False
) -> str:
    """
    Determine reward token based on thresholds or model tier.
    
    Args:
        reward: Trajectory reward value
        low_thresh: Threshold below which reward is 'low'
        high_thresh: Threshold at/above which reward is 'high'
        model_tier: Model tier (expert/intermediate/weak)
        tier_based: If True, use model tier instead of reward value
        
    Returns:
        Reward conditioning token string
    """
    if tier_based and model_tier:
        # Use model tier to assign reward tokens
        # This is useful when rewards are binary (0 or 1)
        tier_map = {
            "expert": HIGH_REWARD_TOKEN,
            "intermediate": MID_REWARD_TOKEN,
            "weak": LOW_REWARD_TOKEN
        }
        return tier_map.get(model_tier.lower(), MID_REWARD_TOKEN)
    
    # Use reward value thresholds
    if reward < low_thresh:
        return LOW_REWARD_TOKEN
    elif reward >= high_thresh:
        return HIGH_REWARD_TOKEN
    else:
        return MID_REWARD_TOKEN


def add_reward_conditioning(
    trajectory: Dict,
    low_thresh: float,
    high_thresh: float,
    token_format: str = "goal",
    tier_based: bool = False,
    use_component_reward: bool = False
) -> Dict:
    """
    Add reward conditioning token to trajectory.
    
    Args:
        trajectory: Trajectory dictionary with conversations
        low_thresh: Low reward threshold
        high_thresh: High reward threshold
        token_format: Token format style ('goal' or 'prefix')
        tier_based: If True, use model tier instead of reward value
        use_component_reward: If True, use averaged component_reward instead of final_reward
                              This gives more granular signals (0.5 instead of binary 0/1)
        
    Returns:
        Modified trajectory with reward token appended
    """
    # Deep copy to avoid modifying original
    trajectory = json.loads(json.dumps(trajectory))
    conversations = trajectory.get("conversations", [])
    
    if not conversations:
        return trajectory
    
    # Get reward - prefer component_reward for more granular conditioning
    if use_component_reward and "component_reward" in trajectory:
        reward = trajectory.get("component_reward", 0.0)
    else:
        reward = trajectory.get("final_reward", 0.0)
    model_tier = trajectory.get("model_tier", None)
    reward_token = get_reward_token(reward, low_thresh, high_thresh, model_tier, tier_based)
    
    # Find first human message and append token
    for i, msg in enumerate(conversations):
        if msg.get("from") == "human":
            original_value = msg.get("value", "")
            
            if token_format == "goal":
                # RC-GRPO style: [Reward Goal: <|xxx_reward|>]
                new_value = f"{original_value} [Reward Goal: {reward_token}]"
            elif token_format == "prefix":
                # Prefix style: <|xxx_reward|> at the beginning
                new_value = f"{reward_token} {original_value}"
            elif token_format == "suffix":
                # Suffix style: <|xxx_reward|> at the end
                new_value = f"{original_value} {reward_token}"
            else:
                # Default to goal format
                new_value = f"{original_value} [Reward Goal: {reward_token}]"
            
            conversations[i] = {"from": "human", "value": new_value}
            break
    
    trajectory["conversations"] = conversations
    trajectory["reward_token"] = reward_token
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Build reward-conditioned data for τ²-bench World Model training"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Input trajectories JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Output reward-conditioned JSON file"
    )
    parser.add_argument(
        "--low-threshold", 
        type=float, 
        default=0.33,
        help="Threshold below which reward is 'low' (default: 0.33)"
    )
    parser.add_argument(
        "--high-threshold", 
        type=float, 
        default=0.67,
        help="Threshold at/above which reward is 'high' (default: 0.67)"
    )
    parser.add_argument(
        "--token-format", 
        type=str, 
        choices=["goal", "prefix", "suffix"], 
        default="goal",
        help="Token format: 'goal' for [Reward Goal: <|xxx|>], "
             "'prefix' for <|xxx|> prefix, 'suffix' for <|xxx|> suffix (default: goal)"
    )
    parser.add_argument(
        "--tier-based",
        action="store_true",
        help="Use model tier to assign reward tokens instead of reward value. "
             "Useful for binary rewards (0/1). Maps: expert→high, intermediate→mid, weak→low"
    )
    parser.add_argument(
        "--use-component-reward",
        action="store_true",
        help="Use component_reward (averaged from DB, COMMUNICATE, etc.) instead of final_reward. "
             "This provides more granular reward signals since final_reward is binary (0 or 1) "
             "while component_reward can be 0.5 when one component passes and another fails."
    )
    
    args = parser.parse_args()
    
    # Validate thresholds
    if args.low_threshold >= args.high_threshold:
        logger.error("low-threshold must be less than high-threshold")
        return 1
    
    if args.low_threshold < 0 or args.high_threshold > 1:
        logger.warning("Thresholds should typically be between 0 and 1")
    
    logger.info(f"Using thresholds: low < {args.low_threshold}, high >= {args.high_threshold}")
    logger.info(f"Token format: {args.token_format}")
    
    # Load trajectories
    input_path = Path(args.input)
    logger.info(f"Loading trajectories from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Add reward conditioning
    conditioned = []
    token_counts = {
        LOW_REWARD_TOKEN: 0, 
        MID_REWARD_TOKEN: 0, 
        HIGH_REWARD_TOKEN: 0
    }
    
    for traj in trajectories:
        conditioned_traj = add_reward_conditioning(
            traj,
            args.low_threshold,
            args.high_threshold,
            args.token_format,
            args.tier_based,
            args.use_component_reward
        )
        conditioned.append(conditioned_traj)
        
        token = conditioned_traj.get("reward_token", MID_REWARD_TOKEN)
        token_counts[token] += 1
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conditioned, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(conditioned)} conditioned trajectories to: {output_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Reward Conditioning Summary")
    logger.info("=" * 60)
    if args.tier_based:
        logger.info("Mode: TIER-BASED (using model tier, not reward value)")
        logger.info("  expert → <|high_reward|>")
        logger.info("  intermediate → <|mid_reward|>")
        logger.info("  weak → <|low_reward|>")
    else:
        reward_source = "component_reward (averaged)" if args.use_component_reward else "final_reward (binary)"
        logger.info(f"Mode: THRESHOLD-BASED (using {reward_source})")
        logger.info(f"Thresholds:")
        logger.info(f"  low:  reward < {args.low_threshold}")
        logger.info(f"  mid:  {args.low_threshold} <= reward < {args.high_threshold}")
        logger.info(f"  high: reward >= {args.high_threshold}")
        if args.use_component_reward:
            logger.info("Note: component_reward = avg(DB, COMMUNICATE, ACTION, etc.)")
            logger.info("  This gives values like 0.5 when 1 of 2 components pass")
    logger.info(f"Token format: {args.token_format}")
    logger.info("-" * 60)
    logger.info("Token distribution:")
    total = len(conditioned)
    for token, count in token_counts.items():
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {token}: {count} ({pct:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"Total trajectories: {total}")
    logger.info(f"Output file: {output_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
