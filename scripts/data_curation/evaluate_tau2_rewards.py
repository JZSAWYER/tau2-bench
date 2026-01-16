#!/usr/bin/env python3
"""
Evaluate Trajectory-Level Rewards for τ²-bench

Analyzes trajectory rewards from rollout data and generates:
1. Statistics JSON with reward distribution
2. Bar chart showing reward distribution (overall and per-model-tier)

This helps decide thresholds for reward conditioning tokens.

Usage:
    python evaluate_tau2_rewards.py \
        --input trajectories.json \
        --output reward_stats.json \
        --plot reward_distribution.png \
        --bins 10 \
        --split-models
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trajectories(input_path: Path) -> List[Dict]:
    """Load trajectories from JSON file."""
    logger.info(f"Loading trajectories from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} trajectories")
    return data


def extract_rewards(
    trajectories: List[Dict], 
    use_component_reward: bool = False
) -> Dict[str, List[float]]:
    """Extract rewards, grouped by model tier if available.
    
    Args:
        trajectories: List of trajectory dictionaries
        use_component_reward: If True, use component_reward (averaged) instead of final_reward
    """
    rewards_by_tier = {
        "all": [], 
        "expert": [], 
        "intermediate": [], 
        "weak": []
    }
    
    for traj in trajectories:
        if use_component_reward and "component_reward" in traj:
            reward = traj.get("component_reward", 0.0)
        else:
            reward = traj.get("final_reward", 0.0)
        rewards_by_tier["all"].append(reward)
        
        tier = traj.get("model_tier", "unknown")
        if tier in rewards_by_tier:
            rewards_by_tier[tier].append(reward)
    
    return rewards_by_tier


def compute_statistics(rewards: List[float]) -> Dict:
    """Compute reward statistics."""
    if not rewards:
        return {"count": 0}
    
    arr = np.array(rewards)
    return {
        "count": len(rewards),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentile_25": float(np.percentile(arr, 25)),
        "percentile_75": float(np.percentile(arr, 75)),
        # Useful for threshold decisions
        "percentile_33": float(np.percentile(arr, 33)),
        "percentile_67": float(np.percentile(arr, 67)),
        # Count by reward value (for binary rewards)
        "count_reward_0": int(np.sum(arr == 0)),
        "count_reward_1": int(np.sum(arr == 1)),
        "success_rate": float(np.mean(arr == 1)) if len(arr) > 0 else 0.0,
    }


def plot_reward_distribution(
    rewards_by_tier: Dict[str, List[float]],
    output_path: Path,
    num_bins: int = 10,
    split_models: bool = False
):
    """Generate bar chart showing reward distribution."""
    
    if split_models and any(rewards_by_tier.get(t) for t in ["expert", "intermediate", "weak"]):
        plot_stacked_bar_chart(rewards_by_tier, output_path, num_bins)
    else:
        plot_simple_bar_chart(rewards_by_tier["all"], output_path, num_bins)


def plot_simple_bar_chart(rewards: List[float], output_path: Path, num_bins: int):
    """Simple bar chart for all rewards."""
    plt.figure(figsize=(12, 8))
    
    bins = np.linspace(0, 1, num_bins + 1)
    counts, bin_edges, patches = plt.hist(
        rewards,
        bins=bins,
        weights=np.ones(len(rewards)) / len(rewards) * 100,
        edgecolor='black',
        alpha=0.7,
        color='steelblue'
    )
    
    # Add percentage labels
    for count, patch in zip(counts, patches):
        if count > 0:
            plt.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.5,
                f'{count:.1f}%',
                ha='center', va='bottom', fontsize=9
            )
    
    # Statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    plt.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_reward:.3f}')
    
    plt.xlabel('Trajectory Reward', fontsize=14)
    plt.ylabel('Percentage of Trajectories (%)', fontsize=14)
    plt.title(
        f'τ²-bench Trajectory Reward Distribution\n'
        f'Total: {len(rewards)} | Mean: {mean_reward:.3f} | Std: {std_reward:.3f}',
        fontsize=16
    )
    plt.legend(loc='upper left', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Bar chart saved to: {output_path}")


def plot_stacked_bar_chart(
    rewards_by_tier: Dict[str, List[float]],
    output_path: Path,
    num_bins: int
):
    """Stacked bar chart showing breakdown by model tier."""
    plt.figure(figsize=(14, 10))
    
    tier_colors = {
        "expert": "#2ecc71",       # Green
        "intermediate": "#3498db",  # Blue
        "weak": "#e74c3c"           # Red
    }
    tier_labels = {
        "expert": "Expert Model",
        "intermediate": "Intermediate Model",
        "weak": "Weak Model"
    }
    
    bins = np.linspace(0, 1, num_bins + 1)
    interval_width = 1.0 / num_bins
    x_positions = bins[:-1] + interval_width / 2
    
    # Calculate counts per tier
    tier_counts = {}
    tier_means = {}
    total_count = 0
    
    for tier_name in ["expert", "intermediate", "weak"]:
        rewards = rewards_by_tier.get(tier_name, [])
        if rewards:
            counts, _ = np.histogram(rewards, bins=bins)
            tier_counts[tier_name] = counts
            tier_means[tier_name] = np.mean(rewards)
            total_count += len(rewards)
        else:
            tier_counts[tier_name] = np.zeros(num_bins)
            tier_means[tier_name] = 0.0
    
    # Convert to percentages
    tier_percentages = {}
    for tier_name in ["expert", "intermediate", "weak"]:
        if total_count > 0:
            tier_percentages[tier_name] = tier_counts[tier_name] / total_count * 100
        else:
            tier_percentages[tier_name] = np.zeros(num_bins)
    
    # Plot stacked bars
    bottom = np.zeros(num_bins)
    for tier_name in ["expert", "intermediate", "weak"]:
        percentages = tier_percentages[tier_name]
        n_traj = len(rewards_by_tier.get(tier_name, []))
        mean_val = tier_means[tier_name]
        label = f"{tier_labels[tier_name]} (n={n_traj}, μ={mean_val:.3f})"
        
        plt.bar(
            x_positions,
            percentages,
            width=interval_width,
            bottom=bottom,
            color=tier_colors[tier_name],
            edgecolor='black',
            linewidth=0.5,
            label=label
        )
        bottom += percentages
    
    # Add total percentage labels on top
    for i, (x, total_pct) in enumerate(zip(x_positions, bottom)):
        if total_pct > 0:
            plt.text(
                x, total_pct + 0.5,
                f'{total_pct:.1f}%',
                ha='center', va='bottom', fontsize=8
            )
    
    # Labels
    plt.xlabel('Trajectory Reward', fontsize=14)
    plt.ylabel('Percentage of Total Trajectories (%)', fontsize=14)
    
    all_rewards = rewards_by_tier["all"]
    overall_mean = np.mean(all_rewards) if all_rewards else 0
    overall_std = np.std(all_rewards) if all_rewards else 0
    
    plt.title(
        f'τ²-bench Trajectory Reward Distribution by Model Tier\n'
        f'Total: {len(all_rewards)} | Overall Mean: {overall_mean:.3f} | Std: {overall_std:.3f}',
        fontsize=16
    )
    
    plt.xticks(bins, [f'{b:.1f}' for b in bins], fontsize=10)
    plt.xlim(0, 1)
    
    # Set y-axis limit
    max_height = np.max(bottom) if len(bottom) > 0 else 100
    plt.ylim(0, max_height * 1.15)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Stacked bar chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate τ²-bench trajectory rewards and visualize distribution"
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
        help="Output statistics JSON file"
    )
    parser.add_argument(
        "--plot", 
        type=str, 
        required=True, 
        help="Output bar chart PNG file"
    )
    parser.add_argument(
        "--bins", 
        type=int, 
        default=10, 
        help="Number of histogram bins (default: 10)"
    )
    parser.add_argument(
        "--split-models", 
        action="store_true", 
        help="Show per-model-tier breakdown in the chart"
    )
    parser.add_argument(
        "--use-component-reward",
        action="store_true",
        help="Use component_reward (averaged from DB, COMMUNICATE, etc.) instead of final_reward. "
             "This provides more granular rewards since final_reward is binary (0 or 1) "
             "while component_reward can be 0.5 when one component passes and another fails."
    )
    
    args = parser.parse_args()
    
    # Load data
    trajectories = load_trajectories(Path(args.input))
    
    # Extract rewards
    reward_type = "component_reward (averaged)" if args.use_component_reward else "final_reward (binary)"
    logger.info(f"Using reward type: {reward_type}")
    rewards_by_tier = extract_rewards(trajectories, args.use_component_reward)
    
    # Compute statistics
    stats = {
        "overall": compute_statistics(rewards_by_tier["all"]),
        "by_tier": {
            tier: compute_statistics(rewards_by_tier[tier])
            for tier in ["expert", "intermediate", "weak"]
            if rewards_by_tier[tier]
        }
    }
    
    # Save statistics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to: {output_path}")
    
    # Generate plot
    plot_reward_distribution(
        rewards_by_tier,
        Path(args.plot),
        num_bins=args.bins,
        split_models=args.split_models
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Reward Distribution Summary")
    logger.info("=" * 60)
    logger.info(f"Total trajectories: {stats['overall']['count']}")
    logger.info(f"Mean reward: {stats['overall']['mean']:.4f}")
    logger.info(f"Std reward: {stats['overall']['std']:.4f}")
    logger.info(f"Median reward: {stats['overall']['median']:.4f}")
    logger.info(f"Min reward: {stats['overall']['min']:.4f}")
    logger.info(f"Max reward: {stats['overall']['max']:.4f}")
    logger.info(f"Success rate (reward=1): {stats['overall']['success_rate']:.2%}")
    logger.info("-" * 60)
    logger.info("Per-model-tier statistics:")
    for tier in ["expert", "intermediate", "weak"]:
        if tier in stats["by_tier"]:
            tier_stats = stats["by_tier"][tier]
            logger.info(f"  {tier.capitalize():12s}: n={tier_stats['count']}, "
                       f"mean={tier_stats['mean']:.3f}, "
                       f"success_rate={tier_stats['success_rate']:.1%}")
    logger.info("-" * 60)
    logger.info("Suggested thresholds based on percentiles:")
    logger.info(f"  33rd percentile: {stats['overall']['percentile_33']:.3f}")
    logger.info(f"  67th percentile: {stats['overall']['percentile_67']:.3f}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
