#!/bin/bash
#
# τ²-bench Data Curation Pipeline Runner
#
# This script runs the full data curation pipeline for τ²-bench:
# 1. Part 1: Generate trajectories using 3 model tiers
# 2. Part 2: Evaluate reward distribution and generate visualization
# 3. Part 3: Add reward conditioning tokens
#
# Usage:
#   ./run_tau2_pipeline.sh --domain retail --expert-model /path/to/72b \
#       --intermediate-model /path/to/7b --weak-model /path/to/1.5b
#
# Prerequisites:
#   - τ²-bench installed with gym environment
#   - HuggingFace models available
#   - API keys configured for user simulator (e.g., OPENAI_API_KEY)

set -e  # Exit on error

# ============================================================
# API Configuration (OpenAI-compatible proxy)
# ============================================================
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-6rEPQHwaNTcNnCPz9xodf6mlrqeEilicfZxASojea4JcWOCf}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://35.220.164.252:3888/v1}"

# Default values
DOMAIN="retail"
OUTPUT_DIR="data/tau2/sft"
USER_LLM="gpt-4.1"
USER_TEMPERATURE="0.7"
NUM_GPUS="1"
QUANTIZATION="none"
MAX_NEW_TOKENS="1024"
MAX_STEPS="50"
TRAJECTORIES_PER_TASK="1"
MAX_TASKS="0"
BINS="10"
LOW_THRESHOLD="0.33"
HIGH_THRESHOLD="0.67"
SKIP_ROLLOUT="false"
SKIP_EVAL="false"

# Model paths (must be provided)
EXPERT_MODEL=""
INTERMEDIATE_MODEL=""
WEAK_MODEL=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

τ²-bench Data Curation Pipeline

Required Arguments:
    --expert-model PATH         Path to expert model (e.g., Qwen2.5-72B)
    --intermediate-model PATH   Path to intermediate model (e.g., Qwen2.5-7B)
    --weak-model PATH          Path to weak model (e.g., Qwen2.5-1.5B)

Optional Arguments:
    --domain DOMAIN            Domain to process: retail, airline, telecom, all (default: retail)
    --output-dir DIR           Output directory (default: data/tau2/sft)
    --user-llm MODEL           LLM for user simulator (default: gpt-4.1)
    --num-gpus N               Number of GPUs for parallelism (default: 1)
    --quantization MODE        Quantization: none, 4bit, 8bit (default: none)
    --trajectories-per-task N  Trajectories per task per model (default: 1)
    --max-tasks N              Limit number of tasks, 0=all (default: 0)
    --bins N                   Number of histogram bins (default: 10)
    --low-threshold FLOAT      Low reward threshold (default: 0.33)
    --high-threshold FLOAT     High reward threshold (default: 0.67)
    --skip-rollout            Skip trajectory rollout (use existing data)
    --skip-eval               Skip reward evaluation (use existing stats)
    -h, --help                Show this help message

Examples:
    # Full pipeline for retail domain
    ./run_tau2_pipeline.sh \\
        --domain retail \\
        --expert-model /path/to/qwen2.5-72b \\
        --intermediate-model /path/to/qwen2.5-7b \\
        --weak-model /path/to/qwen2.5-1.5b \\
        --num-gpus 4

    # Only add reward tokens (skip rollout and eval)
    ./run_tau2_pipeline.sh \\
        --domain retail \\
        --skip-rollout --skip-eval \\
        --low-threshold 0.2 --high-threshold 0.8
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --expert-model)
            EXPERT_MODEL="$2"
            shift 2
            ;;
        --intermediate-model)
            INTERMEDIATE_MODEL="$2"
            shift 2
            ;;
        --weak-model)
            WEAK_MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --user-llm)
            USER_LLM="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --trajectories-per-task)
            TRAJECTORIES_PER_TASK="$2"
            shift 2
            ;;
        --max-tasks)
            MAX_TASKS="$2"
            shift 2
            ;;
        --bins)
            BINS="$2"
            shift 2
            ;;
        --low-threshold)
            LOW_THRESHOLD="$2"
            shift 2
            ;;
        --high-threshold)
            HIGH_THRESHOLD="$2"
            shift 2
            ;;
        --skip-rollout)
            SKIP_ROLLOUT="true"
            shift
            ;;
        --skip-eval)
            SKIP_EVAL="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments (only if not skipping rollout)
if [[ "$SKIP_ROLLOUT" == "false" ]]; then
    if [[ -z "$EXPERT_MODEL" || -z "$INTERMEDIATE_MODEL" || -z "$WEAK_MODEL" ]]; then
        echo "Error: Model paths are required unless --skip-rollout is specified"
        usage
        exit 1
    fi
fi

# Create output directory
mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR}"

# Output file paths
TRAJECTORIES_FILE="${OUTPUT_DIR}/${DOMAIN}_trajectories.json"
STATS_FILE="${OUTPUT_DIR}/${DOMAIN}_reward_stats.json"
PLOT_FILE="${OUTPUT_DIR}/${DOMAIN}_reward_distribution.png"
CONDITIONED_FILE="${OUTPUT_DIR}/${DOMAIN}_reward_conditioned.json"

echo "============================================================"
echo "τ²-bench Data Curation Pipeline"
echo "============================================================"
echo "Domain:         ${DOMAIN}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "User LLM:       ${USER_LLM}"
echo "GPUs:           ${NUM_GPUS}"
echo "Quantization:   ${QUANTIZATION}"
echo "Low threshold:  ${LOW_THRESHOLD}"
echo "High threshold: ${HIGH_THRESHOLD}"
echo "============================================================"

cd "${PROJECT_ROOT}"

# ============================================================
# Part 1: Trajectory Rollout
# ============================================================
if [[ "$SKIP_ROLLOUT" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "Part 1: Generating Trajectories"
    echo "============================================================"
    echo "Expert model:       ${EXPERT_MODEL}"
    echo "Intermediate model: ${INTERMEDIATE_MODEL}"
    echo "Weak model:         ${WEAK_MODEL}"
    echo ""
    
    python scripts/data_curation/build_tau2_trajectories.py \
        --domain "${DOMAIN}" \
        --expert-model-path "${EXPERT_MODEL}" \
        --intermediate-model-path "${INTERMEDIATE_MODEL}" \
        --weak-model-path "${WEAK_MODEL}" \
        --output-path "${TRAJECTORIES_FILE}" \
        --user-llm "${USER_LLM}" \
        --user-temperature "${USER_TEMPERATURE}" \
        --num-gpus "${NUM_GPUS}" \
        --quantization "${QUANTIZATION}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-steps "${MAX_STEPS}" \
        --trajectories-per-task "${TRAJECTORIES_PER_TASK}" \
        --max-tasks "${MAX_TASKS}"
    
    echo "Trajectories saved to: ${TRAJECTORIES_FILE}"
else
    echo ""
    echo "[Skipping Part 1: Trajectory Rollout]"
    if [[ ! -f "${PROJECT_ROOT}/${TRAJECTORIES_FILE}" ]]; then
        echo "Warning: Trajectory file not found: ${TRAJECTORIES_FILE}"
    fi
fi

# ============================================================
# Part 2: Reward Evaluation & Visualization
# ============================================================
if [[ "$SKIP_EVAL" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "Part 2: Evaluating Reward Distribution"
    echo "============================================================"
    
    python scripts/data_curation/evaluate_tau2_rewards.py \
        --input "${TRAJECTORIES_FILE}" \
        --output "${STATS_FILE}" \
        --plot "${PLOT_FILE}" \
        --bins "${BINS}" \
        --split-models
    
    echo ""
    echo "Statistics saved to: ${STATS_FILE}"
    echo "Distribution plot:   ${PLOT_FILE}"
    echo ""
    echo ">>> REVIEW THE PLOT AND ADJUST THRESHOLDS IF NEEDED <<<"
    echo "    Current thresholds: low < ${LOW_THRESHOLD}, high >= ${HIGH_THRESHOLD}"
else
    echo ""
    echo "[Skipping Part 2: Reward Evaluation]"
fi

# ============================================================
# Part 3: Reward Conditioning
# ============================================================
echo ""
echo "============================================================"
echo "Part 3: Adding Reward Conditioning Tokens"
echo "============================================================"
echo "Thresholds: low < ${LOW_THRESHOLD}, high >= ${HIGH_THRESHOLD}"

python scripts/data_curation/build_tau2_reward_conditioned.py \
    --input "${TRAJECTORIES_FILE}" \
    --output "${CONDITIONED_FILE}" \
    --low-threshold "${LOW_THRESHOLD}" \
    --high-threshold "${HIGH_THRESHOLD}" \
    --token-format "goal"

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Trajectories:      ${TRAJECTORIES_FILE}"
echo "  Statistics:        ${STATS_FILE}"
echo "  Distribution plot: ${PLOT_FILE}"
echo "  Conditioned data:  ${CONDITIONED_FILE}"
echo ""
echo "Next steps:"
echo "  1. Review ${PLOT_FILE} to verify reward distribution"
echo "  2. Adjust thresholds if needed and re-run Part 3:"
echo "     python scripts/data_curation/build_tau2_reward_conditioned.py \\"
echo "         --input ${TRAJECTORIES_FILE} \\"
echo "         --output ${CONDITIONED_FILE} \\"
echo "         --low-threshold YOUR_LOW --high-threshold YOUR_HIGH"
echo "  3. Use ${CONDITIONED_FILE} for LLaMA-Factory training"
echo ""
echo "============================================================"
