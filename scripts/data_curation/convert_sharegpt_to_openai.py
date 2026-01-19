#!/usr/bin/env python3
"""
Convert τ²-bench data from ShareGPT format to OpenAI format.

ShareGPT format:
  - Keys: "from", "value"
  - Tags: human, gpt, function_call, observation, system
  - Tool calls are SEPARATE messages
  - Has dummy messages like [Continue], [Processing...]

OpenAI format:
  - Keys: "role", "content"
  - Tags: user, assistant, tool, system
  - Tool calls are INSIDE assistant messages as "tool_calls" array
  - NO dummy messages

This conversion enables proper alignment between SFT training and RL rollout.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dummy messages to skip
DUMMY_MESSAGES = {
    "[Continue]",
    "[Processing...]", 
    "[Conversation starts]",
}


def is_dummy_message(value: str) -> bool:
    """Check if a message is a dummy placeholder."""
    return value.strip() in DUMMY_MESSAGES


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def convert_conversation(sharegpt_messages: list[dict]) -> list[dict]:
    """
    Convert a single conversation from ShareGPT to OpenAI format.
    
    Key transformations:
    1. from/value -> role/content
    2. human -> user, gpt -> assistant, observation -> tool
    3. Merge gpt + function_call + gpt (explanation) into assistant with tool_calls
    4. Remove dummy messages
    
    τ²-bench pattern: gpt → [Continue] → function_call → [Continue] → gpt → observation
    OpenAI pattern:   assistant (content + tool_calls) → tool
    
    The "explanation" gpt after function_call is merged into the tool_calls assistant content.
    """
    openai_messages = []
    i = 0
    pending_tool_call_id = None
    
    while i < len(sharegpt_messages):
        msg = sharegpt_messages[i]
        role = msg.get('from', '')
        value = msg.get('value', '')
        
        # Handle system message
        if role == 'system':
            openai_messages.append({
                "role": "system",
                "content": value
            })
            i += 1
            continue
        
        # Skip dummy messages
        if is_dummy_message(value):
            i += 1
            continue
        
        # human -> user
        if role == 'human':
            openai_messages.append({
                "role": "user",
                "content": value
            })
            i += 1
            continue
        
        # gpt -> assistant (check if followed by function_call)
        if role == 'gpt':
            # Look ahead to see if there's a function_call coming
            # (possibly after skipping dummy messages)
            next_idx = i + 1
            found_function_call = False
            
            while next_idx < len(sharegpt_messages):
                next_msg = sharegpt_messages[next_idx]
                next_role = next_msg.get('from', '')
                next_value = next_msg.get('value', '')
                
                if is_dummy_message(next_value):
                    next_idx += 1
                    continue
                
                if next_role == 'function_call':
                    found_function_call = True
                    # Merge gpt + function_call into single assistant message
                    tool_call_id = generate_tool_call_id()
                    pending_tool_call_id = tool_call_id
                    
                    # Parse function call
                    try:
                        fc_data = json.loads(next_value) if isinstance(next_value, str) else next_value
                    except json.JSONDecodeError:
                        fc_data = {"name": "unknown", "arguments": next_value}
                    
                    # Also look for explanation gpt after function_call (before observation)
                    explanation_text = ""
                    skip_to = next_idx + 1
                    
                    check_idx = next_idx + 1
                    while check_idx < len(sharegpt_messages):
                        check_msg = sharegpt_messages[check_idx]
                        check_role = check_msg.get('from', '')
                        check_value = check_msg.get('value', '')
                        
                        if is_dummy_message(check_value):
                            check_idx += 1
                            skip_to = check_idx
                            continue
                        
                        if check_role == 'gpt':
                            # This is explanation text, merge it
                            if explanation_text:
                                explanation_text += " " + check_value
                            else:
                                explanation_text = check_value
                            check_idx += 1
                            skip_to = check_idx
                            continue
                        elif check_role == 'observation':
                            # Found observation, stop looking
                            break
                        else:
                            # Something else, stop
                            break
                    
                    # Combine original gpt content with explanation
                    combined_content = value
                    if explanation_text:
                        combined_content = value + " " + explanation_text if value else explanation_text
                    
                    openai_messages.append({
                        "role": "assistant",
                        "content": combined_content,
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fc_data.get("name", "unknown"),
                                "arguments": json.dumps(fc_data.get("arguments", {})) if isinstance(fc_data.get("arguments"), dict) else str(fc_data.get("arguments", "{}"))
                            }
                        }]
                    })
                    i = skip_to
                    break
                else:
                    # No function_call following, just regular assistant message
                    break
            
            if not found_function_call:
                # Regular assistant message without tool call
                # Add empty tool_calls to avoid HuggingFace datasets None issue
                openai_messages.append({
                    "role": "assistant",
                    "content": value,
                    "tool_calls": []
                })
                i += 1
            continue
        
        # function_call without preceding gpt (standalone)
        if role == 'function_call':
            tool_call_id = generate_tool_call_id()
            pending_tool_call_id = tool_call_id
            
            try:
                fc_data = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                fc_data = {"name": "unknown", "arguments": value}
            
            # Look for explanation gpt after function_call
            explanation_text = ""
            skip_to = i + 1
            
            check_idx = i + 1
            while check_idx < len(sharegpt_messages):
                check_msg = sharegpt_messages[check_idx]
                check_role = check_msg.get('from', '')
                check_value = check_msg.get('value', '')
                
                if is_dummy_message(check_value):
                    check_idx += 1
                    skip_to = check_idx
                    continue
                
                if check_role == 'gpt':
                    if explanation_text:
                        explanation_text += " " + check_value
                    else:
                        explanation_text = check_value
                    check_idx += 1
                    skip_to = check_idx
                    continue
                elif check_role == 'observation':
                    break
                else:
                    break
            
            openai_messages.append({
                "role": "assistant",
                "content": explanation_text,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fc_data.get("name", "unknown"),
                        "arguments": json.dumps(fc_data.get("arguments", {})) if isinstance(fc_data.get("arguments"), dict) else str(fc_data.get("arguments", "{}"))
                    }
                }]
            })
            i = skip_to
            continue
        
        # observation -> tool
        if role == 'observation':
            openai_messages.append({
                "role": "tool",
                "tool_call_id": pending_tool_call_id or generate_tool_call_id(),
                "content": value
            })
            i += 1
            continue
        
        # Unknown role, skip
        logger.warning(f"Unknown role: {role}")
        i += 1
    
    return openai_messages


def ensure_alternation(messages: list[dict]) -> list[dict]:
    """
    Ensure messages follow proper alternation for LLaMA-Factory.
    
    Rules (after system):
    - Position 0, 2, 4...: user or tool
    - Position 1, 3, 5...: assistant
    
    If conversation starts with assistant, insert a user message.
    """
    if not messages:
        return messages
    
    # Find first non-system message
    first_non_system_idx = 0
    for i, msg in enumerate(messages):
        if msg['role'] != 'system':
            first_non_system_idx = i
            break
    
    # Check if first non-system message is assistant (should be user)
    if first_non_system_idx < len(messages):
        first_msg = messages[first_non_system_idx]
        if first_msg['role'] == 'assistant':
            # Insert a user message before the assistant greeting
            messages.insert(first_non_system_idx, {
                "role": "user",
                "content": "[Start conversation]"
            })
    
    return messages


def convert_sample(sample: dict) -> dict:
    """Convert a single sample from ShareGPT to OpenAI format."""
    result = {}
    
    # Convert conversations
    if 'conversations' in sample:
        messages = convert_conversation(sample['conversations'])
        # Ensure proper alternation
        result['messages'] = ensure_alternation(messages)
    
    # Keep tools (already in OpenAI format)
    if 'tools' in sample:
        result['tools'] = sample['tools']
    
    # Keep metadata
    for key in ['final_reward', 'component_reward', 'reward_breakdown', 
                'domain', 'task_id', 'agent_model', 'trial', 'termination_reason',
                'model_tier']:
        if key in sample:
            result[key] = sample[key]
    
    return result


def convert_file(input_path: Path, output_path: Path) -> dict:
    """Convert a file from ShareGPT to OpenAI format."""
    logger.info(f"Loading: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Converting {len(data)} samples...")
    
    converted = []
    stats = {
        "total_samples": len(data),
        "total_sharegpt_messages": 0,
        "total_openai_messages": 0,
        "dummy_messages_removed": 0,
    }
    
    for sample in data:
        sharegpt_count = len(sample.get('conversations', []))
        stats["total_sharegpt_messages"] += sharegpt_count
        
        converted_sample = convert_sample(sample)
        openai_count = len(converted_sample.get('messages', []))
        stats["total_openai_messages"] += openai_count
        stats["dummy_messages_removed"] += (sharegpt_count - openai_count)
        
        converted.append(converted_sample)
    
    logger.info(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT to OpenAI format")
    parser.add_argument("--input", type=str, required=True, help="Input ShareGPT JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output OpenAI JSON file")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    stats = convert_file(input_path, output_path)
    
    logger.info("=" * 60)
    logger.info("Conversion Statistics")
    logger.info("=" * 60)
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"ShareGPT messages: {stats['total_sharegpt_messages']}")
    logger.info(f"OpenAI messages: {stats['total_openai_messages']}")
    logger.info(f"Messages reduced: {stats['dummy_messages_removed']} ({stats['dummy_messages_removed']/stats['total_sharegpt_messages']*100:.1f}%)")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
