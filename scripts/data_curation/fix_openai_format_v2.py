#!/usr/bin/env python3
"""
Fix OpenAI format dataset for LLaMA-Factory compatibility - V2.

The key insight: LLaMA-Factory expects STRICT alternation:
  user(0) → asst(1) → user(2) → asst(3) → ...

Where:
- Even positions (0,2,4...): user OR observation
- Odd positions (1,3,5...): assistant OR function

The fix strategy:
1. First, merge all consecutive tool-calling turns
2. Then, insert dummy messages where needed to fix alternation
"""

import json
import copy
from pathlib import Path


def get_effective_role(msg):
    """Get the effective role for pattern checking."""
    role = msg['role']
    if role == 'assistant':
        if msg.get('tool_calls') and len(msg.get('tool_calls', [])) > 0:
            return 'function'
        return 'assistant'
    elif role == 'tool':
        return 'observation'
    return role


def fix_conversation(messages):
    """Fix conversation to match strict alternating pattern."""
    if not messages:
        return messages
    
    result = []
    
    # Step 1: Extract system message
    if messages[0]['role'] == 'system':
        result.append(messages[0])
        messages = messages[1:]
    
    # Step 2: First pass - merge consecutive assistant+tool sequences
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
            
            # Collect all consecutive tool responses
            while i < len(messages) and messages[i]['role'] == 'tool':
                combined_tools.append(messages[i])
                i += 1
            
            # Check for more assistant(tool)+tool sequences
            while i < len(messages):
                next_msg = messages[i]
                if next_msg['role'] == 'assistant' and next_msg.get('tool_calls') and len(next_msg.get('tool_calls', [])) > 0:
                    # Another tool-calling assistant
                    if next_msg.get('content'):
                        combined_assistant['content'] += ('\n' if combined_assistant['content'] else '') + next_msg['content']
                    combined_assistant['tool_calls'].extend(next_msg.get('tool_calls', []))
                    i += 1
                    # Collect its tool responses
                    while i < len(messages) and messages[i]['role'] == 'tool':
                        combined_tools.append(messages[i])
                        i += 1
                else:
                    break
            
            merged.append(combined_assistant)
            if combined_tools:
                # Merge all tool responses into one
                merged.append({
                    'role': 'tool',
                    'content': '\n---\n'.join([t.get('content', '') for t in combined_tools]),
                    'tool_call_id': combined_tools[0].get('tool_call_id', 'merged')
                })
        else:
            merged.append(msg)
            i += 1
    
    # Step 3: Second pass - fix alternation by inserting dummy messages
    fixed = []
    for i, msg in enumerate(merged):
        effective = get_effective_role(msg)
        
        if not fixed:
            # First message should be user/observation
            if effective in ('user', 'observation'):
                fixed.append(msg)
            else:
                # Insert dummy user first
                fixed.append({'role': 'user', 'content': '[Start conversation]'})
                fixed.append(msg)
        else:
            last_effective = get_effective_role(fixed[-1])
            expected_odd = len(fixed) % 2 == 1  # Should this position be odd (asst/func)?
            
            if expected_odd:
                # Position should be assistant/function
                if effective in ('assistant', 'function'):
                    fixed.append(msg)
                elif effective in ('user', 'observation'):
                    # Wrong! Need to insert dummy assistant before this
                    fixed.append({
                        'role': 'assistant',
                        'content': 'I understand. Let me help you with that.',
                        'tool_calls': []
                    })
                    fixed.append(msg)
                else:
                    fixed.append(msg)
            else:
                # Position should be user/observation
                if effective in ('user', 'observation'):
                    fixed.append(msg)
                elif effective in ('assistant', 'function'):
                    # Wrong! Need to insert dummy user before this
                    fixed.append({'role': 'user', 'content': '[Continue]'})
                    fixed.append(msg)
                else:
                    fixed.append(msg)
    
    # Step 4: Ensure even count - must end with assistant/function
    if len(fixed) % 2 != 0:
        last_effective = get_effective_role(fixed[-1])
        if last_effective in ('user', 'observation'):
            fixed.append({
                'role': 'assistant',
                'content': 'Is there anything else I can help you with?',
                'tool_calls': []
            })
        else:
            # Ends with assistant/function but odd count - remove last or add user+asst
            fixed.append({'role': 'user', 'content': '[End of conversation]'})
            fixed.append({
                'role': 'assistant', 
                'content': 'Thank you for contacting us. Have a great day!',
                'tool_calls': []
            })
    
    # Rebuild with system message
    return result + fixed


def validate_conversation(messages):
    """Validate conversation matches expected pattern."""
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


def fix_dataset(input_path, output_path):
    """Fix entire dataset."""
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} examples...")
    fixed_data = []
    stats = {'valid_before': 0, 'valid_after': 0, 'fixed': 0, 'still_broken': 0}
    broken_examples = []
    
    for i, example in enumerate(data):
        messages = example['messages']
        
        is_valid, _ = validate_conversation(messages)
        if is_valid:
            stats['valid_before'] += 1
        
        fixed_messages = fix_conversation(messages)
        is_valid_after, reason = validate_conversation(fixed_messages)
        
        if is_valid_after:
            stats['valid_after'] += 1
            if not is_valid:
                stats['fixed'] += 1
        else:
            stats['still_broken'] += 1
            broken_examples.append((i, reason))
        
        fixed_example = copy.deepcopy(example)
        fixed_example['messages'] = fixed_messages
        fixed_data.append(fixed_example)
    
    print(f"\nStats:")
    print(f"  Valid before: {stats['valid_before']}/{len(data)}")
    print(f"  Valid after:  {stats['valid_after']}/{len(data)}")
    print(f"  Fixed:        {stats['fixed']}")
    print(f"  Still broken: {stats['still_broken']}")
    
    if broken_examples:
        print(f"\nFirst few broken examples:")
        for i, reason in broken_examples[:5]:
            print(f"  Example {i}: {reason}")
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    return stats


if __name__ == '__main__':
    base_path = Path('data/tau2/sft')
    
    files_to_fix = [
        ('all_domains_success_train_wm_openai.json', 'all_domains_success_train_wm_openai_v2.json'),
        ('all_domains_success_test_wm_openai.json', 'all_domains_success_test_wm_openai_v2.json'),
        ('all_domains_success_train_sft_openai.json', 'all_domains_success_train_sft_openai_v2.json'),
        ('all_domains_success_test_sft_openai.json', 'all_domains_success_test_sft_openai_v2.json'),
    ]
    
    for input_file, output_file in files_to_fix:
        print("\n" + "=" * 60)
        print(f"FIXING {input_file}")
        print("=" * 60)
        fix_dataset(base_path / input_file, base_path / output_file)
    
    print("\n" + "=" * 60)
    print("DONE! Fixed files saved with '_v2' suffix")
    print("=" * 60)
