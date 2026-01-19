#!/usr/bin/env python3
"""
Fix OpenAI format dataset for LLaMA-Factory compatibility.

LLaMA-Factory expects strict alternating pattern after conversion:
- Index 0, 2, 4...: user or observation
- Index 1, 3, 5...: assistant or function

Problem: Our data has consecutive tool calls:
  user → assistant(tool) → tool → assistant(tool) → tool → ...

Solution: Merge consecutive assistant+tool pairs into single turns:
  user → assistant(all tools merged) → tool(all responses merged) → user → ...
"""

import json
import copy
from pathlib import Path


def fix_conversation(messages):
    """Fix a single conversation to match LLaMA-Factory's expected pattern."""
    if not messages:
        return messages
    
    result = []
    i = 0
    
    # Keep system message
    if messages[0]['role'] == 'system':
        result.append(messages[0])
        i = 1
    
    while i < len(messages):
        msg = messages[i]
        
        if msg['role'] == 'user':
            result.append(msg)
            i += 1
            
        elif msg['role'] == 'assistant':
            # Check if this assistant has tool_calls
            has_tools = msg.get('tool_calls') and len(msg.get('tool_calls', [])) > 0
            
            if not has_tools:
                # Regular assistant message without tools
                result.append(msg)
                i += 1
            else:
                # Assistant with tool calls - need to collect consecutive tool sequences
                merged_assistant = {
                    'role': 'assistant',
                    'content': msg.get('content', '') or '',
                    'tool_calls': list(msg.get('tool_calls', []))
                }
                merged_tool_responses = []
                i += 1
                
                # Collect all consecutive tool responses
                while i < len(messages) and messages[i]['role'] == 'tool':
                    merged_tool_responses.append({
                        'tool_call_id': messages[i].get('tool_call_id', ''),
                        'content': messages[i].get('content', '')
                    })
                    i += 1
                
                # Check for more consecutive assistant(tool)+tool pairs
                while i < len(messages) and messages[i]['role'] == 'assistant':
                    next_msg = messages[i]
                    next_has_tools = next_msg.get('tool_calls') and len(next_msg.get('tool_calls', [])) > 0
                    
                    if next_has_tools:
                        # Merge this assistant's tool calls
                        if next_msg.get('content'):
                            if merged_assistant['content']:
                                merged_assistant['content'] += '\n' + next_msg['content']
                            else:
                                merged_assistant['content'] = next_msg['content']
                        merged_assistant['tool_calls'].extend(next_msg.get('tool_calls', []))
                        i += 1
                        
                        # Collect its tool responses
                        while i < len(messages) and messages[i]['role'] == 'tool':
                            merged_tool_responses.append({
                                'tool_call_id': messages[i].get('tool_call_id', ''),
                                'content': messages[i].get('content', '')
                            })
                            i += 1
                    else:
                        # Next assistant doesn't have tools - stop merging
                        break
                
                # Add merged assistant
                result.append(merged_assistant)
                
                # Add merged tool responses as single tool message
                if merged_tool_responses:
                    # Create combined tool response
                    combined_content = '\n---\n'.join([r['content'] for r in merged_tool_responses])
                    result.append({
                        'role': 'tool',
                        'content': combined_content,
                        'tool_call_id': merged_tool_responses[0]['tool_call_id']  # Use first ID
                    })
                    
        elif msg['role'] == 'tool':
            # Orphan tool message - shouldn't happen but handle it
            result.append(msg)
            i += 1
        else:
            result.append(msg)
            i += 1
    
    return result


def ensure_even_turns(messages):
    """Ensure conversation has even number of turns after system message removal."""
    if not messages:
        return messages
    
    # Separate system message
    system_msg = None
    content_msgs = messages
    if messages[0]['role'] == 'system':
        system_msg = messages[0]
        content_msgs = messages[1:]
    
    # Check if we have even number of content messages
    # LLaMA-Factory expects: user/obs(0) → asst/func(1) → user/obs(2) → asst/func(3) ...
    # So even count is required
    
    if len(content_msgs) % 2 != 0:
        # Need to make it even - check what the last message is
        if content_msgs and content_msgs[-1]['role'] in ('assistant',):
            # Ends with assistant - that's fine, just odd count
            # Add a dummy user acknowledgment at the end? No, let's remove trailing
            pass
        elif content_msgs and content_msgs[-1]['role'] == 'tool':
            # Ends with tool/observation - need an assistant response
            content_msgs.append({
                'role': 'assistant',
                'content': 'Is there anything else I can help you with?',
                'tool_calls': []
            })
        elif content_msgs and content_msgs[-1]['role'] == 'user':
            # Ends with user - need an assistant response  
            content_msgs.append({
                'role': 'assistant',
                'content': 'Thank you for your patience. Is there anything else I can help you with?',
                'tool_calls': []
            })
    
    # Rebuild
    result = []
    if system_msg:
        result.append(system_msg)
    result.extend(content_msgs)
    
    return result


def validate_conversation(messages):
    """Validate conversation matches LLaMA-Factory expected pattern."""
    if not messages:
        return False, "Empty messages"
    
    # Skip system message
    content = messages[1:] if messages[0]['role'] == 'system' else messages
    
    if len(content) % 2 != 0:
        return False, f"Odd message count: {len(content)}"
    
    # Check alternation
    # Even indices (0, 2, 4...): user or observation(tool)
    # Odd indices (1, 3, 5...): assistant or function
    for idx, msg in enumerate(content):
        role = msg['role']
        
        # Convert for checking
        if role == 'assistant' and msg.get('tool_calls') and len(msg.get('tool_calls', [])) > 0:
            effective_role = 'function'
        elif role == 'tool':
            effective_role = 'observation'
        else:
            effective_role = role
        
        if idx % 2 == 0:  # Even - should be user or observation
            if effective_role not in ('user', 'observation'):
                return False, f"Index {idx}: expected user/observation, got {effective_role}"
        else:  # Odd - should be assistant or function
            if effective_role not in ('assistant', 'function'):
                return False, f"Index {idx}: expected assistant/function, got {effective_role}"
    
    return True, "OK"


def fix_dataset(input_path, output_path):
    """Fix entire dataset file."""
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} examples...")
    fixed_data = []
    stats = {'valid_before': 0, 'valid_after': 0, 'fixed': 0, 'still_broken': 0}
    
    for i, example in enumerate(data):
        messages = example['messages']
        
        # Check if already valid
        is_valid, _ = validate_conversation(messages)
        if is_valid:
            stats['valid_before'] += 1
        
        # Apply fix
        fixed_messages = fix_conversation(messages)
        fixed_messages = ensure_even_turns(fixed_messages)
        
        # Validate fixed version
        is_valid_after, reason = validate_conversation(fixed_messages)
        
        if is_valid_after:
            stats['valid_after'] += 1
            if not is_valid:
                stats['fixed'] += 1
        else:
            stats['still_broken'] += 1
            if i < 5:  # Show first few errors
                print(f"  Example {i} still broken: {reason}")
        
        # Create fixed example
        fixed_example = copy.deepcopy(example)
        fixed_example['messages'] = fixed_messages
        fixed_data.append(fixed_example)
    
    print(f"\nStats:")
    print(f"  Valid before: {stats['valid_before']}/{len(data)}")
    print(f"  Valid after:  {stats['valid_after']}/{len(data)}")
    print(f"  Fixed:        {stats['fixed']}")
    print(f"  Still broken: {stats['still_broken']}")
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    return stats


if __name__ == '__main__':
    base_path = Path('data/tau2/sft')
    
    # Fix train file
    print("=" * 60)
    print("FIXING TRAIN FILE")
    print("=" * 60)
    fix_dataset(
        base_path / 'all_domains_success_train_wm_openai.json',
        base_path / 'all_domains_success_train_wm_openai_fixed.json'
    )
    
    # Fix test file
    print("\n" + "=" * 60)
    print("FIXING TEST FILE")
    print("=" * 60)
    fix_dataset(
        base_path / 'all_domains_success_test_wm_openai.json',
        base_path / 'all_domains_success_test_wm_openai_fixed.json'
    )
    
    # Also fix SFT versions
    print("\n" + "=" * 60)
    print("FIXING SFT TRAIN FILE")
    print("=" * 60)
    fix_dataset(
        base_path / 'all_domains_success_train_sft_openai.json',
        base_path / 'all_domains_success_train_sft_openai_fixed.json'
    )
    
    print("\n" + "=" * 60)
    print("FIXING SFT TEST FILE")
    print("=" * 60)
    fix_dataset(
        base_path / 'all_domains_success_test_sft_openai.json',
        base_path / 'all_domains_success_test_sft_openai_fixed.json'
    )
    
    print("\n" + "=" * 60)
    print("DONE! Fixed files saved with '_fixed' suffix")
    print("=" * 60)
