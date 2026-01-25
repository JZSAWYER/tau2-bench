#!/usr/bin/env python3
"""
Split large JSON dataset files into smaller chunks for git tracking.

The large expanded_*.json files are in .gitignore.
This script splits them into chunks that can be committed.

Usage:
    python scripts/data_curation/split_chunks.py
"""

import json
import math
from pathlib import Path


def split_into_chunks(input_file: Path, output_dir: Path, max_size_mb: float = 10.0):
    """
    Split a large JSON file into smaller chunks.
    
    Args:
        input_file: Path to large JSON file
        output_dir: Directory to store chunks
        max_size_mb: Target max size per chunk in MB
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return False
    
    print(f"Splitting {input_file.name}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    file_size_mb = input_file.stat().st_size / (1024 * 1024)
    
    # Estimate number of chunks needed
    num_chunks = max(1, math.ceil(file_size_mb / max_size_mb))
    samples_per_chunk = math.ceil(total_samples / num_chunks)
    
    print(f"  Total: {total_samples} samples, {file_size_mb:.1f} MB")
    print(f"  Splitting into ~{num_chunks} chunks of ~{samples_per_chunk} samples each")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old chunks
    for old_chunk in output_dir.glob("chunk_*.json"):
        old_chunk.unlink()
    
    # Split and save chunks
    for i in range(num_chunks):
        start_idx = i * samples_per_chunk
        end_idx = min((i + 1) * samples_per_chunk, total_samples)
        chunk_data = data[start_idx:end_idx]
        
        if not chunk_data:
            continue
        
        chunk_file = output_dir / f"chunk_{i:03d}.json"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        chunk_size_mb = chunk_file.stat().st_size / (1024 * 1024)
        print(f"  Saved {chunk_file.name}: {len(chunk_data)} samples ({chunk_size_mb:.1f} MB)")
    
    return True


def main():
    base_path = Path(__file__).parent.parent.parent / "data" / "tau2" / "sft"
    
    # Split WM datasets
    print("=" * 60)
    print("WORLD MODEL DATASETS (with reward tokens)")
    print("=" * 60)
    split_into_chunks(
        base_path / "expanded_wm_train.json",
        base_path / "expanded_wm_train_chunks"
    )
    
    print()
    split_into_chunks(
        base_path / "expanded_wm_test.json",
        base_path / "expanded_wm_test_chunks"
    )
    
    # Split SFT datasets
    print("\n" + "=" * 60)
    print("SFT DATASETS (with identical fields, no reward tokens)")
    print("=" * 60)
    split_into_chunks(
        base_path / "expanded_sft_train.json",
        base_path / "expanded_sft_train_chunks"
    )
    
    print()
    split_into_chunks(
        base_path / "expanded_sft_test.json",
        base_path / "expanded_sft_test_chunks"
    )
    
    print("\n" + "=" * 60)
    print("Done! Chunks are ready to commit.")
    print("=" * 60)
    print("\nNote: SFT and WM now have IDENTICAL fields.")
    print("The ONLY difference is the reward token in messages.")


if __name__ == "__main__":
    main()
