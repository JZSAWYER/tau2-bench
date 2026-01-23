#!/usr/bin/env python3
"""
Merge split JSON chunks back into full dataset files.

Run this on the remote machine after cloning to reconstruct the large files.

Usage:
    python scripts/data_curation/merge_chunks.py
"""

import json
from pathlib import Path


def merge_chunks(chunks_dir: Path, output_file: Path):
    """Merge JSON chunks back into a single file."""
    chunks_dir = Path(chunks_dir)
    output_file = Path(output_file)
    
    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return False
    
    chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return False
    
    print(f"Merging {len(chunk_files)} chunks into {output_file}...")
    
    merged_data = []
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        merged_data.extend(chunk_data)
        print(f"  Loaded {chunk_file.name}: {len(chunk_data)} samples")
    
    print(f"  Total: {len(merged_data)} samples")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_file} ({size_mb:.1f} MB)")
    return True


def main():
    base_path = Path(__file__).parent.parent.parent / "data" / "tau2" / "sft"
    
    # Merge train chunks
    print("=" * 60)
    merge_chunks(
        base_path / "expanded_wm_train_chunks",
        base_path / "expanded_wm_train.json"
    )
    
    # Merge test chunks
    print("\n" + "=" * 60)
    merge_chunks(
        base_path / "expanded_wm_test_chunks",
        base_path / "expanded_wm_test.json"
    )
    
    print("\n" + "=" * 60)
    print("Done! Large files reconstructed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
