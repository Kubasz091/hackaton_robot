
import pandas as pd
from pathlib import Path
import sys

def check_episodes(dataset_path):
    root = Path(dataset_path)
    meta_path = root / "meta/episodes"
    
    print(f"Checking metadata in {meta_path}")
    
    total_episodes_found = 0
    episodes_indices = []
    
    # Walk through chunks
    for chunk in sorted(meta_path.glob("chunk-*")):
        print(f"  Checking chunk: {chunk.name}")
        for pfile in sorted(chunk.glob("*.parquet")):
            try:
                df = pd.read_parquet(pfile)
                count = len(df)
                total_episodes_found += count
                # Assuming 'episode_index' is a column
                if 'episode_index' in df.columns:
                    indices = df['episode_index'].tolist()
                    episodes_indices.extend(indices)
                    print(f"    {pfile.name}: {count} episodes (Indices: {min(indices)} to {max(indices)})")
                    
                    if 'dataset_from_index' in df.columns:
                         from_indices = df['dataset_from_index'].tolist()
                         to_indices = df['dataset_to_index'].tolist()
                         print(f"    Frame ranges: {min(from_indices)} - {max(to_indices)}")
                         # Check strict monotonicity
                         if not all(x < y for x, y in zip(from_indices, from_indices[1:])):
                             print("    WARNING: dataset_from_index is not strictly increasing!")
                             
                    if 'meta/episodes/chunk_index' in df.columns:
                        chunks = df['meta/episodes/chunk_index'].tolist()
                        files = df['meta/episodes/file_index'].tolist()
                        # Zip them to see unique combinations
                        pointers = list(zip(chunks, files))
                        unique_pointers = set(pointers)
                        print(f"    Unique data file pointers: {len(unique_pointers)} (rows: {count})")
                        print(f"    Sample pointers: {pointers[:5]} ... {pointers[-5:]}")
                else:
                    print(f"    {pfile.name}: {count} rows (No episode_index column?)")
                    print(f"    Columns: {df.columns}")
            except Exception as e:
                print(f"    Error reading {pfile.name}: {e}")

    print(f"Total episodes found in parquet: {total_episodes_found}")
    print(f"Max episode index: {max(episodes_indices) if episodes_indices else 'None'}")
    
    unique_indices = set(episodes_indices)
    print(f"Unique indices count: {len(unique_indices)}")
    if len(unique_indices) != total_episodes_found:
        print("WARNING: Duplicate episode indices found!")
        from collections import Counter
        duplicates = [item for item, count in Counter(episodes_indices).items() if count > 1]
        print(f"Duplicate sample: {duplicates[:10]}")
    
    if 340 in episodes_indices:
        print("Episode 340 IS present in metadata parquet.")
    else:
        print("Episode 340 IS NOT present in metadata parquet.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_metadata.py <dataset_path>")
    else:
        check_episodes(sys.argv[1])
