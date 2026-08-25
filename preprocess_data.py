"""
Preprocess large CSV files into smaller, memory-efficient .npz files
This dramatically reduces memory usage and loading time
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import signal

# Configuration
DATA_DIR = Path("EEG_Data")
OUTPUT_DIR = Path("preprocessed_data")
SAMPLING_RATE = 256
WINDOW_SIZE_SEC = 4.0
OVERLAP_RATIO = 0.5
TARGET_CHANNELS = 23

# Preprocessing
LOWCUT = 0.5
HIGHCUT = 50.0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def process_csv_file(csv_path, output_path, chunk_size=100000):
    """Process a single CSV file into segmented .npz format"""
    
    # Get file info
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  Processing {csv_path.name} ({file_size_mb:.1f} MB)...")
    
    all_segments = []
    all_labels = []
    
    window_size = int(WINDOW_SIZE_SEC * SAMPLING_RATE)  # 1024 samples
    step = int(window_size * (1 - OVERLAP_RATIO))  # 512 samples
    
    # Read in chunks
    chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True)
    
    buffer_data = None
    buffer_labels = None
    
    for chunk_idx, chunk in enumerate(chunk_iter):
        # Get channel columns (skip Time column, skip last Seizure column)
        channel_cols = chunk.columns[1:-1]
        
        # Limit to TARGET_CHANNELS
        channel_cols = channel_cols[:TARGET_CHANNELS]
        
        # Get data
        chunk_data = chunk[channel_cols].values.astype(np.float32).T  # (channels, time)
        chunk_labels = chunk[chunk.columns[-1]].values.astype(np.int64)
        
        # Append to buffer
        if buffer_data is None:
            buffer_data = chunk_data
            buffer_labels = chunk_labels
        else:
            buffer_data = np.concatenate([buffer_data, chunk_data], axis=1)
            buffer_labels = np.concatenate([buffer_labels, chunk_labels])
        
        # Segment from buffer
        while buffer_data.shape[1] >= window_size:
            segment = buffer_data[:, :window_size]
            seg_labels = buffer_labels[:window_size]
            
            # Apply bandpass filter
            try:
                segment = butter_bandpass_filter(segment, LOWCUT, HIGHCUT, SAMPLING_RATE)
            except:
                pass  # Skip filter if it fails
            
            # Normalize per channel (z-score)
            mean = np.mean(segment, axis=1, keepdims=True)
            std = np.std(segment, axis=1, keepdims=True) + 1e-8
            segment = (segment - mean) / std
            
            # Determine segment label (seizure if >50% of samples are seizure)
            segment_label = 1 if np.mean(seg_labels) > 0.5 else 0
            
            all_segments.append(segment.astype(np.float32))
            all_labels.append(segment_label)
            
            # Shift buffer by step size
            buffer_data = buffer_data[:, step:]
            buffer_labels = buffer_labels[step:]
    
    if len(all_segments) == 0:
        print(f"    No valid segments extracted!")
        return 0
    
    # Stack and save
    segments = np.stack(all_segments, axis=0)  # (n_segments, channels, time)
    labels = np.array(all_labels, dtype=np.int64)
    
    # Save as compressed npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, segments=segments, labels=labels)
    
    n_seizures = np.sum(labels)
    file_size_kb = output_path.stat().st_size / 1024
    print(f"    -> {len(segments)} segments, {n_seizures} seizures, saved as {file_size_kb:.1f} KB")
    
    return len(segments)


def main():
    print("="*60)
    print("EEG Data Preprocessing")
    print("="*60)
    print(f"Input: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Window: {WINDOW_SIZE_SEC}s ({int(WINDOW_SIZE_SEC * SAMPLING_RATE)} samples)")
    print(f"Overlap: {OVERLAP_RATIO*100}%")
    print("="*60)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = list(DATA_DIR.rglob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Process each file
    total_segments = 0
    processed_files = 0
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        patient_id = csv_file.parent.name
        output_subdir = OUTPUT_DIR / patient_id
        output_file = output_subdir / csv_file.name.replace('.csv', '.npz')
        
        if output_file.exists():
            print(f"\n  Skipping {csv_file.name} (already processed)")
            # Load and count existing segments
            data = np.load(output_file)
            total_segments += len(data['segments'])
            processed_files += 1
            continue
        
        try:
            n_segments = process_csv_file(csv_file, output_file)
            total_segments += n_segments
            processed_files += 1
        except Exception as e:
            print(f"\n  Error processing {csv_file.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Files processed: {processed_files}")
    print(f"Total segments: {total_segments}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Show output size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.npz"))
    print(f"Total output size: {total_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
