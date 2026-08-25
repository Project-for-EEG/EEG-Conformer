"""
CHB-MIT Dataset Downloader and Converter
Downloads EDF files from PhysioNet and converts them to labeled CSV format
"""

import os
import urllib.request
import re
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Try to import mne for EDF reading
try:
    import mne
    mne.set_log_level('ERROR')  # Suppress verbose output
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("WARNING: mne library not installed. Install with: pip install mne")

# Configuration
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("EEG_Data")

# Target channels (matching your existing data format)
TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1'
]

# Patients to download (modify this list as needed)
# Skip patients you already have (CHB03, CHB05)
PATIENTS_TO_DOWNLOAD = [
    'chb01', 'chb02', 'chb04', 'chb06', 'chb07', 'chb08',
    'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14',
    'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20',
    'chb21', 'chb22', 'chb23', 'chb24'
]

# Set to None to download ALL files, or a number to limit per patient
# Each file is ~40MB and ~1 hour of recording
# 5 files per patient = ~4.4GB total, good for expanded training
MAX_FILES_PER_PATIENT = 5


def download_file(url, dest_path):
    """Download a file from URL to destination path"""
    if dest_path.exists():
        print(f"  [SKIP] Already exists: {dest_path.name}")
        return True
    
    try:
        print(f"  [DOWNLOAD] {dest_path.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest_path)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Failed: {e}")
        return False


def parse_summary_file(summary_path):
    """Parse the summary file to get seizure annotations"""
    seizures = {}
    
    if not summary_path.exists():
        return seizures
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Split by "File Name:" to get each file's info
    file_sections = re.split(r'File Name:\s*', content)
    
    for section in file_sections[1:]:  # Skip first empty section
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        filename = lines[0].strip()
        
        # Find seizure info.
        # CHB-MIT uses two layouts, sometimes within the same summary:
        #   "Seizure Start Time: 2996 seconds"      (single seizure in file)
        #   "Seizure 1 Start Time: 1724 seconds"    (multiple seizures in file)
        # Anchor the number to the "Start/End Time:" label and require the
        # "seconds" suffix -- a bare \d+ picks up the seizure index instead of
        # the timestamp, and would also match "File Start Time: 19:08:32".
        starts = []
        ends = []
        for line in lines:
            start_match = re.search(r'Start Time:\s*(\d+)\s*seconds', line, re.IGNORECASE)
            if start_match:
                starts.append(int(start_match.group(1)))
                continue
            end_match = re.search(r'End Time:\s*(\d+)\s*seconds', line, re.IGNORECASE)
            if end_match:
                ends.append(int(end_match.group(1)))

        seizure_times = [
            {'start': s, 'end': e}
            for s, e in zip(starts, ends)
            if e > s
        ]

        if seizure_times:
            seizures[filename] = seizure_times
    
    return seizures


def normalize_channel_name(ch_name):
    """Normalize channel name for matching"""
    # Remove common prefixes/suffixes and normalize
    ch = ch_name.upper().strip()
    ch = ch.replace(' ', '').replace('_', '-').replace('.', '-')
    ch = ch.replace('EEG', '').strip()
    ch = ch.lstrip('-').rstrip('-')
    return ch


def find_matching_channel(target_ch, available_channels):
    """Find the best matching channel from available channels"""
    target_norm = normalize_channel_name(target_ch)
    
    # First try exact match
    for i, ch in enumerate(available_channels):
        if normalize_channel_name(ch) == target_norm:
            return i
    
    # Try partial match (e.g., FP1-F7 matches EEG FP1-F7)
    for i, ch in enumerate(available_channels):
        ch_norm = normalize_channel_name(ch)
        if target_norm in ch_norm or ch_norm in target_norm:
            return i
    
    # Try matching just the electrode pair
    target_parts = target_norm.replace('-0', '').replace('-1', '').split('-')
    for i, ch in enumerate(available_channels):
        ch_parts = normalize_channel_name(ch).split('-')
        if len(target_parts) >= 2 and len(ch_parts) >= 2:
            if target_parts[0] == ch_parts[0] and target_parts[1] == ch_parts[1]:
                return i
    
    return None


def convert_edf_to_csv(edf_path, csv_path, seizure_info=None):
    """Convert EDF file to labeled CSV format"""
    if not MNE_AVAILABLE:
        print("  [ERROR] mne library required for conversion")
        return False
    
    try:
        # Read EDF file
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Get data and channel names
        data = raw.get_data()
        ch_names = raw.ch_names
        sfreq = raw.info['sfreq']
        n_samples = data.shape[1]
        
        # Create time column
        times = np.arange(n_samples) / sfreq
        
        # Map channels to target format
        channel_data = {}
        matched_count = 0
        for target_ch in TARGET_CHANNELS:
            idx = find_matching_channel(target_ch, ch_names)
            if idx is not None:
                channel_data[target_ch] = data[idx]
                matched_count += 1
            else:
                # Use zeros for missing channels
                channel_data[target_ch] = np.zeros(n_samples)
        
        # Create seizure labels
        seizure_labels = np.zeros(n_samples, dtype=int)
        if seizure_info:
            for seizure in seizure_info:
                start_idx = int(seizure['start'] * sfreq)
                end_idx = min(int(seizure['end'] * sfreq), n_samples)
                seizure_labels[start_idx:end_idx] = 1
        
        # Build DataFrame
        df_data = {'Time (s)': times}
        for ch in TARGET_CHANNELS:
            df_data[ch] = channel_data[ch]
        df_data['Seizure'] = seizure_labels
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        # Report
        seizure_count = seizure_labels.sum()
        duration_min = n_samples / sfreq / 60
        if seizure_count > 0:
            seizure_dur = seizure_count / sfreq
            print(f"  [*] Converted: {duration_min:.1f}min, {matched_count}/{len(TARGET_CHANNELS)} channels, SEIZURE: {seizure_dur:.1f}s")
        else:
            print(f"  [*] Converted: {duration_min:.1f}min, {matched_count}/{len(TARGET_CHANNELS)} channels")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_patient(patient_id):
    """Download and convert all files for a patient"""
    print(f"\n{'='*60}")
    print(f"Processing {patient_id.upper()}")
    print('='*60)
    
    patient_raw_dir = RAW_DATA_DIR / patient_id
    patient_raw_dir.mkdir(parents=True, exist_ok=True)
    
    patient_output_dir = OUTPUT_DIR / patient_id.upper()
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download summary file first
    summary_url = f"{BASE_URL}/{patient_id}/{patient_id}-summary.txt"
    summary_path = patient_raw_dir / f"{patient_id}-summary.txt"
    download_file(summary_url, summary_path)
    
    # Parse seizure annotations
    seizure_annotations = parse_summary_file(summary_path)
    print(f"Found seizure annotations for {len(seizure_annotations)} files")
    
    # Get list of EDF files for this patient from RECORDS
    records_path = RAW_DATA_DIR / "RECORDS.txt"
    edf_files = []
    with open(records_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{patient_id}/"):
                edf_files.append(line)
    
    # Prioritize files with seizures
    files_with_seizures = [f for f in edf_files if os.path.basename(f) in seizure_annotations]
    files_without_seizures = [f for f in edf_files if os.path.basename(f) not in seizure_annotations]
    
    print(f"Total files: {len(edf_files)} ({len(files_with_seizures)} with seizures)")
    
    # Reorder: seizure files first
    ordered_files = files_with_seizures + files_without_seizures
    
    # Limit files if configured
    if MAX_FILES_PER_PATIENT is not None:
        # Take all seizure files first, then fill with non-seizure
        n_seizure = min(len(files_with_seizures), MAX_FILES_PER_PATIENT)
        n_normal = MAX_FILES_PER_PATIENT - n_seizure
        ordered_files = files_with_seizures[:n_seizure] + files_without_seizures[:n_normal]
        print(f"Limiting to {len(ordered_files)} files")
    
    converted_count = 0
    for edf_file in ordered_files:
        filename = os.path.basename(edf_file)
        edf_url = f"{BASE_URL}/{edf_file}"
        edf_path = patient_raw_dir / filename
        
        # Download EDF
        if download_file(edf_url, edf_path):
            # Convert to CSV
            csv_filename = filename.replace('.edf', '_labeled.csv')
            csv_path = patient_output_dir / csv_filename
            
            if csv_path.exists():
                print(f"  [SKIP] CSV already exists: {csv_filename}")
                converted_count += 1
            else:
                seizure_info = seizure_annotations.get(filename, None)
                if convert_edf_to_csv(edf_path, csv_path, seizure_info):
                    converted_count += 1
    
    return converted_count


def main():
    print("="*60)
    print("CHB-MIT Dataset Downloader and Converter")
    print("="*60)
    
    # Check dependencies
    if not MNE_AVAILABLE:
        print("\n[!] Please install mne library first:")
        print("    pip install mne")
        sys.exit(1)
    
    # Create directories
    RAW_DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Download RECORDS file if not exists
    records_path = RAW_DATA_DIR / "RECORDS.txt"
    if not records_path.exists():
        print("\nDownloading file list...")
        download_file(f"{BASE_URL}/RECORDS", records_path)
    
    # Process each patient
    total_converted = 0
    for patient in PATIENTS_TO_DOWNLOAD:
        try:
            count = download_patient(patient)
            total_converted += count
        except Exception as e:
            print(f"[ERROR] Failed to process {patient}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"Total files converted: {total_converted}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Show what's in the output directory
    print("\nData available:")
    for patient_dir in sorted(OUTPUT_DIR.iterdir()):
        if patient_dir.is_dir():
            csv_files = list(patient_dir.glob("*.csv"))
            seizure_files = 0
            for csv_file in csv_files:
                # Quick check for seizures
                with open(csv_file, 'r') as f:
                    first_lines = f.read(10000)
                    if ',1\n' in first_lines or ',1' in first_lines.split('\n')[-1]:
                        seizure_files += 1
            print(f"  {patient_dir.name}: {len(csv_files)} files ({seizure_files} with seizures)")


if __name__ == "__main__":
    main()
