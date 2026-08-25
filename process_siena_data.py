"""
Process Siena EDF Files to CSV Format

This script converts downloaded Siena EDF files to the same CSV format
as your CHB-MIT data, making it easy to combine datasets.

Usage:
    python process_siena_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import re

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("Installing mne...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mne"])
    import mne
    mne.set_log_level('ERROR')


def time_str_to_seconds(time_str: str) -> float:
    """Convert time string (HH.MM.SS or HH:MM:SS) to seconds since midnight"""
    time_str = time_str.strip().replace(':', '.')
    parts = time_str.split('.')
    
    if len(parts) >= 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    else:
        return float(time_str)


def parse_siena_annotations(patient_dir: Path) -> dict:
    """
    Parse Siena seizure annotation files
    
    Returns dict mapping EDF filename to list of (start_sec, duration_sec) tuples
    where start_sec is relative to the start of the recording
    """
    seizures_by_file = {}
    
    # Find annotation file
    annot_files = list(patient_dir.glob("Seizures-list-*.txt"))
    
    for annot_file in annot_files:
        print(f"  Parsing annotations: {annot_file.name}")
        
        with open(annot_file, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        current_file = None
        reg_start_time = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for file name
            if 'File name:' in line:
                file_match = re.search(r'(PN\d+[-_]?\d*\.edf)', line, re.IGNORECASE)
                if file_match:
                    current_file = file_match.group(1)
                    if current_file not in seizures_by_file:
                        seizures_by_file[current_file] = {'reg_start': None, 'seizures': []}
            
            # Look for registration start time
            if 'Registration start time:' in line:
                time_match = re.search(r':\s*([\d.:]+)', line)
                if time_match and current_file:
                    reg_start_time = time_str_to_seconds(time_match.group(1))
                    seizures_by_file[current_file]['reg_start'] = reg_start_time
            
            # Look for seizure start/end times
            if 'Seizure start time:' in line or 'Start time:' in line:
                time_match = re.search(r'(?:start time|Start time):\s*([\d.:]+)', line, re.IGNORECASE)
                if time_match:
                    start_time = time_str_to_seconds(time_match.group(1))
                    
                    # Look for end time in next line or same line
                    end_time = None
                    for j in range(i, min(i+3, len(lines))):
                        if 'end time' in lines[j].lower():
                            end_match = re.search(r'(?:end time|End time):\s*([\d.:]+)', lines[j], re.IGNORECASE)
                            if end_match:
                                end_time = time_str_to_seconds(end_match.group(1))
                                break
                    
                    if end_time and current_file:
                        # Calculate times relative to registration start
                        reg_start = seizures_by_file[current_file].get('reg_start', 0) or 0
                        
                        # Handle day wraparound
                        if start_time < reg_start:
                            start_time += 24 * 3600
                        if end_time < start_time:
                            end_time += 24 * 3600
                        
                        # Convert to seconds from recording start
                        seizure_start = start_time - reg_start
                        seizure_end = end_time - reg_start
                        
                        # Sanity check
                        if seizure_start >= 0 and seizure_end > seizure_start:
                            seizures_by_file[current_file]['seizures'].append((seizure_start, seizure_end))
                            print(f"    Found seizure in {current_file}: {seizure_start:.1f}s - {seizure_end:.1f}s")
            
            i += 1
    
    # Convert to simple format
    result = {}
    for fname, data in seizures_by_file.items():
        if data['seizures']:
            result[fname] = data['seizures']
    
    return result


def process_edf_to_csv(edf_file: Path, output_dir: Path, seizure_times: list = None) -> bool:
    """
    Convert a single EDF file to CSV format matching CHB-MIT structure
    """
    try:
        print(f"  Processing: {edf_file.name}")
        
        # Load EDF file
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        
        # Get info
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        
        # Filter to EEG channels only
        eeg_channels = []
        for ch in ch_names:
            ch_upper = ch.upper()
            # Include if it looks like an EEG channel
            if any(x in ch_upper for x in ['FP', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'CZ',
                                            'P3', 'P4', 'PZ', 'O1', 'O2', 'T3', 'T4', 'T5', 'T6',
                                            'FZ', 'A1', 'A2', 'T7', 'T8', 'P7', 'P8', 'FC', 'CP']):
                if 'ECG' not in ch_upper and 'EKG' not in ch_upper:
                    eeg_channels.append(ch)
        
        if len(eeg_channels) < 10:
            eeg_channels = [ch for ch in ch_names if 'ECG' not in ch.upper() and 'EKG' not in ch.upper() and 'EMG' not in ch.upper()]
        
        # Limit to 23 channels (matching CHB-MIT)
        if len(eeg_channels) > 23:
            eeg_channels = eeg_channels[:23]
        
        print(f"    Using {len(eeg_channels)} channels (sfreq={sfreq}Hz)")
        
        # Pick channels
        raw.pick_channels(eeg_channels)
        
        # Resample to 256 Hz if needed
        if sfreq != 256:
            print(f"    Resampling from {sfreq} to 256 Hz")
            raw.resample(256)
            sfreq = 256
        
        # Get data
        data = raw.get_data()
        n_channels, n_samples = data.shape
        duration = n_samples / sfreq
        
        # Create labels
        labels = np.zeros(n_samples, dtype=int)
        
        if seizure_times:
            for start_sec, end_sec in seizure_times:
                start_idx = int(start_sec * sfreq)
                end_idx = int(end_sec * sfreq)
                if start_idx < n_samples and end_idx > 0:
                    labels[max(0, start_idx):min(end_idx, n_samples)] = 1
            
            seizure_samples = np.sum(labels)
            seizure_pct = 100 * seizure_samples / n_samples
            print(f"    Duration: {duration:.1f}s, Seizure: {seizure_samples} samples ({seizure_pct:.2f}%)")
        else:
            print(f"    Duration: {duration:.1f}s, No seizure annotations")
        
        # Create time array
        time = np.arange(n_samples) / sfreq
        
        # Build DataFrame
        df_data = {'Time (s)': time}
        
        for i, ch in enumerate(raw.ch_names):
            clean_name = ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip()
            df_data[clean_name] = data[i]
        
        df_data['Seizure'] = labels
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        output_file = output_dir / f"{edf_file.stem}_labeled.csv"
        df.to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")
        
        return True, np.sum(labels)
        
    except Exception as e:
        print(f"    Error: {e}")
        return False, 0


def main():
    siena_dir = Path("additional_data/siena")
    output_dir = Path("EEG_Data_Siena")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PROCESSING SIENA EEG DATA")
    print("=" * 60)
    print(f"Input:  {siena_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Find all EDF files
    edf_files = list(siena_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")
    
    if len(edf_files) == 0:
        print("\nNo EDF files found. Run the download script first:")
        print("  . .\\download_siena.ps1")
        return
    
    # Process each patient
    processed = 0
    failed = 0
    total_seizure_samples = 0
    
    # Group by patient
    patients = {}
    for edf in edf_files:
        patient = edf.parent.name
        if patient not in patients:
            patients[patient] = []
        patients[patient].append(edf)
    
    print(f"Found {len(patients)} patients")
    print()
    
    for patient, files in sorted(patients.items()):
        print(f"\nPatient: {patient} ({len(files)} files)")
        
        # Create patient output directory
        patient_output = output_dir / patient
        patient_output.mkdir(parents=True, exist_ok=True)
        
        # Parse seizure annotations for this patient
        seizure_annotations = parse_siena_annotations(files[0].parent)
        
        for edf_file in sorted(files):
            # Get seizure times for this file
            seizures = seizure_annotations.get(edf_file.name, [])
            
            success, seizure_count = process_edf_to_csv(edf_file, patient_output, seizures)
            if success:
                processed += 1
                total_seizure_samples += seizure_count
            else:
                failed += 1
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {processed} files")
    print(f"Failed: {failed} files")
    print(f"Total seizure samples: {total_seizure_samples:,}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Count seizure files
    seizure_files = 0
    for csv_file in output_dir.rglob("*.csv"):
        df = pd.read_csv(csv_file, nrows=1000)
        if 'Seizure' in df.columns and df['Seizure'].sum() > 0:
            seizure_files += 1
    
    print(f"Files with seizures: {seizure_files}")
    print()
    print("NEXT STEPS:")
    print("-" * 40)
    print("1. Copy the Siena data to your EEG_Data folder:")
    print("   xcopy EEG_Data_Siena\\* EEG_Data\\ /E /I /Y")
    print()
    print("2. Retrain your model with the combined dataset")


if __name__ == "__main__":
    main()
