"""
Download and Process Additional EEG Seizure Datasets

This script downloads:
1. Siena Scalp EEG Database (PhysioNet) - 47 seizures, immediate access
2. TUH EEG Seizure Corpus (TUSZ) - 4,029 seizures, requires registration

After downloading, it processes the data to match your existing CHB-MIT format.
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# Check for required packages
try:
    import mne
except ImportError:
    print("Installing mne...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mne"])
    import mne

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


# ============================================================================
# SIENA DATASET DOWNLOAD AND PROCESSING
# ============================================================================

def download_siena_dataset(output_dir: str = "additional_data/siena"):
    """
    Download Siena Scalp EEG Database from PhysioNet
    
    This dataset contains:
    - 14 patients
    - 47 seizures
    - 128 hours of recording
    - 512 Hz sampling rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://physionet.org/files/siena-scalp-eeg/1.0.0/"
    
    print("=" * 60)
    print("DOWNLOADING SIENA SCALP EEG DATABASE")
    print("=" * 60)
    print(f"Source: {base_url}")
    print(f"Output: {output_dir}")
    print()
    
    # List of patient folders (PN00-PN14, some may be missing)
    patients = [f"PN{i:02d}" for i in range(15)]
    
    # Download patient list first
    try:
        # Download the RECORDS file to get exact file list
        records_url = base_url + "RECORDS"
        print(f"Fetching file list from {records_url}...")
        
        response = requests.get(records_url, timeout=30)
        if response.status_code == 200:
            records = response.text.strip().split('\n')
            print(f"Found {len(records)} files to download")
        else:
            print(f"Could not fetch RECORDS, using default patient list")
            records = None
    except Exception as e:
        print(f"Error fetching records: {e}")
        records = None
    
    # Download each file
    downloaded_files = []
    
    if records:
        for record in records:
            if not record.strip():
                continue
            
            file_url = base_url + record
            local_path = output_dir / record
            
            # Create subdirectory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if local_path.exists():
                print(f"  Already exists: {record}")
                downloaded_files.append(local_path)
                continue
            
            try:
                print(f"  Downloading: {record}...", end=" ")
                response = requests.get(file_url, timeout=120)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    print("OK")
                    downloaded_files.append(local_path)
                else:
                    print(f"Failed ({response.status_code})")
            except Exception as e:
                print(f"Error: {e}")
    
    print(f"\nDownloaded {len(downloaded_files)} files")
    return output_dir


def parse_siena_annotations(annotation_file: Path) -> List[Tuple[float, float]]:
    """Parse Siena annotation file to get seizure times"""
    seizures = []
    
    if not annotation_file.exists():
        return seizures
    
    with open(annotation_file, 'r') as f:
        content = f.read()
    
    # Parse seizure start and end times
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if 'seizure' in line.lower() or 'start' in line.lower():
            # Try to extract times
            parts = line.split()
            for i, part in enumerate(parts):
                try:
                    time = float(part)
                    if i + 1 < len(parts):
                        end_time = float(parts[i + 1])
                        seizures.append((time, end_time))
                        break
                except ValueError:
                    continue
    
    return seizures


def process_siena_to_csv(siena_dir: str, output_dir: str = "EEG_Data_Siena"):
    """
    Process Siena EDF files to CSV format matching CHB-MIT structure
    """
    siena_dir = Path(siena_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("PROCESSING SIENA DATA TO CSV")
    print("=" * 60)
    
    # Find all EDF files
    edf_files = list(siena_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")
    
    # Target channels (matching CHB-MIT 23 channels)
    target_channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ',
        'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
    ]
    
    # Alternative channel names that might be in Siena
    alt_channels = [
        'Fp1', 'F7', 'T3', 'T5', 'O1',
        'Fp1', 'F3', 'C3', 'P3', 'O1',
        'Fp2', 'F4', 'C4', 'P4', 'O2',
        'Fp2', 'F8', 'T4', 'T6', 'O2',
        'Fz', 'Cz', 'Pz'
    ]
    
    processed_count = 0
    total_seizure_segments = 0
    
    for edf_file in edf_files:
        try:
            print(f"\nProcessing: {edf_file.name}")
            
            # Load EDF
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Get channel names
            ch_names = raw.ch_names
            print(f"  Channels: {len(ch_names)}")
            
            # Select EEG channels only
            eeg_channels = [ch for ch in ch_names if 'EEG' in ch.upper() or 
                           any(c in ch.upper() for c in ['FP', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'FZ', 'CZ', 'PZ', 'T3', 'T4', 'T5', 'T6', 'F7', 'F8'])]
            
            if len(eeg_channels) < 10:
                # Try all channels except ECG
                eeg_channels = [ch for ch in ch_names if 'ECG' not in ch.upper() and 'EKG' not in ch.upper()]
            
            print(f"  EEG channels: {len(eeg_channels)}")
            
            if len(eeg_channels) < 10:
                print(f"  Skipping (too few channels)")
                continue
            
            # Pick channels
            raw.pick_channels(eeg_channels[:23])  # Take first 23 channels
            
            # Resample to 256 Hz if needed
            if raw.info['sfreq'] != 256:
                print(f"  Resampling from {raw.info['sfreq']} to 256 Hz")
                raw.resample(256)
            
            # Get data
            data = raw.get_data()  # Shape: (channels, samples)
            sfreq = raw.info['sfreq']
            
            # Look for annotation file
            annotation_file = edf_file.with_suffix('.txt')
            if not annotation_file.exists():
                # Try in parent directory
                annotation_file = edf_file.parent / (edf_file.stem + '-summary.txt')
            
            # Parse seizure annotations
            seizures = []
            
            # Check MNE annotations
            if raw.annotations:
                for annot in raw.annotations:
                    desc = annot['description'].lower()
                    if 'seizure' in desc or 'sz' in desc:
                        start = annot['onset']
                        duration = annot['duration']
                        seizures.append((start, start + duration))
            
            # Create labels
            n_samples = data.shape[1]
            labels = np.zeros(n_samples, dtype=int)
            
            for start, end in seizures:
                start_idx = int(start * sfreq)
                end_idx = int(end * sfreq)
                labels[start_idx:min(end_idx, n_samples)] = 1
            
            seizure_samples = np.sum(labels)
            total_seizure_segments += seizure_samples
            
            print(f"  Samples: {n_samples}, Seizure: {seizure_samples} ({100*seizure_samples/n_samples:.2f}%)")
            
            # Create DataFrame
            time = np.arange(n_samples) / sfreq
            
            df_data = {'Time (s)': time}
            for i, ch in enumerate(raw.ch_names):
                # Clean channel name
                clean_name = ch.replace('EEG ', '').replace('-REF', '').strip()
                df_data[clean_name] = data[i]
            df_data['Seizure'] = labels
            
            df = pd.DataFrame(df_data)
            
            # Save to CSV
            patient_dir = output_dir / edf_file.parent.name
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = patient_dir / f"{edf_file.stem}_labeled.csv"
            df.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"  Files processed: {processed_count}")
    print(f"  Total seizure samples: {total_seizure_segments}")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")
    
    return output_dir


# ============================================================================
# TUSZ DATASET INSTRUCTIONS
# ============================================================================

def print_tusz_instructions():
    """Print instructions for downloading TUSZ dataset"""
    print("""
================================================================================
TUH EEG SEIZURE CORPUS (TUSZ) - DOWNLOAD INSTRUCTIONS
================================================================================

TUSZ contains 4,029 seizures - this will greatly improve your model!

STEP 1: Register for Access
--------------------------
1. Go to: https://isip.piconepress.com/projects/tuh_eeg/html/request_access.php
2. Fill out the registration form
3. Download and sign the data use agreement
4. Email the signed form to: help@nedcdata.org
   Subject: "Download The TUH EEG Corpus"
5. Wait ~24 hours for credentials

STEP 2: Download the Data
-------------------------
Once you receive credentials, you can download using rsync:

    rsync -auxvL --progress nedc_tuh_eeg@www.isip.piconepress.com:~/data/tuh_eeg_seizure/ ./tusz_data/

Or download specific files through the web interface.

STEP 3: Process the Data
------------------------
After downloading, run this script again with:

    python download_additional_data.py --process-tusz ./tusz_data

The TUSZ dataset uses the same EDF format as CHB-MIT, so processing is straightforward.

================================================================================
""")


def process_tusz_to_csv(tusz_dir: str, output_dir: str = "EEG_Data_TUSZ"):
    """
    Process TUSZ EDF files to CSV format
    
    TUSZ structure:
    - edf/: Contains EDF files organized by version/split/patient
    - csv/: Contains seizure annotations
    """
    tusz_dir = Path(tusz_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("PROCESSING TUSZ DATA TO CSV")
    print("=" * 60)
    
    # Find EDF files
    edf_files = list(tusz_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files")
    
    if len(edf_files) == 0:
        print("No EDF files found. Make sure you've downloaded the TUSZ dataset.")
        return None
    
    # Process each file (limit for testing)
    processed_count = 0
    max_files = 100  # Process in batches
    
    for edf_file in edf_files[:max_files]:
        try:
            print(f"\nProcessing: {edf_file.name}")
            
            # Find corresponding annotation file
            csv_file = edf_file.with_suffix('.csv')
            if not csv_file.exists():
                # Try looking in parallel csv directory
                csv_path = str(edf_file).replace('/edf/', '/csv/').replace('\\edf\\', '\\csv\\')
                csv_file = Path(csv_path).with_suffix('.csv')
            
            # Load EDF
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Select EEG channels
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=['bads'])
            if len(eeg_picks) > 23:
                eeg_picks = eeg_picks[:23]
            
            raw.pick(eeg_picks)
            
            # Resample to 256 Hz
            if raw.info['sfreq'] != 256:
                raw.resample(256)
            
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            n_samples = data.shape[1]
            
            # Parse annotations
            labels = np.zeros(n_samples, dtype=int)
            
            if csv_file.exists():
                # TUSZ CSV format: channel,start_time,stop_time,label,probability
                annot_df = pd.read_csv(csv_file, comment='#')
                for _, row in annot_df.iterrows():
                    if 'seiz' in str(row.get('label', '')).lower():
                        start = int(float(row['start_time']) * sfreq)
                        stop = int(float(row['stop_time']) * sfreq)
                        labels[start:min(stop, n_samples)] = 1
            
            seizure_pct = 100 * np.mean(labels)
            print(f"  Channels: {len(raw.ch_names)}, Seizure: {seizure_pct:.2f}%")
            
            # Create DataFrame
            time = np.arange(n_samples) / sfreq
            df_data = {'Time (s)': time}
            for i, ch in enumerate(raw.ch_names):
                df_data[ch] = data[i]
            df_data['Seizure'] = labels
            
            df = pd.DataFrame(df_data)
            
            # Save
            patient_id = edf_file.parent.name
            patient_dir = output_dir / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = patient_dir / f"{edf_file.stem}_labeled.csv"
            df.to_csv(output_file, index=False)
            
            processed_count += 1
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nProcessed {processed_count} files to {output_dir}")
    return output_dir


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process additional seizure datasets")
    parser.add_argument('--siena', action='store_true', help='Download and process Siena dataset')
    parser.add_argument('--tusz-instructions', action='store_true', help='Show TUSZ download instructions')
    parser.add_argument('--process-tusz', type=str, help='Process downloaded TUSZ data from specified directory')
    parser.add_argument('--all', action='store_true', help='Download all available datasets')
    
    args = parser.parse_args()
    
    if args.tusz_instructions:
        print_tusz_instructions()
        return
    
    if args.process_tusz:
        process_tusz_to_csv(args.process_tusz)
        return
    
    if args.siena or args.all or len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print("ADDITIONAL SEIZURE DATA DOWNLOAD")
        print("=" * 60)
        print("\nThis will download the Siena Scalp EEG Database (47 seizures)")
        print("For TUSZ (4,029 seizures), run with --tusz-instructions\n")
        
        # Download Siena
        siena_dir = download_siena_dataset()
        
        # Process to CSV
        process_siena_to_csv(siena_dir)
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("""
1. The Siena data has been saved to EEG_Data_Siena/

2. To use this data with your model, update config.py:
   data_dir: str = "EEG_Data"  # Change to include new data
   
   Or merge the directories:
   - Copy EEG_Data_Siena/* to EEG_Data/

3. For 20x more seizures, register for TUSZ:
   python download_additional_data.py --tusz-instructions
""")


if __name__ == "__main__":
    main()
