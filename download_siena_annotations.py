"""
Download Siena Seizure Annotations from PhysioNet

The seizure annotations are in text files that need to be downloaded separately.
"""
import requests
from pathlib import Path

def download_file(url, local_path, timeout=60):
    """Download a file from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        pass
    return False

def main():
    base_url = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
    output_dir = Path("additional_data/siena")
    
    print("=" * 60)
    print("DOWNLOADING SIENA SEIZURE ANNOTATIONS")
    print("=" * 60)
    
    # Known annotation files in Siena dataset
    # These contain seizure start/end times
    annotation_files = {
        "PN00": ["PN00-Seizures-list-grouped.txt"],
        "PN01": ["PN01-Seizures-list-grouped.txt"],
        "PN03": ["PN03-Seizures-list-grouped.txt"],
        "PN05": ["PN05-Seizures-list-grouped.txt"],
        "PN06": ["PN06-Seizures-list-grouped.txt"],
        "PN07": ["PN07-Seizures-list-grouped.txt"],
        "PN09": ["PN09-Seizures-list-grouped.txt"],
        "PN10": ["PN10-Seizures-list-grouped.txt"],
        "PN11": ["PN11-Seizures-list-grouped.txt"],
        "PN12": ["PN12-Seizures-list-grouped.txt"],
        "PN13": ["PN13-Seizures-list-grouped.txt"],
        "PN14": ["PN14-Seizures-list-grouped.txt"],
    }
    
    downloaded = 0
    
    for patient, files in annotation_files.items():
        patient_dir = output_dir / patient
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in files:
            url = f"{base_url}/{patient}/{fname}"
            local_path = patient_dir / fname
            
            if local_path.exists():
                print(f"  Already exists: {patient}/{fname}")
                downloaded += 1
                continue
            
            print(f"  Downloading: {patient}/{fname}...", end=" ")
            if download_file(url, local_path):
                print("OK")
                downloaded += 1
                
                # Show content
                with open(local_path, 'r') as f:
                    content = f.read()
                if 'seizure' in content.lower():
                    print(f"    Contains seizure annotations!")
            else:
                print("Not found or failed")
    
    # Also try to download the summary file
    summary_url = f"{base_url}/RECORDS-seizures"
    summary_path = output_dir / "RECORDS-seizures"
    if not summary_path.exists():
        print(f"\nDownloading seizure records list...")
        download_file(summary_url, summary_path)
    
    print(f"\nDownloaded {downloaded} annotation files")
    
    # Show what we have
    print("\n" + "=" * 60)
    print("CHECKING DOWNLOADED ANNOTATIONS")
    print("=" * 60)
    
    for patient_dir in sorted(output_dir.iterdir()):
        if patient_dir.is_dir():
            txt_files = list(patient_dir.glob("*.txt"))
            edf_files = list(patient_dir.glob("*.edf"))
            print(f"{patient_dir.name}: {len(edf_files)} EDF files, {len(txt_files)} annotation files")
            
            for txt in txt_files:
                with open(txt, 'r') as f:
                    content = f.read()
                # Count seizures mentioned
                seizure_count = content.lower().count('seizure')
                if seizure_count > 0:
                    print(f"  -> {txt.name}: {seizure_count} seizure mentions")


if __name__ == "__main__":
    main()
