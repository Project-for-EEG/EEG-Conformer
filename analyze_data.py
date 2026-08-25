import pandas as pd
from pathlib import Path

siena_dir = Path('EEG_Data_Siena')
csv_files = list(siena_dir.rglob('*_labeled.csv'))

print('='*70)
print('SIENA DATASET ANALYSIS')
print('='*70)

total_samples = 0
total_seizure = 0
files_with_seizure = 0
files_no_seizure = 0
patient_stats = {}

for f in sorted(csv_files):
    try:
        df = pd.read_csv(f, usecols=['Seizure'])
        samples = len(df)
        seizure = int(df['Seizure'].sum())
        total_samples += samples
        total_seizure += seizure
        
        patient = f.parent.name
        if patient not in patient_stats:
            patient_stats[patient] = {'files': 0, 'samples': 0, 'seizure': 0}
        patient_stats[patient]['files'] += 1
        patient_stats[patient]['samples'] += samples
        patient_stats[patient]['seizure'] += seizure
        
        if seizure > 0:
            files_with_seizure += 1
        else:
            files_no_seizure += 1
    except Exception as e:
        print(f'  Error with {f.name}: {e}')

print()
print('Per-Patient Summary:')
print('-'*70)
print(f"{'Patient':<10} {'Files':<8} {'Samples':<15} {'Seizure':<15} {'%':<8}")
print('-'*70)
for p in sorted(patient_stats.keys()):
    s = patient_stats[p]
    pct = 100 * s['seizure'] / s['samples'] if s['samples'] > 0 else 0
    print(f"{p:<10} {s['files']:<8} {s['samples']:<15,} {s['seizure']:<15,} {pct:<8.2f}")

print()
print('='*70)
print(f'Total Siena files:     {len(csv_files)}')
print(f'Files with seizure:    {files_with_seizure}')
print(f'Files without seizure: {files_no_seizure}')
print(f'Total samples:         {total_samples:,}')
print(f'Total seizure samples: {total_seizure:,}')
print(f'Seizure percentage:    {100*total_seizure/total_samples:.2f}%')
print(f'Seizure duration:      {total_seizure/256/60:.1f} minutes ({total_seizure/256/3600:.2f} hours)')
print('='*70)
