"""
Quick Threshold Analysis - Uses existing model to find optimal threshold
"""
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import get_config
from model import create_model


def main():
    print("="*60)
    print("QUICK THRESHOLD OPTIMIZATION")
    print("="*60)
    
    config = get_config()
    
    # Load model
    model = create_model(config.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_path = Path(config.training.save_dir) / 'best.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {best_path}")
    
    model = model.to(device)
    model.eval()
    
    # Load preprocessed data directly
    print("\nLoading preprocessed data...")
    data_dir = Path('preprocessed_data')
    
    all_data = []
    all_labels = []
    
    # Search recursively in subdirectories
    npz_files = list(data_dir.glob('**/*.npz'))
    print(f"Found {len(npz_files)} preprocessed files")
    
    # Use subset for quick analysis (every 3rd file to get variety)
    subset_files = npz_files[::3][:20]
    print(f"Using {len(subset_files)} files for quick analysis")
    
    for f in tqdm(subset_files, desc="Loading"):
        data = np.load(f)
        all_data.append(data['segments'])
        all_labels.append(data['labels'])
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"\nLoaded {len(y)} samples")
    print(f"  Non-seizure: {np.sum(y == 0)}")
    print(f"  Seizure: {np.sum(y == 1)}")
    
    # Get predictions
    print("\nRunning inference...")
    all_probs = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size)):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
    
    probs = np.array(all_probs)
    labels = y
    
    # Analyze thresholds
    print("\n" + "-"*60)
    print("THRESHOLD ANALYSIS")
    print("-"*60)
    
    print(f"\n{'Threshold':<12} {'Sens':<10} {'Spec':<10} {'Prec':<10} {'F1':<10} {'Missed':<10}")
    print("-"*60)
    
    best_f1 = 0
    best_thresh_f1 = 0.5
    best_for_90_sens = None
    
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (probs >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        
        print(f"{thresh:<12.1f} {sens:<10.2%} {spec:<10.2%} {prec:<10.2%} {f1:<10.3f} {fn:<10d}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = thresh
        
        if sens >= 0.90 and best_for_90_sens is None:
            best_for_90_sens = thresh
    
    # Fine-grained search
    print("\n" + "-"*60)
    print("FINE-GRAINED SEARCH (for ≥90% sensitivity)")
    print("-"*60)
    
    for thresh in np.arange(0.05, 0.5, 0.05):
        preds = (probs >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        fa_rate = (fp * 4 / 3600) if (tn + fp) > 0 else 0  # per hour
        
        marker = " ← RECOMMENDED" if sens >= 0.90 and (best_for_90_sens is None or thresh >= best_for_90_sens - 0.05) else ""
        print(f"t={thresh:.2f}: Sens={sens:.1%}, Spec={spec:.1%}, F1={f1:.3f}, FA/hr={fa_rate:.1f}{marker}")
        
        if sens >= 0.90 and best_for_90_sens is None:
            best_for_90_sens = thresh
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"""
Current threshold: 0.5
Best F1 threshold: {best_thresh_f1}
Best for ≥90% sensitivity: {best_for_90_sens if best_for_90_sens else 'Need lower threshold'}

To use the optimized threshold in your code:
    pred = 1 if prob > {best_for_90_sens or 0.3} else 0
""")


if __name__ == "__main__":
    main()
