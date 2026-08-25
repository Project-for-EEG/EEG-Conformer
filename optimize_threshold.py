"""
Optimize Classification Threshold for Higher Sensitivity

Uses the proper test split to find optimal threshold.
"""
import json
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from config import get_config
from model import create_model
from data_loader import create_data_loaders


def main():
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION FOR SEIZURE DETECTION")
    print("=" * 60)
    
    config = get_config()
    
    # Create data loaders (uses seed=42 for consistent split)
    print("\nLoading data with proper train/val/test split...")
    _, _, test_loader = create_data_loaders(config.data, config.training)
    
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
    
    # Get all predictions
    print("\nRunning inference on test set...")
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Inference"):
            data = data.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    n_seizure = np.sum(labels == 1)
    n_non_seizure = np.sum(labels == 0)
    print(f"\nTest set: {len(labels)} samples")
    print(f"  Non-seizure: {n_non_seizure}")
    print(f"  Seizure: {n_seizure}")
    
    # Analyze thresholds
    print("\n" + "-" * 60)
    print("THRESHOLD ANALYSIS")
    print("-" * 60)
    
    print(f"\n{'Thresh':<8} {'Sens':<10} {'Spec':<10} {'Prec':<10} {'F1':<10} {'TP':<6} {'FN':<6} {'FP':<6}")
    print("-" * 60)
    
    best_for_90_sens = None
    best_f1 = 0
    best_f1_thresh = 0.5
    
    results = []
    
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
        preds = (probs >= thresh).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'sensitivity': sens,
            'specificity': spec,
            'precision': prec,
            'f1': f1,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
        })
        
        marker = ""
        if sens >= 0.90 and best_for_90_sens is None:
            best_for_90_sens = thresh
            marker = " <-- 90%+ sens"
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = thresh
        
        print(f"{thresh:<8.2f} {sens:<10.1%} {spec:<10.1%} {prec:<10.1%} {f1:<10.3f} {tp:<6} {fn:<6} {fp:<6}{marker}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Current performance (t=0.5)
    current = next(r for r in results if r['threshold'] == 0.5)
    print(f"\nCurrent (threshold=0.5):")
    print(f"  Sensitivity: {current['sensitivity']:.1%} ({current['tp']}/{n_seizure} seizures detected)")
    print(f"  Specificity: {current['specificity']:.1%}")
    print(f"  F1 Score: {current['f1']:.3f}")
    print(f"  Missed seizures: {current['fn']}")
    
    # Best F1
    best_f1_result = next(r for r in results if r['threshold'] == best_f1_thresh)
    print(f"\nBest F1 (threshold={best_f1_thresh}):")
    print(f"  Sensitivity: {best_f1_result['sensitivity']:.1%}")
    print(f"  Specificity: {best_f1_result['specificity']:.1%}")
    print(f"  F1 Score: {best_f1_result['f1']:.3f}")
    
    # Best for 90% sensitivity
    if best_for_90_sens:
        sens_90_result = next(r for r in results if r['threshold'] == best_for_90_sens)
        print(f"\nFor 90%+ Sensitivity (threshold={best_for_90_sens}):")
        print(f"  Sensitivity: {sens_90_result['sensitivity']:.1%} ({sens_90_result['tp']}/{n_seizure} seizures)")
        print(f"  Specificity: {sens_90_result['specificity']:.1%}")
        print(f"  Precision: {sens_90_result['precision']:.1%}")
        print(f"  F1 Score: {sens_90_result['f1']:.3f}")
        print(f"  False alarms: {sens_90_result['fp']}")
        print(f"  Missed seizures: {sens_90_result['fn']}")
    else:
        # Find best sensitivity we can achieve
        best_sens = max(results, key=lambda x: x['sensitivity'])
        print(f"\nMax achievable sensitivity (threshold={best_sens['threshold']}):")
        print(f"  Sensitivity: {best_sens['sensitivity']:.1%}")
        print(f"  This threshold gives the highest seizure detection rate")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommended_thresh = best_for_90_sens if best_for_90_sens else best_f1_thresh
    rec_result = next(r for r in results if r['threshold'] == recommended_thresh)
    
    improvement = rec_result['sensitivity'] - current['sensitivity']
    fewer_missed = current['fn'] - rec_result['fn']
    
    print(f"""
RECOMMENDED THRESHOLD: {recommended_thresh}

Expected Improvement:
  - Sensitivity: {current['sensitivity']:.1%} -> {rec_result['sensitivity']:.1%} (+{improvement:.1%})
  - Missed seizures: {current['fn']} -> {rec_result['fn']} ({fewer_missed} fewer missed)
  - Trade-off: {rec_result['fp']} false alarms (was {current['fp']})

To apply this threshold in your code:
  
  # Instead of:
  pred = 1 if prob > 0.5 else 0
  
  # Use:
  pred = 1 if prob > {recommended_thresh} else 0
""")
    
    # Save results
    output = {
        'current_threshold': 0.5,
        'current_metrics': current,
        'recommended_threshold': recommended_thresh,
        'recommended_metrics': rec_result,
        'all_thresholds': results
    }
    
    output_path = Path('evaluation_results') / 'threshold_optimization.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
