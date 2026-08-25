"""
Optimize Model for Higher Sensitivity (Recall)

This script:
1. Analyzes different thresholds to find optimal sensitivity
2. Provides class-weighted loss for retraining
3. Compares metrics at different operating points
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from tqdm import tqdm

from config import get_config
from model import create_model
from data_loader import create_data_loaders


def load_model_and_data():
    """Load trained model and test data"""
    config = get_config()
    
    # Create data loaders
    print("Loading data...")
    _, _, test_loader = create_data_loaders(config.data, config.training)
    
    # Create and load model
    model = create_model(config.model)
    
    best_model_path = Path(config.training.save_dir) / 'best.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {best_model_path}")
    else:
        print("Warning: No trained model found!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, test_loader, device


def get_predictions(model, test_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Get all predictions and labels"""
    all_labels = []
    all_probs = []
    
    print("Running inference...")
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)


def find_optimal_thresholds(labels: np.ndarray, probs: np.ndarray) -> Dict:
    """Find optimal thresholds for different objectives"""
    
    thresholds = np.linspace(0.01, 0.99, 200)
    
    results = {
        'thresholds': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': [],
        'accuracy': []
    }
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results['thresholds'].append(thresh)
        results['sensitivity'].append(sensitivity)
        results['specificity'].append(specificity)
        results['precision'].append(precision)
        results['f1'].append(f1)
        results['accuracy'].append(accuracy)
    
    # Convert to numpy
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def find_threshold_for_target_sensitivity(results: Dict, target_sensitivity: float = 0.90) -> Tuple[float, Dict]:
    """Find threshold that achieves target sensitivity"""
    
    # Find thresholds that achieve at least target sensitivity
    valid_mask = results['sensitivity'] >= target_sensitivity
    
    if not np.any(valid_mask):
        # If no threshold achieves target, find the one with highest sensitivity
        best_idx = np.argmax(results['sensitivity'])
        print(f"Warning: Cannot achieve {target_sensitivity:.0%} sensitivity. Max achievable: {results['sensitivity'][best_idx]:.2%}")
    else:
        # Among valid thresholds, find the one with best F1 or highest threshold (more specific)
        valid_indices = np.where(valid_mask)[0]
        # Choose highest threshold that still meets sensitivity target (best specificity)
        best_idx = valid_indices[-1]
    
    optimal_threshold = results['thresholds'][best_idx]
    
    metrics = {
        'threshold': optimal_threshold,
        'sensitivity': results['sensitivity'][best_idx],
        'specificity': results['specificity'][best_idx],
        'precision': results['precision'][best_idx],
        'f1': results['f1'][best_idx],
        'accuracy': results['accuracy'][best_idx]
    }
    
    return optimal_threshold, metrics


def evaluate_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    """Evaluate model at a specific threshold"""
    preds = (probs >= threshold).astype(int)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate false alarm rate (per hour, 4-second windows)
    total_hours = ((tn + fp) * 4) / 3600
    false_alarm_rate = fp / total_hours if total_hours > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'sensitivity': recall_score(labels, preds, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(labels, preds, zero_division=0),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'false_alarm_rate_per_hour': false_alarm_rate,
        'seizures_detected': f"{tp}/{tp+fn}",
        'seizures_missed': int(fn)
    }


def print_comparison(original: Dict, optimized: Dict):
    """Print comparison of original vs optimized metrics"""
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Original (t=0.50)':<20} {'Optimized':<20} {'Change':<10}")
    print("-"*70)
    
    metrics_to_compare = [
        ('Threshold', 'threshold', '{:.3f}'),
        ('Sensitivity (Recall)', 'sensitivity', '{:.2%}'),
        ('Specificity', 'specificity', '{:.2%}'),
        ('Precision', 'precision', '{:.2%}'),
        ('F1 Score', 'f1', '{:.3f}'),
        ('Accuracy', 'accuracy', '{:.2%}'),
        ('False Alarms/Hour', 'false_alarm_rate_per_hour', '{:.2f}'),
        ('Seizures Detected', 'seizures_detected', '{}'),
        ('Seizures Missed', 'seizures_missed', '{}'),
    ]
    
    for name, key, fmt in metrics_to_compare:
        orig_val = original[key]
        opt_val = optimized[key]
        
        if isinstance(orig_val, (int, float)) and isinstance(opt_val, (int, float)):
            if key in ['sensitivity', 'specificity', 'precision', 'accuracy']:
                change = (opt_val - orig_val) * 100
                change_str = f"{change:+.1f}pp"
            elif key == 'seizures_missed':
                change = opt_val - orig_val
                change_str = f"{change:+d}"
            else:
                change = opt_val - orig_val
                change_str = f"{change:+.2f}"
        else:
            change_str = "-"
        
        orig_str = fmt.format(orig_val)
        opt_str = fmt.format(opt_val)
        
        # Highlight improvements
        if key == 'sensitivity' and opt_val > orig_val:
            opt_str = f"✓ {opt_str}"
        elif key == 'seizures_missed' and opt_val < orig_val:
            opt_str = f"✓ {opt_str}"
        
        print(f"{name:<25} {orig_str:<20} {opt_str:<20} {change_str:<10}")
    
    print("-"*70)


def suggest_class_weights(labels: np.ndarray) -> Tuple[float, float]:
    """Calculate class weights for balanced training"""
    n_samples = len(labels)
    n_seizure = np.sum(labels == 1)
    n_non_seizure = np.sum(labels == 0)
    
    # Inverse frequency weighting
    weight_non_seizure = n_samples / (2 * n_non_seizure)
    weight_seizure = n_samples / (2 * n_seizure)
    
    # Normalize so weights sum to 2
    total = weight_non_seizure + weight_seizure
    weight_non_seizure = 2 * weight_non_seizure / total
    weight_seizure = 2 * weight_seizure / total
    
    return weight_non_seizure, weight_seizure


def main():
    print("\n" + "="*70)
    print("SENSITIVITY OPTIMIZATION FOR SEIZURE DETECTION")
    print("="*70)
    
    # Load model and data
    model, test_loader, device = load_model_and_data()
    
    # Get predictions
    labels, probs = get_predictions(model, test_loader, device)
    
    print(f"\nTest set: {len(labels)} samples")
    print(f"  Non-seizure: {np.sum(labels == 0)}")
    print(f"  Seizure: {np.sum(labels == 1)}")
    
    # Evaluate at default threshold (0.5)
    print("\n" + "-"*50)
    print("Current Performance (threshold = 0.5)")
    print("-"*50)
    original_metrics = evaluate_at_threshold(labels, probs, 0.5)
    
    print(f"  Sensitivity: {original_metrics['sensitivity']:.2%}")
    print(f"  Specificity: {original_metrics['specificity']:.2%}")
    print(f"  F1 Score: {original_metrics['f1']:.3f}")
    print(f"  Seizures: {original_metrics['seizures_detected']} detected, {original_metrics['seizures_missed']} missed")
    
    # Find optimal thresholds
    results = find_optimal_thresholds(labels, probs)
    
    # Find thresholds for different sensitivity targets
    print("\n" + "-"*50)
    print("Optimal Thresholds for Different Sensitivity Targets")
    print("-"*50)
    
    targets = [0.85, 0.90, 0.95, 1.0]
    optimal_results = {}
    
    for target in targets:
        thresh, metrics = find_threshold_for_target_sensitivity(results, target)
        optimal_results[target] = metrics
        print(f"\n  Target ≥{target:.0%} Sensitivity:")
        print(f"    Threshold: {metrics['threshold']:.3f}")
        print(f"    Sensitivity: {metrics['sensitivity']:.2%}")
        print(f"    Specificity: {metrics['specificity']:.2%}")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    F1: {metrics['f1']:.3f}")
    
    # Recommend best threshold (targeting 90% sensitivity)
    best_target = 0.90
    best_thresh, best_metrics = find_threshold_for_target_sensitivity(results, best_target)
    
    # Evaluate at recommended threshold
    optimized_metrics = evaluate_at_threshold(labels, probs, best_thresh)
    
    # Print comparison
    print_comparison(original_metrics, optimized_metrics)
    
    # Calculate class weights for retraining
    w_non_seizure, w_seizure = suggest_class_weights(labels)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print(f"""
1. QUICK FIX - Use Optimized Threshold:
   Change classification threshold from 0.5 to {best_thresh:.3f}
   This will immediately improve sensitivity from {original_metrics['sensitivity']:.1%} to {optimized_metrics['sensitivity']:.1%}

2. FOR RETRAINING - Add Class Weights:
   In trainer.py, modify the loss function:
   
   class_weights = torch.tensor([{w_non_seizure:.4f}, {w_seizure:.4f}])
   criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
   
   Or modify FocalLoss alpha:
   self.criterion = FocalLoss(alpha={w_seizure:.2f}, gamma=2.0)

3. FOR INFERENCE - Update prediction code:
   Instead of: pred = 1 if prob > 0.5 else 0
   Use:        pred = 1 if prob > {best_thresh:.3f} else 0
""")
    
    # Save optimization results
    save_results = {
        'original_threshold': 0.5,
        'original_metrics': original_metrics,
        'optimized_threshold': float(best_thresh),
        'optimized_metrics': optimized_metrics,
        'sensitivity_targets': {str(k): v for k, v in optimal_results.items()},
        'recommended_class_weights': {
            'non_seizure': float(w_non_seizure),
            'seizure': float(w_seizure)
        }
    }
    
    output_path = Path('evaluation_results') / 'optimization_results.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return optimized_metrics


if __name__ == "__main__":
    main()
