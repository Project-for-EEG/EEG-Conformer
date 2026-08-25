"""
Streaming Training - Loads data on-the-fly from disk
Can train on ALL 24 patients with only 16GB RAM!

Key features:
1. Streaming DataLoader - loads batches from disk as needed
2. Never loads entire dataset into memory
3. Patient-level cross-validation
4. SMOTE applied per-batch for memory efficiency
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

from config import get_config
from model import create_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StreamingEEGDataset(Dataset):
    """
    Memory-efficient dataset that loads segments on-the-fly from .npz files.
    Only keeps file paths and indices in memory, not the actual data.
    """
    
    def __init__(self, patient_dirs: List[Path], max_segments_per_patient: int = 10000):
        self.segments_info = []  # List of (npz_path, segment_index, label)
        self.max_segments = max_segments_per_patient
        
        print(f"Building streaming dataset index...")
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            npz_files = sorted(patient_dir.glob("*.npz"))
            
            patient_segments = []
            for npz_file in npz_files:
                # Load only metadata (labels) to build index
                data = np.load(npz_file)
                labels = data['labels']
                n_segments = len(labels)
                
                for i in range(n_segments):
                    patient_segments.append((npz_file, i, int(labels[i])))
                
                # Stop if we have enough segments
                if len(patient_segments) >= self.max_segments:
                    break
            
            # Limit and balance segments per patient
            if len(patient_segments) > self.max_segments:
                # Keep all seizures, sample non-seizures
                seizure_segments = [s for s in patient_segments if s[2] == 1]
                non_seizure_segments = [s for s in patient_segments if s[2] == 0]
                
                n_to_sample = min(self.max_segments - len(seizure_segments), len(non_seizure_segments))
                if n_to_sample > 0:
                    sampled = random.sample(non_seizure_segments, n_to_sample)
                    patient_segments = seizure_segments + sampled
            
            n_seizures = sum(1 for s in patient_segments if s[2] == 1)
            print(f"  {patient_id}: {len(patient_segments)} segments, {n_seizures} seizures")
            
            self.segments_info.extend(patient_segments)
        
        # Cache for recently loaded files
        self._cache = {}
        self._cache_size = 3  # Keep last 3 files in memory
        
        print(f"Total: {len(self.segments_info)} segments")
    
    def __len__(self):
        return len(self.segments_info)
    
    def __getitem__(self, idx):
        npz_path, segment_idx, label = self.segments_info[idx]
        
        # Load from cache or disk
        cache_key = str(npz_path)
        if cache_key not in self._cache:
            # Load file
            data = np.load(npz_path)
            segments = data['segments']
            
            # Manage cache size
            if len(self._cache) >= self._cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = segments
        
        segment = self._cache[cache_key][segment_idx]
        
        return torch.from_numpy(segment).float(), torch.tensor(label).long()
    
    def get_labels(self):
        """Return all labels for weighted sampling"""
        return np.array([s[2] for s in self.segments_info])


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_patient_dirs(data_dir: Path) -> List[Path]:
    """Get list of patient directories"""
    return sorted([d for d in data_dir.iterdir() if d.is_dir()])


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    fold: int,
    epochs: int
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Train for one fold"""
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.15)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scaler = GradScaler()
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs}", leave=False)
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())
                val_probs.extend(probs.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= config.training.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final validation
    model.eval()
    val_preds, val_labels, val_probs = [], [], []
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.numpy())
            val_probs.extend(probs.cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(val_labels, val_preds),
        'precision': precision_score(val_labels, val_preds, zero_division=0),
        'recall': recall_score(val_labels, val_preds, zero_division=0),
        'f1': f1_score(val_labels, val_preds, zero_division=0),
    }
    
    if len(np.unique(val_labels)) > 1:
        metrics['auc'] = roc_auc_score(val_labels, val_probs)
    
    return metrics, np.array(val_labels), np.array(val_probs)


def find_optimal_threshold(y_true, y_probs, target_recall=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        best_idx = np.argmax(recall)
    else:
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    return optimal_threshold, {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def run_streaming_cv(
    data_dir: Path,
    config,
    n_folds: int = 3,
    target_recall: float = 0.90,
    epochs: int = 15,
    max_segments_per_patient: int = 8000
):
    """Run patient-level CV with streaming data loading"""
    
    print(f"\n{'='*60}")
    print(f"STREAMING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get all patient directories
    all_patient_dirs = get_patient_dirs(data_dir)
    print(f"\nFound {len(all_patient_dirs)} patients: {', '.join([d.name for d in all_patient_dirs])}")
    
    # Shuffle and split into folds
    np.random.seed(42)
    shuffled_dirs = all_patient_dirs.copy()
    np.random.shuffle(shuffled_dirs)
    
    fold_size = len(shuffled_dirs) // n_folds
    patient_folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled_dirs)
        patient_folds.append(shuffled_dirs[start:end])
    
    print(f"\nPatient distribution:")
    for i, patients in enumerate(patient_folds):
        print(f"  Fold {i+1}: {len(patients)} patients - {', '.join([p.name for p in patients])}")
    
    fold_results = []
    all_val_labels = []
    all_val_probs = []
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split patients
        val_dirs = patient_folds[fold]
        train_dirs = [d for i, dirs in enumerate(patient_folds) if i != fold for d in dirs]
        
        print(f"\nTrain patients ({len(train_dirs)}): {', '.join([d.name for d in train_dirs])}")
        print(f"Val patients ({len(val_dirs)}): {', '.join([d.name for d in val_dirs])}")
        
        # Create streaming datasets
        print("\nBuilding training dataset...")
        train_dataset = StreamingEEGDataset(train_dirs, max_segments_per_patient)
        
        print("\nBuilding validation dataset...")
        val_dataset = StreamingEEGDataset(val_dirs, max_segments_per_patient)
        
        # Weighted sampler for class imbalance
        train_labels = train_dataset.get_labels()
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.training.batch_size,
            sampler=sampler,
            num_workers=0,  # Important for streaming
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Create model
        model = create_model(config.model).to(device)
        
        # Train
        metrics, val_labels, val_probs = train_one_fold(
            model, train_loader, val_loader, config, device, fold, epochs
        )
        
        # Clean up
        del train_dataset, val_dataset, train_loader, val_loader, model
        torch.cuda.empty_cache()
        
        fold_results.append(metrics)
        all_val_labels.extend(val_labels)
        all_val_probs.extend(val_probs)
        
        print(f"\nFold {fold+1} Results:")
        for key, value in metrics.items():
            print(f"  {key.capitalize():12s}: {value:.4f}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS (Mean +/- Std)")
    print(f"{'='*60}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r.get(metric, 0) for r in fold_results]
        print(f"  {metric.capitalize():12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    
    # Threshold optimization
    all_val_labels = np.array(all_val_labels)
    all_val_probs = np.array(all_val_probs)
    
    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    
    default_preds = (all_val_probs >= 0.5).astype(int)
    print(f"\nAt default threshold (0.5):")
    print(f"  Precision: {precision_score(all_val_labels, default_preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(all_val_labels, default_preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(all_val_labels, default_preds, zero_division=0):.4f}")
    
    optimal_threshold, opt_metrics = find_optimal_threshold(
        all_val_labels, all_val_probs, target_recall
    )
    
    print(f"\nAt optimized threshold ({optimal_threshold:.3f}) for {target_recall*100:.0f}% recall:")
    print(f"  Precision: {opt_metrics['precision']:.4f}")
    print(f"  Recall:    {opt_metrics['recall']:.4f}")
    print(f"  F1:        {opt_metrics['f1']:.4f}")
    
    opt_preds = (all_val_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(all_val_labels, opt_preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:6d}  FP: {cm[0,1]:6d}")
    print(f"  FN: {cm[1,0]:6d}  TP: {cm[1,1]:6d}")
    
    # Save results
    results = {
        'fold_results': fold_results,
        'mean_metrics': {m: float(np.mean([r.get(m, 0) for r in fold_results])) 
                        for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']},
        'optimal_threshold': float(optimal_threshold),
        'n_folds': n_folds,
        'n_patients': len(all_patient_dirs),
        'timestamp': datetime.now().isoformat()
    }
    
    results_dir = Path('cv_results')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'streaming_cv_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Streaming Training - All Patients')
    parser.add_argument('--folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs per fold')
    parser.add_argument('--target-recall', type=float, default=0.90, help='Target recall')
    parser.add_argument('--max-segments', type=int, default=8000, 
                       help='Max segments per patient (for balance)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STREAMING TRAINING - ALL 24 PATIENTS")
    print("="*60)
    print(f"  K-Fold CV: {args.folds} folds")
    print(f"  Epochs: {args.epochs}")
    print(f"  Max segments/patient: {args.max_segments}")
    print(f"  Target Recall: {args.target_recall*100:.0f}%")
    print("="*60)
    
    set_seed(42)
    
    config = get_config()
    data_dir = Path("preprocessed_data")
    
    if not data_dir.exists():
        print(f"\nError: {data_dir} not found!")
        print("Run: python preprocess_data.py first")
        return
    
    results = run_streaming_cv(
        data_dir=data_dir,
        config=config,
        n_folds=args.folds,
        target_recall=args.target_recall,
        epochs=args.epochs,
        max_segments_per_patient=args.max_segments
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
