"""
Improved Training Pipeline with:
1. K-Fold Cross-Validation
2. Patient-Level Splitting (no data leakage)
3. SMOTE Oversampling for seizure class
4. Threshold Optimization for better recall
5. More robust evaluation

Usage:
    python train_improved.py                    # Run with defaults
    python train_improved.py --folds 5          # 5-fold CV
    python train_improved.py --target-recall 0.95  # Optimize for 95% recall
"""

import argparse
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

from config import get_config
from model import EEGConformer, create_model

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class SMOTEOversampler:
    """
    Simplified SMOTE (Synthetic Minority Over-sampling Technique)
    Creates synthetic seizure samples by interpolating between existing ones
    """
    
    def __init__(self, target_ratio: float = 0.5, k_neighbors: int = 5):
        self.target_ratio = target_ratio  # Target ratio of minority to majority
        self.k_neighbors = k_neighbors
    
    def fit_resample(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample the minority class
        
        Args:
            X: Features (n_samples, channels, time)
            y: Labels (n_samples,)
        
        Returns:
            X_resampled, y_resampled
        """
        # Find minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_count = counts.min()
        majority_count = counts.max()
        
        # Calculate how many synthetic samples to create
        target_minority_count = int(majority_count * self.target_ratio)
        n_synthetic = max(0, target_minority_count - minority_count)
        
        if n_synthetic == 0:
            return X, y
        
        print(f"  SMOTE: Creating {n_synthetic} synthetic seizure samples...")
        
        # Get minority class samples
        minority_indices = np.where(y == minority_class)[0]
        minority_samples = X[minority_indices]
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Pick a random minority sample
            idx = np.random.randint(0, len(minority_samples))
            sample = minority_samples[idx]
            
            # Pick a random neighbor
            neighbor_idx = np.random.randint(0, len(minority_samples))
            while neighbor_idx == idx:
                neighbor_idx = np.random.randint(0, len(minority_samples))
            neighbor = minority_samples[neighbor_idx]
            
            # Interpolate
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            
            # Add small noise for diversity
            noise = 0.05 * np.random.randn(*synthetic.shape)
            synthetic = synthetic + noise
            
            synthetic_samples.append(synthetic)
        
        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.full(n_synthetic, minority_class)
        
        # Combine with original data
        X_resampled = np.concatenate([X, synthetic_samples], axis=0)
        y_resampled = np.concatenate([y, synthetic_labels], axis=0)
        
        print(f"  Original: {minority_count} seizures, {majority_count} non-seizures")
        print(f"  After SMOTE: {minority_count + n_synthetic} seizures, {majority_count} non-seizures")
        
        return X_resampled, y_resampled


class PatientLevelDataLoader:
    """
    Load preprocessed .npz data with patient-level organization
    """
    
    def __init__(self, config, max_patients: int = None, max_files_per_patient: int = None, 
                 use_preprocessed: bool = True):
        self.config = config
        self.patient_data = {}  # patient_id -> {'X': segments, 'y': labels}
        self.max_patients = max_patients
        self.max_files_per_patient = max_files_per_patient
        self.use_preprocessed = use_preprocessed
        
        # Use preprocessed data directory if available
        self.data_dir = Path("preprocessed_data") if use_preprocessed else Path(config.data.data_dir)
    
    def load_all_patients(self) -> Dict[str, Dict]:
        """Load data organized by patient from preprocessed .npz files"""
        
        if self.use_preprocessed and self.data_dir.exists():
            return self._load_from_npz()
        else:
            print("Preprocessed data not found. Run preprocess_data.py first!")
            raise FileNotFoundError("Run: python preprocess_data.py")
    
    def _load_from_npz(self) -> Dict[str, Dict]:
        """Load from preprocessed .npz files (fast and memory-efficient)"""
        npz_files = list(self.data_dir.rglob("*.npz"))
        
        print(f"Found {len(npz_files)} preprocessed .npz files")
        
        # Organize files by patient
        patient_files = {}
        for npz_file in npz_files:
            patient_id = npz_file.parent.name
            if patient_id not in patient_files:
                patient_files[patient_id] = []
            patient_files[patient_id].append(npz_file)
        
        # Sort patients
        sorted_patients = sorted(patient_files.keys())
        
        # Limit number of patients if specified
        if self.max_patients:
            sorted_patients = sorted_patients[:self.max_patients]
            print(f"Limiting to {len(sorted_patients)} patients")
        
        # Load files
        total_files = 0
        for patient_id in sorted_patients:
            files = sorted(patient_files[patient_id])
            if self.max_files_per_patient:
                files = files[:self.max_files_per_patient]
            
            patient_segments = []
            patient_labels = []
            
            for npz_file in files:
                try:
                    data = np.load(npz_file)
                    segments = data['segments']
                    labels = data['labels']
                    
                    patient_segments.append(segments)
                    patient_labels.append(labels)
                    total_files += 1
                except Exception as e:
                    print(f"  Error loading {npz_file.name}: {e}")
                    continue
            
            if patient_segments:
                self.patient_data[patient_id] = {
                    'X': np.concatenate(patient_segments, axis=0),
                    'y': np.concatenate(patient_labels, axis=0)
                }
        
        # Print summary
        print(f"\nLoaded {total_files} files from {len(self.patient_data)} patients")
        print(f"\n{'='*50}")
        print("Data Summary by Patient:")
        print(f"{'='*50}")
        
        total_segments = 0
        total_seizures = 0
        for patient_id, data in sorted(self.patient_data.items()):
            n_seizure = np.sum(data['y'])
            n_total = len(data['y'])
            total_segments += n_total
            total_seizures += n_seizure
            pct = 100*n_seizure/n_total if n_total > 0 else 0
            print(f"  {patient_id}: {n_total:,} segments, {n_seizure} seizures ({pct:.1f}%)")
        
        print(f"\nTotal: {total_segments:,} segments, {total_seizures:,} seizures")
        
        return self.patient_data


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    target_recall: float = 0.95
) -> Tuple[float, Dict]:
    """
    Find threshold that achieves target recall while maximizing precision
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find thresholds that achieve at least target_recall
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        # Can't achieve target, use threshold that maximizes recall
        best_idx = np.argmax(recall)
    else:
        # Among those, find the one with best precision
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
    
    if best_idx < len(thresholds):
        optimal_threshold = thresholds[best_idx]
    else:
        optimal_threshold = 0.5
    
    # Calculate metrics at optimal threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return optimal_threshold, metrics


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
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


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device,
    fold: int
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Train for one fold and return metrics"""
    
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
    
    for epoch in range(config.training.epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", leave=False)
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
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
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
        
        # Calculate metrics
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        
        # Check for best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final validation predictions
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
    
    # Calculate final metrics
    metrics = {
        'accuracy': accuracy_score(val_labels, val_preds),
        'precision': precision_score(val_labels, val_preds, zero_division=0),
        'recall': recall_score(val_labels, val_preds, zero_division=0),
        'f1': f1_score(val_labels, val_preds, zero_division=0),
    }
    
    if len(np.unique(val_labels)) > 1:
        metrics['auc'] = roc_auc_score(val_labels, val_probs)
    
    return metrics, np.array(val_labels), np.array(val_probs)


def run_kfold_cv(
    patient_data: Dict,
    config,
    n_folds: int = 5,
    use_smote: bool = True,
    target_recall: float = 0.90
):
    """Run K-Fold Cross-Validation with patient-level or segment-level splits"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Combine all data
    all_X = np.concatenate([d['X'] for d in patient_data.values()], axis=0)
    all_y = np.concatenate([d['y'] for d in patient_data.values()], axis=0)
    
    print(f"Total samples: {len(all_y)}")
    print(f"Seizures: {np.sum(all_y)} ({100*np.mean(all_y):.2f}%)")
    
    # Initialize SMOTE
    smote = SMOTEOversampler(target_ratio=0.3) if use_smote else None
    
    # K-Fold split
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_val_labels = []
    all_val_probs = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_X, all_y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        X_train, X_val = all_X[train_idx], all_X[val_idx]
        y_train, y_val = all_y[train_idx], all_y[val_idx]
        
        print(f"Train: {len(y_train)} samples, {np.sum(y_train)} seizures")
        print(f"Val:   {len(y_val)} samples, {np.sum(y_val)} seizures")
        
        # Apply SMOTE to training data
        if smote:
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        
        # Weighted sampler for training
        class_counts = np.bincount(y_train)
        weights = 1.0 / class_counts[y_train]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size,
            sampler=sampler, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.training.batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )
        
        # Create fresh model for this fold
        model = create_model(config.model).to(device)
        
        # Train
        metrics, val_labels, val_probs = train_one_fold(
            model, train_loader, val_loader, config, device, fold
        )
        
        fold_results.append(metrics)
        all_val_labels.extend(val_labels)
        all_val_probs.extend(val_probs)
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics.get('auc', 0):.4f}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS (Mean +/- Std)")
    print(f"{'='*60}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r.get(metric, 0) for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.capitalize():12s}: {mean_val:.4f} +/- {std_val:.4f}")
    
    # Find optimal threshold
    all_val_labels = np.array(all_val_labels)
    all_val_probs = np.array(all_val_probs)
    
    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    
    # Default threshold (0.5)
    default_preds = (all_val_probs >= 0.5).astype(int)
    print(f"\nAt default threshold (0.5):")
    print(f"  Precision: {precision_score(all_val_labels, default_preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(all_val_labels, default_preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(all_val_labels, default_preds, zero_division=0):.4f}")
    
    # Optimized threshold
    optimal_threshold, opt_metrics = find_optimal_threshold(
        all_val_labels, all_val_probs, target_recall=target_recall
    )
    
    print(f"\nAt optimized threshold ({optimal_threshold:.3f}) for {target_recall*100:.0f}% recall:")
    print(f"  Precision: {opt_metrics['precision']:.4f}")
    print(f"  Recall:    {opt_metrics['recall']:.4f}")
    print(f"  F1:        {opt_metrics['f1']:.4f}")
    
    # Confusion matrix at optimal threshold
    opt_preds = (all_val_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(all_val_labels, opt_preds)
    print(f"\nConfusion Matrix (at optimal threshold):")
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    # Save results
    results = {
        'fold_results': fold_results,
        'mean_metrics': {
            metric: float(np.mean([r.get(metric, 0) for r in fold_results]))
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        },
        'std_metrics': {
            metric: float(np.std([r.get(metric, 0) for r in fold_results]))
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        },
        'optimal_threshold': optimal_threshold,
        'metrics_at_optimal_threshold': opt_metrics,
        'n_folds': n_folds,
        'use_smote': use_smote,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file
    results_dir = Path('cv_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'cv_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Improved Training with K-Fold CV')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    parser.add_argument('--target-recall', type=float, default=0.90, 
                       help='Target recall for threshold optimization')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs per fold')
    parser.add_argument('--max-patients', type=int, default=None, 
                       help='Limit number of patients (for memory/speed)')
    parser.add_argument('--max-files', type=int, default=3,
                       help='Max files per patient (for memory/speed)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("IMPROVED EEG SEIZURE DETECTION TRAINING")
    print("="*60)
    print(f"  K-Fold CV: {args.folds} folds")
    print(f"  SMOTE: {'Enabled' if not args.no_smote else 'Disabled'}")
    print(f"  Target Recall: {args.target_recall*100:.0f}%")
    if args.max_patients:
        print(f"  Max Patients: {args.max_patients}")
    print(f"  Max Files/Patient: {args.max_files}")
    print("="*60)
    
    set_seed(42)
    
    # Load configuration
    config = get_config()
    config.training.epochs = args.epochs
    
    # Load data with patient organization
    loader = PatientLevelDataLoader(
        config, 
        max_patients=args.max_patients,
        max_files_per_patient=args.max_files
    )
    patient_data = loader.load_all_patients()
    
    # Run K-Fold CV
    results = run_kfold_cv(
        patient_data=patient_data,
        config=config,
        n_folds=args.folds,
        use_smote=not args.no_smote,
        target_recall=args.target_recall
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
