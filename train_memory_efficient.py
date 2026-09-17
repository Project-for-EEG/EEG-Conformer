"""
Memory-Efficient Training with Patient-Level K-Fold Cross-Validation

Key improvements:
1. Patient-level splits (no data leakage)
2. Loads only one fold at a time (memory efficient)
3. Can handle all 24 patients on 16GB RAM
4. Better generalization across patients
"""

import argparse
import gc
import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

from config import get_config
from model import create_model

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class SMOTEOversampler:
    """SMOTE for minority class oversampling"""
    
    def __init__(self, target_ratio: float = 0.3):
        self.target_ratio = target_ratio
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return X, y
        
        minority_class = unique[np.argmin(counts)]
        minority_count = counts.min()
        majority_count = counts.max()
        
        target_minority_count = int(majority_count * self.target_ratio)
        n_synthetic = max(0, target_minority_count - minority_count)
        
        if n_synthetic == 0:
            return X, y
        
        print(f"  SMOTE: Creating {n_synthetic} synthetic seizure samples...")
        
        minority_indices = np.where(y == minority_class)[0]
        minority_samples = X[minority_indices]
        
        synthetic_samples = []
        for _ in range(n_synthetic):
            idx = np.random.randint(0, len(minority_samples))
            neighbor_idx = np.random.randint(0, len(minority_samples))
            while neighbor_idx == idx:
                neighbor_idx = np.random.randint(0, len(minority_samples))
            
            alpha = np.random.random()
            synthetic = minority_samples[idx] + alpha * (minority_samples[neighbor_idx] - minority_samples[idx])
            synthetic += 0.05 * np.random.randn(*synthetic.shape)
            synthetic_samples.append(synthetic)
        
        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.full(n_synthetic, minority_class)
        
        X_resampled = np.concatenate([X, synthetic_samples], axis=0)
        y_resampled = np.concatenate([y, synthetic_labels], axis=0)
        
        print(f"  After SMOTE: {minority_count + n_synthetic} seizures, {majority_count} non-seizures")
        return X_resampled, y_resampled


def load_patient_data(patient_id: str, data_dir, max_segments: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Load one patient's segments, capped at max_segments, keeping all seizures.

    Done in two passes. The previous version appended whole files until it hit
    the budget and then stopped, so for patients whose recordings are long the
    loop broke before it ever opened the seizure-bearing files -- CHB04 loaded
    chb04_01 and chb04_02 (both seizure-free) and silently dropped all 187 of
    its seizure segments. Reading the labels first makes the choice of what to
    keep global to the patient rather than an accident of file order.
    """
    patient_dir = resolve_patient_dir(patient_id, data_dir)
    if patient_dir is None:
        return None, None
    npz_files = sorted(patient_dir.glob("*.npz"))

    if not npz_files:
        return None, None

    # Pass 1: labels only. .npz members decompress lazily, so this touches a few
    # KB per file instead of the 150-600 MB of signal sitting next to them.
    per_file_labels = []
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            per_file_labels.append(data['labels'].copy())

    offsets = np.cumsum([0] + [len(lab) for lab in per_file_labels])
    y_all = np.concatenate(per_file_labels)

    # Keep every seizure segment; spend whatever budget is left on a random
    # sample of non-seizures. If a patient has more seizures than the budget we
    # still keep them all -- they are the scarce class, the cap exists to bound
    # the negatives.
    seizure_idx = np.where(y_all == 1)[0]
    non_seizure_idx = np.where(y_all == 0)[0]
    n_non_seizures = max(0, max_segments - len(seizure_idx))
    if n_non_seizures < len(non_seizure_idx):
        non_seizure_idx = np.random.choice(non_seizure_idx, size=n_non_seizures,
                                           replace=False)
    keep = np.sort(np.concatenate([seizure_idx, non_seizure_idx]))

    # Pass 2: open only the files that actually contribute a kept row, and take
    # just those rows so peak memory stays at roughly one file.
    X_parts, y_parts = [], []
    for i, npz_file in enumerate(npz_files):
        lo, hi = offsets[i], offsets[i + 1]
        local = keep[(keep >= lo) & (keep < hi)] - lo
        if len(local) == 0:
            continue
        with np.load(npz_file) as data:
            X_parts.append(data['segments'][local])
        y_parts.append(per_file_labels[i][local])

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Deliberately not shuffled here: training draws through a
    # WeightedRandomSampler and validation metrics are order-independent, so a
    # permutation would only cost a full extra copy of X.
    return X, y


def as_roots(data_dir) -> List[Path]:
    """One directory or several, always as a list.

    Every caller used to pass a single preprocessed_data/ root. Merging a
    second cohort means a patient can live under either root, so both are
    searched -- a single Path still behaves exactly as before.
    """
    if isinstance(data_dir, (list, tuple)):
        return [Path(d) for d in data_dir]
    return [Path(data_dir)]


def resolve_patient_dir(patient_id: str, data_dir):
    """Which root holds this patient. None when no root does."""
    for root in as_roots(data_dir):
        cand = root / patient_id
        if cand.is_dir():
            return cand
    return None


FULL_CHANNELS = 23      # CHB-MIT's native montage


def apply_channel_drop(X, drop_channels):
    """Reduce one patient's array to the kept channels.

    This has to happen per patient, before any concatenation. CHB-MIT arrives
    with all 23 derivations; Siena arrives already at 20, because three of
    CHB-MIT's channels use FT9/FT10 electrodes that Siena's montage does not
    contain and cannot reconstruct. Concatenating first and slicing after --
    which is what this used to do -- fails the moment both cohorts are loaded
    together, since the arrays disagree on axis 1.

    An array already at the target width is left alone: siena_to_npz.py builds
    it in the post-drop order, so it is already the same montage.
    """
    if not drop_channels:
        return X
    n = X.shape[1]
    target = FULL_CHANNELS - len(drop_channels)
    if n == target:
        return X
    if n != FULL_CHANNELS:
        raise ValueError(
            "expected %d or %d channels, got %d; --drop-channels indices refer "
            "to the %d-channel CHB-MIT montage"
            % (FULL_CHANNELS, target, n, FULL_CHANNELS))
    keep = [c for c in range(n) if c not in drop_channels]
    return X[:, keep, :]


def get_patient_list(data_dir) -> List[str]:
    """Every patient across every root.

    Colliding ids are an error rather than a silent merge: CHB-MIT uses CHBnn
    and Siena uses PNnn so they cannot clash today, but a third cohort reusing
    an id would otherwise have its recordings quietly appended to another
    patient, which is the kind of leak that does not show up in any metric.
    """
    seen, patients = {}, []
    for root in as_roots(data_dir):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if d.name in seen:
                raise ValueError(
                    "patient id %s appears in both %s and %s; ids must be "
                    "unique across datasets" % (d.name, seen[d.name], root))
            seen[d.name] = root
            patients.append(d.name)
    return sorted(patients)


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
    fold: int,
    epochs: int
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Train for one fold"""
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0,
                          label_smoothing=config.model.label_smoothing)
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
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
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


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, target_recall: float = 0.90):
    """Find optimal threshold"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        best_idx = np.argmax(recall)
    else:
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return optimal_threshold, metrics


def run_patient_level_cv(
    data_dir: Path,
    config,
    n_folds: int = 5,
    balance: str = 'sampler',
    target_recall: float = 0.90,
    epochs: int = 15,
    max_patients: int = None,
    max_segments: int = 10000,
    exclude: list = None,
    drop_channels: list = None,
    seed: int = 42,
    ckpt_prefix: str = "fold",
    always_train: list = None
):
    """Run patient-level K-fold cross-validation (memory efficient)"""
    
    print(f"\n{'='*60}")
    print(f"PATIENT-LEVEL {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get all patients
    all_patients = get_patient_list(data_dir)

    # Patients pinned to training never enter the fold rotation, so they are
    # never held out and never scored. That is what makes a second cohort a
    # controlled addition: the held-out patients are identical with and
    # without it, and only the training set differs.
    always_train = list(always_train or [])
    if always_train:
        unknown = [p for p in always_train if p not in all_patients]
        if unknown:
            raise ValueError("--always-train names patients that do not "
                             "exist: " + ", ".join(unknown))
        all_patients = [p for p in all_patients if p not in always_train]
        print("pinned to every training set, never held out: %d patients"
              % len(always_train))
    if exclude:
        dropped = [p for p in all_patients if p in exclude]
        all_patients = [p for p in all_patients if p not in exclude]
        if dropped:
            print(f"excluding {', '.join(dropped)}")
    print(f"\nFound {len(all_patients)} patients total")
    
    # Limit patients if specified
    if max_patients and max_patients < len(all_patients):
        all_patients = all_patients[:max_patients]
        print(f"Limiting to {max_patients} patients: {', '.join(all_patients)}")
    else:
        print(f"Using all {len(all_patients)} patients: {', '.join(all_patients)}")
    
    # Split patients into folds.
    #
    # Which patients share a fold moves the headline by ~8 points, so a single
    # split is one sample from a wide distribution. Varying the seed and
    # re-running gives paired folds across several splits, which is the only
    # way to tell a real effect from a lucky grouping at this fold count.
    np.random.seed(seed)
    np.random.shuffle(all_patients)
    fold_size = len(all_patients) // n_folds
    
    patient_folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(all_patients)
        patient_folds.append(all_patients[start:end])

    # CHB-MIT cases chb01 and chb21 are the SAME child, recorded 1.5 years
    # apart (stated in the PhysioNet dataset description). If they land in
    # different folds, one person sits on both sides of a "patient-independent"
    # split. Force them into the same fold, swapping a partner back so fold
    # sizes stay unchanged.
    SAME_SUBJECT = [("CHB01", "CHB21")]
    for a, b in SAME_SUBJECT:
        fa = next((i for i, f in enumerate(patient_folds) if a in f), None)
        fb = next((i for i, f in enumerate(patient_folds) if b in f), None)
        if fa is None or fb is None or fa == fb:
            continue
        partner = next(p for p in patient_folds[fa] if p != a)
        patient_folds[fa][patient_folds[fa].index(partner)] = b
        patient_folds[fb][patient_folds[fb].index(b)] = partner
        print(f"  note: {b} moved into {a}'s fold (same subject); "
              f"{partner} swapped out to keep fold sizes")

    print(f"\nPatient distribution across folds:")
    for i, patients in enumerate(patient_folds):
        print(f"  Fold {i+1}: {len(patients)} patients - {', '.join(patients)}")
    
    # Only one resampling mechanism by default. Stacking SMOTE (lifts seizures
    # to ~23%) on top of a WeightedRandomSampler (pushes each batch to ~50%)
    # trained the model on a prior ~45x away from the 1.1% it meets at
    # inference, which is what forced the decision threshold down to 0.016.
    smote = SMOTEOversampler(target_ratio=0.3) if balance in ('smote', 'both') else None
    use_sampler = balance in ('sampler', 'both')

    fold_results = []
    all_val_labels = []
    all_val_probs = []
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        # Determine train and val patients
        val_patients = patient_folds[fold]
        train_patients = [p for i, patients in enumerate(patient_folds) if i != fold for p in patients]
        # pinned patients join every fold's training set
        train_patients = train_patients + always_train
        
        print(f"Train patients: {', '.join(train_patients)}")
        print(f"Val patients: {', '.join(val_patients)}")
        
        # Load training data (only this fold)
        print("\nLoading training data...")
        train_X_list, train_y_list = [], []
        for patient in train_patients:
            X, y = load_patient_data(patient, data_dir, max_segments)
            if X is not None:
                X = apply_channel_drop(X, drop_channels)
                train_X_list.append(X)
                train_y_list.append(y)
                print(f"  {patient}: {len(X):,} segments, {np.sum(y)} seizures")
        
        X_train = np.concatenate(train_X_list, axis=0)
        y_train = np.concatenate(train_y_list, axis=0)

        
        print(f"\nTotal training: {len(y_train):,} segments, {np.sum(y_train)} seizures")
        
        # Load validation data
        print("\nLoading validation data...")
        val_X_list, val_y_list = [], []
        for patient in val_patients:
            X, y = load_patient_data(patient, data_dir, max_segments)
            if X is not None:
                X = apply_channel_drop(X, drop_channels)
                val_X_list.append(X)
                val_y_list.append(y)
                print(f"  {patient}: {len(X):,} segments, {np.sum(y)} seizures")
        
        X_val = np.concatenate(val_X_list, axis=0)
        y_val = np.concatenate(val_y_list, axis=0)

        
        print(f"\nTotal validation: {len(y_val):,} segments, {np.sum(y_val)} seizures")
        
        # Apply SMOTE
        if smote:
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Create TRAINING dataloader first
        print("\nCreating training dataloader...")
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        
        if use_sampler:
            class_counts = np.bincount(y_train)
            weights = 1.0 / class_counts[y_train]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size,
            sampler=sampler, shuffle=shuffle,
            num_workers=0, pin_memory=False  # Don't pin for memory
        )
        
        # FREE MEMORY before loading validation
        print("Freeing training arrays from memory...")
        del X_train, y_train, train_X_list, train_y_list
        gc.collect()
        
        # NOW load validation data
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.training.batch_size,
            shuffle=False, num_workers=0, pin_memory=False
        )
        
        # Free validation arrays too
        del X_val, y_val, val_X_list, val_y_list
        gc.collect()
        
        # Create model
        model = create_model(config.model).to(device)
        
        # Train
        metrics, val_labels, val_probs = train_one_fold(
            model, train_loader, val_loader, config, device, fold, epochs
        )

        # Save the fold's model. Without this the CV run produced metrics but no
        # artifact, so nothing downstream (temporal smoothing, inference on new
        # recordings) had a model to load. val_patients is recorded so evaluation
        # can stay honest about which patients this fold never saw.
        ckpt_dir = Path(config.training.save_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Prefixed so runs cannot overwrite each other. The plain "fold1.pt"
        # name is what the published 24-patient models use, and an ablation
        # run silently replaced them once already.
        ckpt_path = ckpt_dir / f'{ckpt_prefix}{fold + 1}.pt'
        torch.save({
            'fold': fold + 1,
            'model_state_dict': model.state_dict(),
            'model_config': asdict(config.model),
            'val_patients': val_patients,
            'train_patients': train_patients,
            'metrics': metrics,
            'balance': balance,
            'max_segments': max_segments,
        }, ckpt_path)
        print(f"  Saved {ckpt_path}")

        # Free memory. X_train/y_train and X_val/y_val were already released
        # above, once their tensors were handed to the datasets -- deleting them
        # again here raised UnboundLocalError and killed the run after fold 1.
        del train_dataset, val_dataset
        del train_loader, val_loader, model
        gc.collect()
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
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.capitalize():12s}: {mean_val:.4f} +/- {std_val:.4f}")
    
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
        all_val_labels, all_val_probs, target_recall=target_recall
    )
    
    print(f"\nAt optimized threshold ({optimal_threshold:.3f}) for {target_recall*100:.0f}% recall:")
    print(f"  Precision: {opt_metrics['precision']:.4f}")
    print(f"  Recall:    {opt_metrics['recall']:.4f}")
    print(f"  F1:        {opt_metrics['f1']:.4f}")
    
    opt_preds = (all_val_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(all_val_labels, opt_preds)
    print(f"\nConfusion Matrix (at optimal threshold):")
    print(f"  TN: {cm[0,0]:6d}  FP: {cm[0,1]:6d}")
    print(f"  FN: {cm[1,0]:6d}  TP: {cm[1,1]:6d}")
    
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
        'optimal_threshold': float(optimal_threshold),
        'metrics_at_optimal_threshold': {k: float(v) for k, v in opt_metrics.items()},
        'n_folds': n_folds,
        'n_patients': len(all_patients),
        'seed': seed,
        'drop_channels': drop_channels,
        'balance': balance,
        'max_segments': max_segments,
        'timestamp': datetime.now().isoformat()
    }
    
    results_dir = Path('cv_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'patient_cv_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Memory-Efficient Patient-Level CV')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs per fold')
    parser.add_argument('--balance', choices=['sampler', 'smote', 'both', 'none'],
                        default='sampler',
                        help="How to counter class imbalance. 'sampler' = weighted "
                             "batch sampling only (default). 'smote' = synthetic "
                             "oversampling only. 'both' = the old stacked behaviour, "
                             "which skews the training prior to ~50%% seizure against "
                             "1.1%% at inference and wrecks threshold calibration. "
                             "'none' = train on the natural distribution.")
    parser.add_argument('--target-recall', type=float, default=0.90, help='Target recall')
    parser.add_argument('--max-patients', type=int, default=None, help='Limit number of patients')
    parser.add_argument('--always-train', nargs='*', default=None,
                        help='patients that join every fold training set and '
                             'are never held out; use to add a second cohort '
                             'while still measuring on the first')
    parser.add_argument('--ckpt-prefix', default='fold',
                        help='checkpoint filename prefix, so a side experiment '
                             'does not overwrite the published fold1-3.pt')
    parser.add_argument('--seed', type=int, default=42,
                        help='controls the patient-to-fold assignment and the '
                             'background subsample; vary it to measure how '
                             'much of a result is the grouping')
    parser.add_argument('--data-dirs', nargs='*', default=['preprocessed_data'],
                        help='one or more preprocessed roots; pass '
                             'preprocessed_data preprocessed_data_siena to '
                             'train on both cohorts')
    parser.add_argument('--drop-channels', type=int, nargs='*', default=None,
                        help='channel indices to drop, e.g. 19 20 21 for the '
                             'FT9/FT10 derivations that cannot be reconstructed '
                             "from the Siena unipolar montage")
    parser.add_argument('--exclude', nargs='*', default=None,
                        help='patient ids to drop entirely, e.g. CHB03 CHB05 '
                             'whose local data predates raw_data/ and does not '
                             'match the PhysioNet recordings')
    parser.add_argument('--max-segments', type=int, default=10000,
                        help='Max segments loaded per patient. Seizure segments are '
                             'always kept; only non-seizure ones are subsampled. '
                             'Each segment is ~94 KB, so 24 patients x 10000 needs '
                             '~15 GB of RAM -- lower this on a 16 GB machine.')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MEMORY-EFFICIENT PATIENT-LEVEL TRAINING")
    print("="*60)
    print(f"  K-Fold CV: {args.folds} folds")
    print(f"  Epochs: {args.epochs}")
    print(f"  Balancing: {args.balance}")
    print(f"  Target Recall: {args.target_recall*100:.0f}%")
    print(f"  Max segments/patient: {args.max_segments:,}")
    print("="*60)
    
    set_seed(args.seed)
    
    config = get_config()
    if args.drop_channels:
        config.model.num_channels -= len(args.drop_channels)
        print(f"  dropping channels {args.drop_channels} -> "
              f"{config.model.num_channels} channels")
    data_dir = [Path(d) for d in args.data_dirs]
    
    missing = [d for d in data_dir if not d.exists()]
    if missing:
        print("Error: %s not found!"
              % ", ".join(str(m) for m in missing))
        return
    results = run_patient_level_cv(
        data_dir=data_dir,
        config=config,
        n_folds=args.folds,
        balance=args.balance,
        target_recall=args.target_recall,
        epochs=args.epochs,
        max_patients=args.max_patients,
        max_segments=args.max_segments,
        exclude=args.exclude,
        drop_channels=args.drop_channels,
        seed=args.seed,
        ckpt_prefix=args.ckpt_prefix,
        always_train=args.always_train
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
