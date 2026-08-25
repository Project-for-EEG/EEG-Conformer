"""
EEG Data Loading and Preprocessing Pipeline
Handles CHB-MIT dataset format with advanced signal processing
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import DataConfig, get_config


class EEGPreprocessor:
    """Advanced EEG signal preprocessing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = None
        
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove low and high frequency noise"""
        nyquist = self.config.sampling_rate / 2
        low = self.config.lowcut / nyquist
        high = self.config.highcut / nyquist
        
        # Ensure valid filter parameters
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Remove powerline interference (50/60 Hz)"""
        nyquist = self.config.sampling_rate / 2
        freq = self.config.notch_freq / nyquist
        
        if freq >= 1.0:
            return data  # Skip if notch freq is above Nyquist
            
        b, a = iirnotch(freq, Q=30)
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def remove_artifacts(self, data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Remove extreme artifact values using z-score clipping"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
        
        z_scores = (data - mean) / std
        data = np.where(np.abs(z_scores) > threshold, mean, data)
        
        return data
    
    def normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize EEG signals"""
        original_shape = data.shape
        
        if self.config.normalize_method == "zscore":
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                data_flat = data.reshape(-1, 1)
                self.scaler.fit(data_flat)
            data_flat = data.reshape(-1, 1)
            data = self.scaler.transform(data_flat).reshape(original_shape)
            
        elif self.config.normalize_method == "robust":
            if fit or self.scaler is None:
                self.scaler = RobustScaler()
                data_flat = data.reshape(-1, 1)
                self.scaler.fit(data_flat)
            data_flat = data.reshape(-1, 1)
            data = self.scaler.transform(data_flat).reshape(original_shape)
            
        elif self.config.normalize_method == "minmax":
            if fit or self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                data_flat = data.reshape(-1, 1)
                self.scaler.fit(data_flat)
            data_flat = data.reshape(-1, 1)
            data = self.scaler.transform(data_flat).reshape(original_shape)
            
        else:  # Per-channel z-score (default for EEG)
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True) + 1e-8
            data = (data - mean) / std
        
        return data
    
    def preprocess(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Full preprocessing pipeline"""
        # 1. Bandpass filter
        data = self.bandpass_filter(data)
        
        # 2. Notch filter for powerline noise
        data = self.notch_filter(data)
        
        # 3. Artifact removal
        data = self.remove_artifacts(data)
        
        # 4. Normalization
        data = self.normalize(data, fit=fit)
        
        return data.astype(np.float32)


class EEGDataLoader:
    """Load and segment EEG data from CSV files"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.preprocessor = EEGPreprocessor(config)
        
    def load_csv(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single CSV file"""
        print(f"Loading {file_path}...")
        
        # Read CSV with optimized settings
        df = pd.read_csv(file_path, low_memory=False)
        
        # Extract time, channels, and labels
        time_col = df.columns[0]  # "Time (s)"
        label_col = df.columns[-1]  # "Seizure"
        channel_cols = df.columns[1:-1]  # All EEG channels
        
        # Convert to numpy arrays
        eeg_data = df[channel_cols].values.T  # Shape: (channels, samples)
        labels = df[label_col].values  # Shape: (samples,)
        
        print(f"  Loaded: {eeg_data.shape[1]} samples, {eeg_data.shape[0]} channels")
        print(f"  Seizure samples: {np.sum(labels)} ({100*np.mean(labels):.2f}%)")
        
        return eeg_data.astype(np.float32), labels.astype(np.int64)
    
    def segment_data(
        self, 
        eeg_data: np.ndarray, 
        labels: np.ndarray,
        overlap: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment continuous EEG into fixed-length windows"""
        window_size = self.config.window_size_samples
        
        if overlap:
            step = window_size - self.config.overlap_samples
        else:
            step = window_size
        
        n_samples = eeg_data.shape[1]
        n_windows = (n_samples - window_size) // step + 1
        
        segments = []
        segment_labels = []
        
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            
            segment = eeg_data[:, start:end]
            segment_label_values = labels[start:end]
            
            # Window is labeled as seizure if >50% of samples are seizure
            window_label = 1 if np.mean(segment_label_values) > 0.5 else 0
            
            segments.append(segment)
            segment_labels.append(window_label)
        
        return np.array(segments), np.array(segment_labels)
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load all CSV files from data directory"""
        data_dir = Path(self.config.data_dir)
        
        all_segments = []
        all_labels = []
        file_sources = []
        
        # Find all CSV files
        csv_files = list(data_dir.rglob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            # Load and preprocess
            eeg_data, labels = self.load_csv(str(csv_file))
            
            # Preprocess (fit scaler on first file only)
            fit_scaler = len(all_segments) == 0
            eeg_data = self.preprocessor.preprocess(eeg_data, fit=fit_scaler)
            
            # Segment
            segments, segment_labels = self.segment_data(eeg_data, labels)
            
            all_segments.append(segments)
            all_labels.append(segment_labels)
            file_sources.extend([csv_file.stem] * len(segments))
            
            print(f"  Segments: {len(segments)}, Seizure: {np.sum(segment_labels)}")
        
        # Concatenate all data
        X = np.concatenate(all_segments, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        print(f"\nTotal: {len(X)} segments")
        print(f"Seizure: {np.sum(y)} ({100*np.mean(y):.2f}%)")
        print(f"Non-seizure: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.2f}%)")
        
        return X, y, file_sources


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(
        self, 
        segments: np.ndarray, 
        labels: np.ndarray,
        augment: bool = False
    ):
        self.segments = torch.from_numpy(segments).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.segments[idx]
        y = self.labels[idx]
        
        if self.augment:
            x = self._augment(x)
        
        return x, y
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply strong data augmentation to prevent overfitting"""
        # Random amplitude scaling (more aggressive)
        if torch.rand(1) < 0.7:
            scale = 0.7 + 0.6 * torch.rand(1)
            x = x * scale
        
        # Random noise addition (more frequent)
        if torch.rand(1) < 0.5:
            noise = 0.1 * torch.randn_like(x)
            x = x + noise
        
        # Random time shift (more aggressive)
        if torch.rand(1) < 0.5:
            shift = torch.randint(-100, 100, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=1)
        
        # Random channel dropout (more aggressive)
        if torch.rand(1) < 0.4:
            n_drop = torch.randint(1, 6, (1,)).item()
            drop_channels = torch.randperm(x.shape[0])[:n_drop]
            x[drop_channels] = 0
        
        # Random time masking (new)
        if torch.rand(1) < 0.3:
            mask_len = torch.randint(50, 150, (1,)).item()
            mask_start = torch.randint(0, x.shape[1] - mask_len, (1,)).item()
            x[:, mask_start:mask_start + mask_len] = 0
        
        # Mixup-style amplitude variation per channel (new)
        if torch.rand(1) < 0.3:
            channel_scales = 0.8 + 0.4 * torch.rand(x.shape[0], 1)
            x = x * channel_scales
        
        return x


def create_data_loaders(
    config: DataConfig,
    training_config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Load all data
    loader = EEGDataLoader(config)
    X, y, sources = loader.load_all_data()
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, 
        test_size=config.test_ratio,
        random_state=training_config.seed,
        stratify=y
    )
    
    # Second split: train vs val
    val_ratio_adjusted = config.val_ratio / (1 - config.test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        random_state=training_config.seed,
        stratify=y_trainval
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} (Seizure: {np.sum(y_train)})")
    print(f"  Val:   {len(X_val)} (Seizure: {np.sum(y_val)})")
    print(f"  Test:  {len(X_test)} (Seizure: {np.sum(y_test)})")
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train, augment=True)
    val_dataset = EEGDataset(X_val, y_val, augment=False)
    test_dataset = EEGDataset(X_test, y_test, augment=False)
    
    # Handle class imbalance with weighted sampler
    if config.balance_classes:
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    config = get_config()
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data, config.training
    )
    
    # Check a batch
    for batch_x, batch_y in train_loader:
        print(f"\nBatch shape: {batch_x.shape}")
        print(f"Labels: {batch_y}")
        break
