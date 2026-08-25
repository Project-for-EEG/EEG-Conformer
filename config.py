"""
Configuration for EEG Seizure Detection using SOTA EEG-Conformer Model
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "EEG_Data"
    
    # EEG Parameters
    sampling_rate: int = 256  # Hz (CHB-MIT dataset)
    num_channels: int = 23
    
    # Segmentation
    window_size_sec: float = 4.0  # 4-second windows (1024 samples)
    overlap_ratio: float = 0.5  # 50% overlap for training
    
    # Preprocessing
    lowcut: float = 0.5  # Hz - high-pass filter
    highcut: float = 50.0  # Hz - low-pass filter
    notch_freq: float = 60.0  # Hz - powerline noise (US)
    
    # Normalization
    normalize_method: str = "zscore"  # "zscore", "minmax", or "robust"
    
    # Train/Val/Test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Class balancing
    balance_classes: bool = True
    oversample_ratio: float = 1.0  # Target ratio of seizure to non-seizure
    
    @property
    def window_size_samples(self) -> int:
        return int(self.window_size_sec * self.sampling_rate)
    
    @property
    def overlap_samples(self) -> int:
        return int(self.window_size_samples * self.overlap_ratio)


@dataclass
class ModelConfig:
    """EEG-Conformer Model Configuration"""
    # Input dimensions
    num_channels: int = 23
    sequence_length: int = 1024  # 4 seconds at 256 Hz
    
    # These are the throttled values, and they are deliberate. Restoring the
    # pre-throttle capacity (40/40, depth 6, heads 8, mlp 4.0, dropout 0.3,
    # hidden 256, lr 1e-4) was measured on 2026-08-17: 340,530 params scored
    # AUC 0.7574 +/- 0.006 against 0.7927 +/- 0.022 for the 121k model, with
    # precision up (0.32 -> 0.55) and recall down (0.52 -> 0.33) -- classic
    # overfitting. 24 patients is still too few for that much capacity.
    # spatial_conv_filters MUST equal transformer_dim: PatchEmbedding feeds the
    # conformer blocks directly, with no projection between them.

    # Temporal Convolution Block
    temp_conv_filters: List[int] = field(default_factory=lambda: [32, 32])
    temp_conv_kernels: List[int] = field(default_factory=lambda: [25, 15])
    temp_pool_size: int = 8

    # Spatial Convolution Block
    spatial_conv_filters: int = 32
    spatial_dropout: float = 0.6

    # Transformer Block
    transformer_dim: int = 32
    transformer_depth: int = 4
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 2.0
    transformer_dropout: float = 0.5
    attention_dropout: float = 0.4

    # Classification Head
    hidden_dim: int = 128
    num_classes: int = 2  # Seizure vs Non-seizure
    classifier_dropout: float = 0.6

    # Regularization
    label_smoothing: float = 0.15
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 5e-5   # see ModelConfig note -- 1e-4 measured worse
    weight_decay: float = 1e-3
    
    # Optimizer
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 3  # Reduced from 5
    min_lr: float = 1e-7
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10  # Reduced from 15
    min_delta: float = 1e-4
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Mixed precision training
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    

@dataclass 
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment
    experiment_name: str = "eeg_conformer_seizure_detection"
    log_dir: str = "logs"
    
    def __post_init__(self):
        # Ensure directories exist
        os.makedirs(self.training.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Sync parameters
        self.model.num_channels = self.data.num_channels
        self.model.sequence_length = self.data.window_size_samples


def get_config() -> Config:
    """Get default configuration"""
    return Config()


if __name__ == "__main__":
    config = get_config()
    print(f"Window size: {config.data.window_size_samples} samples")
    print(f"Model input: {config.model.num_channels} x {config.model.sequence_length}")
