"""
Training Pipeline for EEG-Conformer Seizure Detection

Features:
- Mixed precision training
- Cosine annealing with warmup
- Early stopping
- TensorBoard logging
- Model checkpointing
- Comprehensive metrics
"""
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from config import Config, TrainingConfig, ModelConfig
from model import EEGConformer, create_model


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Reduces loss for well-classified examples, focusing on hard cases
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(
        self, 
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    Trainer class for EEG-Conformer model
    """
    
    def __init__(
        self,
        model: EEGConformer,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        print(f"\nTraining on: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Loss function (Focal Loss for class imbalance)
        self.criterion = FocalLoss(
            alpha=0.25, 
            gamma=2.0,
            label_smoothing=config.model.label_smoothing
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            min_lr=config.training.min_lr
        )
        
        # Mixed precision training
        self.use_amp = config.training.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='max'
        ) if config.training.early_stopping else None
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(config.log_dir) / f"{config.experiment_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_f1 = 0.0
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': []
        }
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay"""
        tc = self.config.training
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': tc.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if tc.optimizer == 'adamw':
            return optim.AdamW(param_groups, lr=tc.learning_rate, betas=tc.betas)
        elif tc.optimizer == 'adam':
            return optim.Adam(param_groups, lr=tc.learning_rate, betas=tc.betas)
        else:
            return optim.SGD(param_groups, lr=tc.learning_rate, momentum=0.9)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_lr():.2e}'
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, loader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        """Validate on a data loader"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(loader, desc=f"[{desc}]")
        
        for data, labels in pbar:
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / len(loader)
        
        return metrics
    
    def _calculate_metrics(
        self, 
        labels: list, 
        preds: list, 
        probs: Optional[list] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'specificity': self._specificity(labels, preds)
        }
        
        if probs is not None and len(np.unique(labels)) > 1:
            try:
                metrics['auc'] = roc_auc_score(labels, probs)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def _specificity(self, labels: list, preds: list) -> float:
        """Calculate specificity (true negative rate)"""
        cm = confusion_matrix(labels, preds)
        if cm.shape[0] < 2:
            return 0.0
        tn, fp = cm[0, 0], cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def train(self) -> Dict[str, list]:
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            # Update learning rate
            self.scheduler.step(epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader, desc="Val")
            
            # Log to TensorBoard
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(self.scheduler.get_lr())
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  [*] New best model! F1: {self.best_val_f1:.4f}")
            
            # Early stopping
            if self.early_stopping and self.early_stopping(val_metrics['f1']):
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training complete
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total time: {training_time/60:.1f} minutes")
        print(f"  Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        # Final evaluation on test set
        if self.test_loader is not None:
            self._final_evaluation()
        
        # Save training history
        self._save_history()
        
        self.writer.close()
        
        return self.history
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard"""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        self.writer.add_scalar('lr', self.scheduler.get_lr(), epoch)
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, float], 
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            }
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    
    def _final_evaluation(self):
        """Evaluate on test set"""
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set")
        print(f"{'='*60}")
        
        # Load best model
        best_path = self.save_dir / 'best.pt'
        if best_path.exists():
            self.load_checkpoint(str(best_path))
        
        test_metrics = self.validate(self.test_loader, desc="Test")
        
        print(f"\nTest Results:")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Precision:   {test_metrics['precision']:.4f}")
        print(f"  Recall:      {test_metrics['recall']:.4f}")
        print(f"  F1 Score:    {test_metrics['f1']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
        print(f"  AUC-ROC:     {test_metrics.get('auc', 0):.4f}")
        
        # Get predictions for confusion matrix
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
        
        # Print classification report
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds, 
                                    target_names=['Non-Seizure', 'Seizure']))
        
        # Save test results
        results = {
            'metrics': test_metrics,
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.log_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_history(self):
        """Save training history"""
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    from config import get_config
    from data_loader import create_data_loaders
    
    # Get configuration
    config = get_config()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data, config.training
    )
    
    # Create model
    model = create_model(config.model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Train
    history = trainer.train()
