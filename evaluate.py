"""
Evaluation and Visualization Module for EEG Seizure Detection

Features:
- Comprehensive metrics computation
- ROC and Precision-Recall curves
- Confusion matrix visualization
- Attention map visualization
- Training history plots
- EEG signal visualization with predictions
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from config import Config, get_config
from model import EEGConformer


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(
        self,
        model: EEGConformer,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        save_dir: str = "evaluation_results"
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        self.all_features = []
        
    @torch.no_grad()
    def run_inference(self) -> Dict[str, np.ndarray]:
        """Run inference on test set"""
        print("Running inference on test set...")
        
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        
        for data, labels in tqdm(self.test_loader):
            data = data.to(self.device)
            
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            self.all_labels.extend(labels.numpy())
            self.all_preds.extend(preds.cpu().numpy())
            self.all_probs.extend(probs[:, 1].cpu().numpy())
        
        self.all_labels = np.array(self.all_labels)
        self.all_preds = np.array(self.all_preds)
        self.all_probs = np.array(self.all_probs)
        
        return {
            'labels': self.all_labels,
            'predictions': self.all_preds,
            'probabilities': self.all_probs
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive metrics"""
        if len(self.all_labels) == 0:
            self.run_inference()
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(self.all_labels, self.all_preds),
            'precision': precision_score(self.all_labels, self.all_preds, zero_division=0),
            'recall': recall_score(self.all_labels, self.all_preds, zero_division=0),
            'f1_score': f1_score(self.all_labels, self.all_preds, zero_division=0),
        }
        
        # Confusion matrix based metrics
        cm = confusion_matrix(self.all_labels, self.all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        
        # AUC metrics
        if len(np.unique(self.all_labels)) > 1:
            metrics['auc_roc'] = roc_auc_score(self.all_labels, self.all_probs)
            
            precision_curve, recall_curve, _ = precision_recall_curve(
                self.all_labels, self.all_probs
            )
            metrics['auc_pr'] = auc(recall_curve, precision_curve)
        
        # False alarm rate (per hour, assuming 4-second windows)
        total_non_seizure_windows = tn + fp
        total_hours = (total_non_seizure_windows * 4) / 3600  # 4-second windows
        metrics['false_alarm_rate_per_hour'] = fp / total_hours if total_hours > 0 else 0
        
        return metrics
    
    def print_report(self):
        """Print detailed evaluation report"""
        metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\n{'Performance Metrics':^60}")
        print("-"*60)
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision (PPV): {metrics['precision']:.4f}")
        print(f"  Recall (Sens.):  {metrics['recall']:.4f}")
        print(f"  Specificity:     {metrics['specificity']:.4f}")
        print(f"  F1 Score:        {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:         {metrics.get('auc_roc', 'N/A'):.4f}")
        print(f"  AUC-PR:          {metrics.get('auc_pr', 'N/A'):.4f}")
        
        print(f"\n{'Confusion Matrix':^60}")
        print("-"*60)
        print(f"  True Negatives:  {metrics['true_negatives']:,}")
        print(f"  False Positives: {metrics['false_positives']:,}")
        print(f"  False Negatives: {metrics['false_negatives']:,}")
        print(f"  True Positives:  {metrics['true_positives']:,}")
        
        print(f"\n{'Clinical Metrics':^60}")
        print("-"*60)
        print(f"  False Alarm Rate: {metrics['false_alarm_rate_per_hour']:.3f} per hour")
        print(f"  Sensitivity:      {metrics['sensitivity']:.4f}")
        print(f"  Specificity:      {metrics['specificity']:.4f}")
        
        print("\n" + "="*60)
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=['Non-Seizure', 'Seizure']
        ))
        
        return metrics
    
    def plot_confusion_matrix(self, save: bool = True) -> plt.Figure:
        """Plot confusion matrix heatmap"""
        if len(self.all_labels) == 0:
            self.run_inference()
        
        cm = confusion_matrix(self.all_labels, self.all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Seizure', 'Seizure'],
            yticklabels=['Non-Seizure', 'Seizure'],
            ax=axes[0]
        )
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix (Counts)')
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Non-Seizure', 'Seizure'],
            yticklabels=['Non-Seizure', 'Seizure'],
            ax=axes[1]
        )
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, save: bool = True) -> plt.Figure:
        """Plot ROC curve"""
        if len(self.all_labels) == 0:
            self.run_inference()
        
        fpr, tpr, thresholds = roc_curve(self.all_labels, self.all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Mark optimal point
        ax.scatter(
            fpr[optimal_idx], tpr[optimal_idx], 
            marker='o', s=100, c='green', 
            label=f'Optimal (threshold={optimal_threshold:.3f})'
        )
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, save: bool = True) -> plt.Figure:
        """Plot Precision-Recall curve"""
        if len(self.all_labels) == 0:
            self.run_inference()
        
        precision, recall, thresholds = precision_recall_curve(
            self.all_labels, self.all_probs
        )
        pr_auc = auc(recall, precision)
        
        # Baseline (prevalence)
        baseline = np.sum(self.all_labels) / len(self.all_labels)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision (PPV)', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'precision_recall_curve.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_probability_distribution(self, save: bool = True) -> plt.Figure:
        """Plot probability distribution for each class"""
        if len(self.all_labels) == 0:
            self.run_inference()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        seizure_probs = self.all_probs[self.all_labels == 1]
        non_seizure_probs = self.all_probs[self.all_labels == 0]
        
        axes[0].hist(non_seizure_probs, bins=50, alpha=0.7, label='Non-Seizure', color='blue')
        axes[0].hist(seizure_probs, bins=50, alpha=0.7, label='Seizure', color='red')
        axes[0].axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        axes[0].set_xlabel('Predicted Probability of Seizure')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Probability Distribution by Class')
        axes[0].legend()
        
        # KDE plot
        sns.kdeplot(non_seizure_probs, ax=axes[1], label='Non-Seizure', fill=True, alpha=0.5)
        sns.kdeplot(seizure_probs, ax=axes[1], label='Seizure', fill=True, alpha=0.5)
        axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[1].set_xlabel('Predicted Probability of Seizure')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Probability Density by Class')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'probability_distribution.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_threshold_analysis(self, save: bool = True) -> plt.Figure:
        """Analyze metrics across different thresholds"""
        if len(self.all_labels) == 0:
            self.run_inference()
        
        thresholds = np.linspace(0.01, 0.99, 100)
        
        precisions = []
        recalls = []
        f1s = []
        specificities = []
        
        for thresh in thresholds:
            preds = (self.all_probs >= thresh).astype(int)
            
            precisions.append(precision_score(self.all_labels, preds, zero_division=0))
            recalls.append(recall_score(self.all_labels, preds, zero_division=0))
            f1s.append(f1_score(self.all_labels, preds, zero_division=0))
            
            cm = confusion_matrix(self.all_labels, preds)
            tn, fp = cm[0, 0], cm[0, 1]
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall (Sensitivity)', linewidth=2)
        ax.plot(thresholds, f1s, label='F1 Score', linewidth=2)
        ax.plot(thresholds, specificities, label='Specificity', linewidth=2)
        
        # Mark optimal F1
        optimal_idx = np.argmax(f1s)
        ax.axvline(x=thresholds[optimal_idx], color='gray', linestyle='--', alpha=0.7)
        ax.scatter(thresholds[optimal_idx], f1s[optimal_idx], s=100, c='red', zorder=5,
                  label=f'Optimal F1 @ {thresholds[optimal_idx]:.2f}')
        
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics vs Classification Threshold', fontsize=14)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_results(self, metrics: Dict[str, float]):
        """Save all results to JSON"""
        results = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items()},
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.all_labels),
            'class_distribution': {
                'non_seizure': int(np.sum(self.all_labels == 0)),
                'seizure': int(np.sum(self.all_labels == 1))
            }
        }
        
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {self.save_dir}")
    
    def full_evaluation(self):
        """Run complete evaluation with all visualizations"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Run inference
        self.run_inference()
        
        # Compute and print metrics
        metrics = self.print_report()
        
        # Generate all plots
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_probability_distribution()
        self.plot_threshold_analysis()
        
        # Save results
        self.save_results(metrics)
        
        print("\n[OK] Evaluation complete!")
        print(f"  Results saved to: {self.save_dir}")
        
        return metrics


def plot_training_history(history_path: str, save_dir: str = "evaluation_results"):
    """Plot training history from saved JSON file"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    
    print(f"Training history plot saved to: {save_dir / 'training_history.png'}")
    
    return fig


def visualize_eeg_predictions(
    model: EEGConformer,
    data_loader: DataLoader,
    num_samples: int = 5,
    save_dir: str = "evaluation_results"
):
    """Visualize EEG signals with model predictions"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get samples
    samples_shown = 0
    
    for data, labels in data_loader:
        for i in range(len(data)):
            if samples_shown >= num_samples:
                return
            
            eeg = data[i].numpy()  # (channels, time)
            label = labels[i].item()
            
            # Get prediction
            with torch.no_grad():
                output = model(data[i:i+1].to(device))
                prob = torch.softmax(output, dim=1)[0, 1].item()
                pred = 1 if prob > 0.5 else 0
            
            # Create figure
            fig, axes = plt.subplots(5, 5, figsize=(20, 15))
            axes = axes.flatten()
            
            # Plot each channel
            time = np.arange(eeg.shape[1]) / 256  # Assuming 256 Hz
            
            for ch in range(min(23, len(axes) - 2)):
                axes[ch].plot(time, eeg[ch], linewidth=0.5, color='blue')
                axes[ch].set_title(f'Channel {ch+1}', fontsize=8)
                axes[ch].set_xlim([0, time[-1]])
                axes[ch].tick_params(axis='both', labelsize=6)
            
            # Hide unused subplots
            for idx in range(23, len(axes)):
                axes[idx].axis('off')
            
            # Overall title with prediction
            status = "CORRECT" if pred == label else "INCORRECT"
            color = "green" if pred == label else "red"
            true_label = "Seizure" if label == 1 else "Non-Seizure"
            pred_label = "Seizure" if pred == 1 else "Non-Seizure"
            
            fig.suptitle(
                f'{status}\n'
                f'True: {true_label} | Predicted: {pred_label} (p={prob:.3f})',
                fontsize=14,
                color=color
            )
            
            plt.tight_layout()
            fig.savefig(save_dir / f'eeg_sample_{samples_shown+1}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            samples_shown += 1
    
    print(f"EEG visualizations saved to: {save_dir}")


if __name__ == "__main__":
    # Example usage
    from config import get_config
    from data_loader import create_data_loaders
    from model import create_model
    
    config = get_config()
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(config.data, config.training)
    
    # Create and load model
    model = create_model(config.model)
    
    # Check if trained model exists
    best_model_path = Path(config.training.save_dir) / 'best.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {best_model_path}")
    
    # Run evaluation
    evaluator = Evaluator(model, test_loader)
    evaluator.full_evaluation()
