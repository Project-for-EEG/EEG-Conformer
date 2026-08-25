"""
EEG Seizure Detection using SOTA EEG-Conformer Model

Main entry point for training, evaluation, and inference.

Usage:
    python main.py train                    # Train the model
    python main.py evaluate                 # Evaluate trained model
    python main.py predict --file FILE      # Predict on new data
    python main.py visualize                # Generate visualizations
    
Author: AI Assistant
Date: 2024
"""
import argparse
import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        EEG SEIZURE DETECTION - SOTA EEG-CONFORMER            ║
    ║                                                              ║
    ║   State-of-the-Art Deep Learning for Epilepsy Detection      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if all required packages are installed"""
    required = [
        'torch', 'numpy', 'pandas', 'scipy', 'sklearn', 
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    return True


def train(args):
    """Train the EEG-Conformer model"""
    from config import get_config
    from data_loader import create_data_loaders
    from model import create_model
    from trainer import Trainer
    
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Get configuration
    config = get_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Set seed
    set_seed(config.training.seed)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Data directory: {config.data.data_dir}")
    print(f"  Window size: {config.data.window_size_sec}s ({config.data.window_size_samples} samples)")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create data loaders
    print("\n" + "-"*60)
    print("Loading and preprocessing data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data, config.training
    )
    
    # Create model
    print("\n" + "-"*60)
    print("Creating model...")
    model = create_model(config.model)
    
    # Resume training if checkpoint exists and requested
    if args.resume:
        checkpoint_path = Path(config.training.save_dir) / 'latest.pt'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
    
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
    
    print("\n[OK] Training complete!")
    print(f"  Best model saved to: {config.training.save_dir}/best.pt")
    print(f"  Logs saved to: {trainer.log_dir}")
    
    return history


def evaluate(args):
    """Evaluate the trained model"""
    from config import get_config
    from data_loader import create_data_loaders
    from model import create_model
    from evaluate import Evaluator, plot_training_history
    
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    config = get_config()
    set_seed(config.training.seed)
    
    # Create data loaders
    print("\nLoading data...")
    _, _, test_loader = create_data_loaders(config.data, config.training)
    
    # Create model
    model = create_model(config.model)
    
    # Load best model
    model_path = args.model if args.model else Path(config.training.save_dir) / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"\n[X] Model not found: {model_path}")
        print("  Please train a model first using: python main.py train")
        return None
    
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    save_dir = args.output if args.output else "evaluation_results"
    evaluator = Evaluator(model, test_loader, save_dir=save_dir)
    
    # Run evaluation
    metrics = evaluator.full_evaluation()
    
    # Plot training history if available
    history_path = None
    for log_dir in Path(config.log_dir).iterdir():
        hist_file = log_dir / 'history.json'
        if hist_file.exists():
            history_path = hist_file
            break
    
    if history_path:
        print(f"\nPlotting training history from: {history_path}")
        plot_training_history(str(history_path), save_dir)
    
    return metrics


def predict(args):
    """Make predictions on new EEG data"""
    from config import get_config
    from data_loader import EEGDataLoader, EEGPreprocessor
    from model import create_model
    import pandas as pd
    
    print("\n" + "="*60)
    print("PREDICTION MODE")
    print("="*60)
    
    config = get_config()
    
    # Check input file
    if not Path(args.file).exists():
        print(f"\n[X] File not found: {args.file}")
        return None
    
    # Load model
    model_path = args.model if args.model else Path(config.training.save_dir) / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"\n[X] Model not found: {model_path}")
        return None
    
    print(f"\nLoading model from: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config.model)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess data
    print(f"Loading data from: {args.file}")
    loader = EEGDataLoader(config.data)
    eeg_data, labels = loader.load_csv(args.file)
    
    # Preprocess
    preprocessor = EEGPreprocessor(config.data)
    eeg_data = preprocessor.preprocess(eeg_data, fit=True)
    
    # Segment
    segments, segment_labels = loader.segment_data(eeg_data, labels, overlap=False)
    
    print(f"\nAnalyzing {len(segments)} segments...")
    
    # Predict
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(segments), config.training.batch_size):
            batch = segments[i:i+config.training.batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'segment_id': range(len(all_preds)),
        'start_time': [i * config.data.window_size_sec for i in range(len(all_preds))],
        'end_time': [(i + 1) * config.data.window_size_sec for i in range(len(all_preds))],
        'true_label': segment_labels,
        'predicted': all_preds,
        'seizure_probability': all_probs
    })
    
    # Summary
    seizure_segments = sum(all_preds)
    total_segments = len(all_preds)
    
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"  Total segments:   {total_segments}")
    print(f"  Seizure detected: {seizure_segments} ({100*seizure_segments/total_segments:.1f}%)")
    print(f"  Normal:           {total_segments - seizure_segments} ({100*(total_segments-seizure_segments)/total_segments:.1f}%)")
    
    # Show seizure segments
    seizure_df = results_df[results_df['predicted'] == 1]
    if len(seizure_df) > 0:
        print(f"\nSeizure activity detected at:")
        for _, row in seizure_df.iterrows():
            print(f"  {row['start_time']:.1f}s - {row['end_time']:.1f}s "
                  f"(probability: {row['seizure_probability']:.3f})")
    else:
        print("\n[OK] No seizure activity detected")
    
    # Save results
    output_path = args.output if args.output else "predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df


def visualize(args):
    """Generate visualizations"""
    from config import get_config
    from data_loader import create_data_loaders
    from model import create_model
    from evaluate import Evaluator, visualize_eeg_predictions, plot_training_history
    
    print("\n" + "="*60)
    print("VISUALIZATION MODE")
    print("="*60)
    
    config = get_config()
    set_seed(config.training.seed)
    
    save_dir = args.output if args.output else "visualizations"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_path = args.model if args.model else Path(config.training.save_dir) / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"\n[X] Model not found: {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    
    model = create_model(config.model)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    _, _, test_loader = create_data_loaders(config.data, config.training)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Evaluation plots
    evaluator = Evaluator(model, test_loader, save_dir=save_dir)
    evaluator.run_inference()
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    evaluator.plot_precision_recall_curve()
    evaluator.plot_probability_distribution()
    evaluator.plot_threshold_analysis()
    
    # EEG sample visualizations
    visualize_eeg_predictions(model, test_loader, num_samples=args.num_samples, save_dir=save_dir)
    
    # Training history if available
    for log_dir in Path(config.log_dir).iterdir():
        hist_file = log_dir / 'history.json'
        if hist_file.exists():
            plot_training_history(str(hist_file), save_dir)
            break
    
    print(f"\n[OK] Visualizations saved to: {save_dir}")


def info():
    """Display system and model information"""
    from config import get_config
    from model import create_model, count_parameters
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # System info
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model info
    config = get_config()
    model = create_model(config.model)
    
    print(f"\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"\nEEG-Conformer Configuration:")
    print(f"  Input: {config.model.num_channels} channels × {config.model.sequence_length} samples")
    print(f"  Temporal conv filters: {config.model.temp_conv_filters}")
    print(f"  Transformer depth: {config.model.transformer_depth}")
    print(f"  Transformer heads: {config.model.transformer_heads}")
    print(f"  Transformer dim: {config.model.transformer_dim}")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Data info
    print(f"\n" + "="*60)
    print("DATA CONFIGURATION")
    print("="*60)
    print(f"\n  Data directory: {config.data.data_dir}")
    print(f"  Sampling rate: {config.data.sampling_rate} Hz")
    print(f"  Window size: {config.data.window_size_sec}s")
    print(f"  Bandpass filter: {config.data.lowcut}-{config.data.highcut} Hz")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EEG Seizure Detection using SOTA EEG-Conformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                     Train the model
  python main.py train --epochs 50         Train for 50 epochs
  python main.py evaluate                  Evaluate trained model
  python main.py predict --file data.csv   Predict on new data
  python main.py visualize                 Generate visualizations
  python main.py info                      Show system info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    eval_parser.add_argument('--output', type=str, help='Output directory')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict on new data')
    pred_parser.add_argument('--file', type=str, required=True, help='Input CSV file')
    pred_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    pred_parser.add_argument('--output', type=str, help='Output file path')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    viz_parser.add_argument('--output', type=str, help='Output directory')
    viz_parser.add_argument('--num-samples', type=int, default=5, help='Number of EEG samples to visualize')
    
    # Info command
    subparsers.add_parser('info', help='Display system and model info')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'visualize':
        visualize(args)
    elif args.command == 'info':
        info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
