"""
Main training script for the Informer model.

Usage:
    python main.py --config config.yaml
"""

import os
import argparse
import yaml
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from torch.cuda.amp.grad_scaler import GradScaler
import torch.backends.cuda
import torch.backends.cudnn

from data.preprocessing import DataPreprocessor
from data.dataset import InformerDataset, DatasetSplitter
from model.informer import Informer
from model.config import InformerConfig
from training.trainer import Trainer
from utils.cuda_utils import verify_cuda_environment, get_device
from utils.metrics import ModelEvaluator
from utils.visualization import TimeSeriesVisualizer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def setup_logging(config: dict):
    """Setup logging configuration"""
    log_dir = config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config: dict, logger: logging.Logger):
    """Prepare datasets and dataloaders"""
    logger.info("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(config['data_path'], parse_dates=[config['timestamp_col']])
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    features, preprocess_info = preprocessor.preprocess(
        df,
        timestamp_col=config['timestamp_col'],
        target_col=config['target_col'],
        is_training=True
    )
    
    # Split data
    train_data, val_data, test_data = DatasetSplitter.split_data(
        features,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )
    
    # Create datasets
    train_dataset = InformerDataset(
        train_data,
        input_window=config['input_window'],
        prediction_window=config['prediction_window'],
        stride=config['stride']
    )
    
    val_dataset = InformerDataset(
        val_data,
        input_window=config['input_window'],
        prediction_window=config['prediction_window'],
        stride=config['stride']
    )
    
    test_dataset = InformerDataset(
        test_data,
        input_window=config['input_window'],
        prediction_window=config['prediction_window'],
        stride=config['stride']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, preprocessor

def init_model(config: dict, device: torch.device):
    """Initialize model and optimizer"""
    model = Informer(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        e_layers=config['e_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        activation=config['activation'],
        distil=config['distil'],
        prediction_window=config['prediction_window']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    criterion = torch.nn.MSELoss()
    
    return model, optimizer, criterion

def main():
    torch.cuda.set_per_process_memory_fraction(0.5)  # Helps prevent fragmentation
    torch.backends.cudnn.benchmark = True            # Optimize CUDA operations
    torch.backends.cudnn.deterministic = True        # More memory efficient
    
    scaler = GradScaler()
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Check CUDA environment
    device = get_device(use_cuda=config['use_cuda'])
    logger.info("CUDA Environment:")
    logger.info(verify_cuda_environment())
    
    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Prepare data
    train_loader, val_loader, test_loader, preprocessor = prepare_data(config, logger)
    
    # Initialize model
    model, optimizer, criterion = init_model(config, device)
    
    # Initialize visualizer early
    visualizer = TimeSeriesVisualizer(figsize=(15, 7))
    
    # Initialize trainer with visualizer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
        visualizer=visualizer  # Pass visualizer to trainer
    )
    
    # Train model (now includes per-epoch visualization)
    logger.info("Starting training...")
    train_losses, val_losses = trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(model, device)
    test_metrics, predictions, targets = evaluator.evaluate(test_loader)
    
    # Log results
    logger.info("Test Metrics:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Visualize results
    visualizer = TimeSeriesVisualizer(figsize=(15, 7))

    # Plot overall predictions
    pred_fig = visualizer.plot_predictions(
        true_values=targets,
        predictions=predictions,
        title="Overall Prediction Performance"
    )
    visualizer.save_figure(pred_fig, os.path.join(config['output_dir'], 'predictions.png'))

    # Plot sample predictions from batches
    batch_fig = visualizer.plot_batch_predictions(
        true_values=targets[:32],  # Take first batch
        predictions=predictions[:32],
        n_samples=5,
        title="Sample Prediction Cases"
    )
    visualizer.save_figure(batch_fig, os.path.join(config['output_dir'], 'sample_predictions.png'))

    # Plot training history
    history_fig = visualizer.plot_training_history(
        train_losses=train_losses,
        val_losses=val_losses
    )
    visualizer.save_figure(history_fig, os.path.join(config['output_dir'], 'training_history.png'))
    
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()