"""
Visualization utilities for time series forecasting.

Dependencies:
- matplotlib>=3.7.2
- seaborn>=0.12.2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple
import torch

class TimeSeriesVisualizer:
    """Visualization tools for time series data and model results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training History"
    ) -> plt.Figure:
        """
        Plot training and validation losses
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_predictions(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        timestamps: Optional[List] = None,
        title: str = "Predictions vs Actual"
    ) -> plt.Figure:
        """
        Plot predictions against actual values
        
        Args:
            true_values: Ground truth values
            predictions: Model predictions
            timestamps: Optional list of timestamps
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        # Convert to numpy if tensors
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_values = timestamps if timestamps is not None else range(len(true_values))
        
        ax.plot(x_values, true_values, 'b-', label='Actual', alpha=0.7)
        ax.plot(x_values, predictions, 'r--', label='Predicted', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Time' if timestamps is None else 'Timestamp')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Rotate x-axis labels if timestamps are provided
        if timestamps is not None:
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        return fig
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        index: int = 0
    ) -> plt.Figure:
        """
        Plot attention weights heatmap
        
        Args:
            attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
            index: Batch index to plot
            
        Returns:
            matplotlib figure
        """
        # Get weights for specified batch
        weights = attention_weights[index].cpu().numpy()
        
        # Create subplot for each attention head
        n_heads = weights.shape[0]
        fig, axes = plt.subplots(
            1, n_heads,
            figsize=(4 * n_heads, 4),
            squeeze=False
        )
        
        for i, ax in enumerate(axes[0]):
            sns.heatmap(
                weights[i],
                ax=ax,
                cmap='viridis',
                cbar=True
            )
            ax.set_title(f'Head {i+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
            
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(
        self,
        importance_scores: np.ndarray,
        feature_names: List[str]
    ) -> plt.Figure:
        """
        Plot feature importance scores
        
        Args:
            importance_scores: Array of importance scores
            feature_names: List of feature names
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        ax.barh(pos, importance_scores[sorted_idx])
        ax.set_yticks(pos)
        ax.set_yticklabels(np.array(feature_names)[sorted_idx])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(
        fig: plt.Figure,
        filename: str,
        dpi: int = 300
    ):
        """
        Save figure to file
        
        Args:
            fig: matplotlib figure
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        
        

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from typing import Tuple, Optional, List

class PredictionVisualizer:
    """Handles visualization of model predictions with input windows"""
    
    def __init__(
            self,
            output_dir: str = "prediction_plots",
            fig_size: Tuple[int, int] = (15, 7),
            dpi: int = 300
    ):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
            fig_size: Figure size for plots
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir
        self.fig_size = fig_size
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
    
    def _generate_timestamps(
            self,
            start_time: datetime,
            sequence_length: int,
            freq: str = 'H'
    ) -> List[datetime]:
        """
        Generate timestamps for x-axis
        
        Args:
            start_time: Starting timestamp
            sequence_length: Number of timestamps to generate
            freq: Frequency of timestamps ('H' for hourly)
            
        Returns:
            List of timestamps
        """
        try:
            return [start_time + timedelta(hours=i) for i in range(sequence_length)]
        except Exception as e:
            print(f"Error generating timestamps: {str(e)}")
            # Return fallback numeric x-axis
            return [start_time + timedelta(hours=i) for i in range(sequence_length)]
    
    def plot_prediction_sample(
            self,
            input_seq: torch.Tensor,
            actual_seq: torch.Tensor,
            predicted_seq: torch.Tensor,
            sample_id: int,
            start_time: Optional[datetime] = None,
            scaler = None
    ):
        """
        Create and save a plot showing input, actual and predicted values
        
        Args:
            input_seq: Input sequence tensor
            actual_seq: Actual values tensor
            predicted_seq: Predicted values tensor
            sample_id: Sample identifier for filename
            start_time: Starting timestamp (optional)
            scaler: Scaler object for inverse transform (optional)
        """
        try:
            # Move tensors to CPU and convert to numpy
            input_seq = input_seq.cpu().numpy()
            actual_seq = actual_seq.cpu().numpy()
            predicted_seq = predicted_seq.cpu().numpy()
            
            # If we have a scaler, inverse transform the values
            if scaler is not None:
                try:
                    input_seq = scaler.inverse_transform(input_seq.reshape(-1, 1)).reshape(input_seq.shape)
                    actual_seq = scaler.inverse_transform(actual_seq.reshape(-1, 1)).reshape(actual_seq.shape)
                    predicted_seq = scaler.inverse_transform(predicted_seq.reshape(-1, 1)).reshape(predicted_seq.shape)
                except Exception as e:
                    print(f"Warning: Could not apply inverse transform: {str(e)}")
            
            # Get sequence lengths
            input_len = len(input_seq)
            prediction_len = len(predicted_seq)
            total_len = input_len + prediction_len
            
            # Generate x-axis values
            if start_time is not None:
                x_values = self._generate_timestamps(start_time, total_len)
            else:
                x_values = list(range(total_len))
            
            # Create figure
            plt.figure(figsize=self.fig_size)
            
            # Plot input sequence
            plt.plot(x_values[:input_len], 
                    input_seq[:, 0], 
                    'b-', 
                    label='Input', 
                    alpha=0.5)
            
            # Plot actual values
            plt.plot(x_values[input_len:], 
                    actual_seq[:, 0], 
                    'g-', 
                    label='Actual', 
                    linewidth=2)
            
            # Plot predictions
            plt.plot(x_values[input_len:], 
                    predicted_seq[:, 0], 
                    'r--', 
                    label='Predicted', 
                    linewidth=2)
            
            # Add vertical line separating input and prediction
            plt.axvline(x=x_values[input_len-1], 
                       color='gray', 
                       linestyle='--', 
                       alpha=0.5)
            
            # Customize plot
            plt.title('Input Sequence and Predictions', pad=20)
            plt.xlabel('Time' if isinstance(x_values[0], datetime) else 'Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if using timestamps
            if isinstance(x_values[0], datetime):
                plt.xticks(rotation=45)
            
            # Tight layout to prevent label cutoff
            plt.tight_layout()
            
            # Save plot
            filename = os.path.join(self.output_dir, f'prediction_sample_{sample_id}.png')
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"Saved prediction plot to {filename}")
            
        except Exception as e:
            print(f"Error creating prediction plot: {str(e)}")
            # Ensure figure is closed even if error occurs
            plt.close()
    
    def plot_multiple_samples(
            self,
            model: torch.nn.Module,
            dataset,
            num_samples: int = 5,
            scaler = None,
            device: torch.device = torch.device('cpu')
    ):
        """
        Create plots for multiple random samples from dataset
        
        Args:
            model: Trained model
            dataset: Dataset containing samples
            num_samples: Number of samples to plot
            scaler: Scaler object for inverse transform
            device: Device to run model on
        """
        try:
            model.eval()  # Set model to evaluation mode
            
            # Generate random indices
            total_samples = len(dataset)
            indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
            
            with torch.no_grad():
                for idx in indices:
                    # Get sample from dataset
                    input_seq, decoder_input, target = dataset[idx]
                    
                    # Prepare input for model
                    input_batch = input_seq.unsqueeze(0).to(device)
                    
                    # Get prediction
                    prediction, _ = model(input_batch)
                    
                    # Remove batch dimension
                    prediction = prediction.squeeze(0)
                    
                    # Plot sample
                    self.plot_prediction_sample(
                        input_seq=input_seq,
                        actual_seq=target,
                        predicted_seq=prediction,
                        sample_id=idx,
                        scaler=scaler
                    )
                    
        except Exception as e:
            print(f"Error plotting multiple samples: {str(e)}")

    def create_error_analysis_plots(
            self,
            actual_values: torch.Tensor,
            predicted_values: torch.Tensor,
            scaler = None
    ):
        """
        Create error analysis plots (error distribution, scatter plot)
        
        Args:
            actual_values: Tensor of actual values
            predicted_values: Tensor of predicted values
            scaler: Scaler object for inverse transform
        """
        try:
            # Convert to numpy and reshape
            actual = actual_values.cpu().numpy().reshape(-1)
            predicted = predicted_values.cpu().numpy().reshape(-1)
            
            if scaler is not None:
                try:
                    actual = scaler.inverse_transform(actual.reshape(-1, 1)).reshape(-1)
                    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(-1)
                except Exception as e:
                    print(f"Warning: Could not apply inverse transform: {str(e)}")
            
            # Calculate errors
            errors = predicted - actual
            
            # Error distribution plot
            plt.figure(figsize=self.fig_size)
            plt.hist(errors, bins=50, edgecolor='black')
            plt.title('Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'), 
                       dpi=self.dpi, 
                       bbox_inches='tight')
            plt.close()
            
            # Scatter plot
            plt.figure(figsize=self.fig_size)
            plt.scatter(actual, predicted, alpha=0.5)
            z
            # Add diagonal line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'), 
                       dpi=self.dpi, 
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating analysis plots: {str(e)}")
            plt.close()