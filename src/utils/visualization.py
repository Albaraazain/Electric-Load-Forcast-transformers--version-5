"""
Visualization utilities for time series forecasting.

Dependencies:
- matplotlib>=3.7.2
- seaborn>=0.12.2
- numpy>=1.24.3
- torch>=2.0.1
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.dates import date2num, num2date
import seaborn as sns
import numpy as np
import numpy.typing as npt
import torch
from typing import List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.dates import date2num
import seaborn as sns
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from datetime import datetime, timedelta
import os
import pandas as pd

class PredictionVisualizer:
    """Handles visualization of model predictions with input windows"""
    
    def __init__(
            self,
            output_dir: str = "prediction_plots",
            fig_size: Tuple[int, int] = (15, 7),
            dpi: int = 300
    ):
        self.output_dir = output_dir
        self.fig_size = fig_size
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn')
    
    def _convert_time_to_num(self, time_point: Union[datetime, int]) -> float:
        """Helper to safely convert datetime or int to float for plotting"""
        if isinstance(time_point, datetime):
            return date2num(time_point)
        return float(time_point)
    
    def _generate_timestamps(
            self,
            start_time: datetime,
            sequence_length: int,
            freq: str = 'H'
    ) -> List[datetime]:
        """Generate timestamps list"""
        timestamps = []
        current_time = start_time
        try:
            for _ in range(sequence_length):
                timestamps.append(current_time)
                current_time += timedelta(hours=1 if freq == 'H' else 0)
            return timestamps
        except Exception as e:
            print(f"Error generating timestamps: {str(e)}")
            return [start_time + timedelta(hours=i) for i in range(sequence_length)]
    
    def plot_prediction_sample(
            self,
            input_seq: Union[torch.Tensor, np.ndarray],
            actual_seq: Union[torch.Tensor, np.ndarray],
            predicted_seq: Union[torch.Tensor, np.ndarray],
            sample_id: int,
            start_time: Optional[datetime] = None,
            scaler: Any = None
    ) -> None:
        """Create and save a plot showing input, actual and predicted values"""
        try:
            # Convert to numpy arrays safely
            input_seq = self._ensure_numpy(input_seq)
            actual_seq = self._ensure_numpy(actual_seq)
            predicted_seq = self._ensure_numpy(predicted_seq)
            
            # Apply inverse transform if scaler provided
            if scaler is not None:
                try:
                    input_seq = self._apply_scaler(input_seq, scaler)
                    actual_seq = self._apply_scaler(actual_seq, scaler)
                    predicted_seq = self._apply_scaler(predicted_seq, scaler)
                except Exception as e:
                    print(f"Warning: Inverse transform failed: {str(e)}")
            
            # Create x-axis values
            input_len = len(input_seq)
            prediction_len = len(predicted_seq)
            total_len = input_len + prediction_len
            
            x_values = (self._generate_timestamps(start_time, total_len) 
                       if start_time is not None 
                       else np.arange(total_len))
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=self.fig_size)
            
            # Plot sequences
            ax.plot(x_values[:input_len], input_seq[:, 0], 'b-', 
                   label='Input', alpha=0.5)
            ax.plot(x_values[input_len:], actual_seq[:, 0], 'g-', 
                   label='Actual', linewidth=2)
            ax.plot(x_values[input_len:], predicted_seq[:, 0], 'r--', 
                   label='Predicted', linewidth=2)
            
            # Add vertical separator line
            separator_x = (self._convert_time_to_num(x_values[input_len-1]) 
                         if isinstance(x_values[0], datetime) 
                         else float(x_values[input_len-1]))
            ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.5)
            
            # Customize plot
            ax.set_title('Input Sequence and Predictions', pad=20)
            ax.set_xlabel('Time' if start_time is None else 'Timestamp')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if isinstance(x_values[0], datetime):
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            filename = os.path.join(self.output_dir, f'prediction_sample_{sample_id}.png')
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved prediction plot to {filename}")
        
        except Exception as e:
            print(f"Error creating prediction plot: {str(e)}")
            plt.close('all')

    def _ensure_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Safely convert input to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return np.asarray(data)

    def _apply_scaler(self, data: np.ndarray, scaler: Any) -> np.ndarray:
        """Apply inverse transform using scaler"""
        reshaped = data.reshape(-1, 1)
        transformed = scaler.inverse_transform(reshaped)
        return transformed.reshape(data.shape)
    
    
    
    
    
class TimeSeriesVisualizer:
    """Visualization tools for time series data and model results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """Initialize visualizer"""
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize

    def _prepare_data_for_plotting(
        self, 
        data: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        # Convert to numpy if it's a tensor
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        else:
            data = np.asarray(data)
            
        # Handle different input shapes
        if len(data.shape) == 3:  # [batch, seq_len, features]
            # Take first feature and average across batches
            data = np.mean(data[:, :, 0], axis=0)
        elif len(data.shape) == 2:  # [seq_len, features]
            data = data[:, 0]
            
        # Ensure data is a NumPy array before calling astype
        data = np.asarray(data)
        return data.astype(np.float32)


    def plot_predictions(
        self,
        true_values: Union[torch.Tensor, np.ndarray],
        predictions: Union[torch.Tensor, np.ndarray],
        timestamps: Optional[List] = None,
        title: str = "Predictions vs Actual"
    ) -> Figure:
        """Plot predictions against actual values"""
        # Prepare data for plotting
        true_values = self._prepare_data_for_plotting(true_values)
        predictions = self._prepare_data_for_plotting(predictions)
        
        # Create x-axis values
        x_values = timestamps if timestamps is not None else np.arange(len(true_values))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot data
        ax.plot(x_values, true_values, 'b-', label='Actual', alpha=0.7)
        ax.plot(x_values, predictions, 'r--', label='Predicted', alpha=0.7)
        
        # Customize plot
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

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training History"
    ) -> Figure:
        """Plot training and validation losses"""
        fig, ax = plt.subplots(figsize=self.figsize)
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        importance_scores: np.ndarray,
        feature_names: List[str]
    ) -> Figure:
        """Plot feature importance scores"""
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

    def plot_batch_predictions(
        self,
        true_values: Union[torch.Tensor, np.ndarray],
        predictions: Union[torch.Tensor, np.ndarray],
        batch_idx: int = 0,
        n_samples: int = 5,
        title: str = "Sample Predictions"
    ) -> Figure:
        # Convert to numpy if needed
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.cpu().detach().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()
            
        
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
        
        
        # Ensure axes is always a list of Axes objects
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each sample
        for i, ax in enumerate(axes[:n_samples]):
            true_seq = true_values[i, :, 0]
            pred_seq = predictions[i, :, 0]
            
            ax.plot(true_seq, 'b-', label='Actual', alpha=0.7)
            ax.plot(pred_seq, 'r--', label='Predicted', alpha=0.7)
            
            ax.set_title(f'Sample {i+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig


    @staticmethod
    def save_figure(
        fig: Figure,
        filename: str,
        dpi: int = 300
    ) -> None:
        """Save figure to file"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)