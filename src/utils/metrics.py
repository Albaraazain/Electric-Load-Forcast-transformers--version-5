"""
Evaluation metrics for time series forecasting.

Dependencies:
- torch>=2.0.1
- numpy>=1.24.3
- sklearn>=1.3.0
"""

import torch
import numpy as np
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesMetrics:
    """Metrics for evaluating time series forecasting models"""
    
    @staticmethod
    def _reshape_arrays(y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Reshape 3D tensors to 2D arrays for sklearn metrics
        
        Args:
            y_true: Ground truth values [batch, seq_len, features]
            y_pred: Predicted values [batch, seq_len, features]
            
        Returns:
            Tuple of reshaped numpy arrays
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
            
        # Reshape from [batch, seq_len, features] to [batch * seq_len, features]
        y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
        y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])
        
        return y_true_reshaped, y_pred_reshaped
    
    @staticmethod
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Mean Squared Error"""
        y_true, y_pred = TimeSeriesMetrics._reshape_arrays(y_true, y_pred)
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Root Mean Squared Error"""
        y_true, y_pred = TimeSeriesMetrics._reshape_arrays(y_true, y_pred)
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Mean Absolute Error"""
        y_true, y_pred = TimeSeriesMetrics._reshape_arrays(y_true, y_pred)
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Mean Absolute Percentage Error"""
        y_true, y_pred = TimeSeriesMetrics._reshape_arrays(y_true, y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        y_true_safe = y_true[mask]
        y_pred_safe = y_pred[mask]
        
        return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100
    
    @staticmethod
    def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        y_true, y_pred = TimeSeriesMetrics._reshape_arrays(y_true, y_pred)
        
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    @staticmethod
    def evaluate_all(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Calculate all metrics"""
        return {
            'mse': TimeSeriesMetrics.mse(y_true, y_pred),
            'rmse': TimeSeriesMetrics.rmse(y_true, y_pred),
            'mae': TimeSeriesMetrics.mae(y_true, y_pred),
            'mape': TimeSeriesMetrics.mape(y_true, y_pred),
            'smape': TimeSeriesMetrics.smape(y_true, y_pred)
        }
        
        
class ModelEvaluator:
    """Handles model evaluation and prediction"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.metrics = TimeSeriesMetrics()
        
    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate predictions
        
        Args:
            inputs: Input tensor
            batch_size: Batch size for prediction
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        predictions = []
        
        # Create dataloader for batched prediction
        dataset = torch.utils.data.TensorDataset(inputs)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        for batch in dataloader:
            batch_input = batch[0].to(self.device)
            batch_pred, _ = self.model(batch_input)
            predictions.append(batch_pred.cpu())
            
        return torch.cat(predictions, dim=0)
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """
        Evaluate model on test data with memory optimization
        """
        self.model.eval()
        predictions = []
        targets = []
        
        # Process in smaller batches
        batch_size = 16
        
        for encoder_inputs, _, target in dataloader:
            # Process data in chunks
            for i in range(0, len(encoder_inputs), batch_size):
                chunk_inputs = encoder_inputs[i:i+batch_size].to(self.device)
                chunk_targets = target[i:i+batch_size]
                
                # Generate predictions
                with torch.cuda.amp.autocast():  # Use mixed precision
                    with torch.no_grad():
                        pred, _ = self.model(chunk_inputs)
                
                # Store predictions and targets
                predictions.append(pred.cpu())
                targets.append(chunk_targets.cpu())
                
                # Clean up memory
                del chunk_inputs, pred
                torch.cuda.empty_cache()
            
        # Concatenate all predictions and targets
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets, dim=0)
        
        # Calculate metrics
        metrics = self.metrics.evaluate_all(all_targets, all_predictions)
        
        return metrics, all_predictions, all_targets