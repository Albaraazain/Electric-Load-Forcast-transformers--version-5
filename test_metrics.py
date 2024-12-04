import torch
import numpy as np

from src.utils.metrics import TimeSeriesMetrics

def test_metrics():
    """Test TimeSeriesMetrics with various input scenarios"""
    
    # Create sample data that matches our model's output format
    batch_size = 16
    seq_length = 24
    n_features = 1
    
    # Create dummy predictions and targets with the same shape as our model
    y_true = torch.randn(batch_size, seq_length, n_features)
    y_pred = torch.randn(batch_size, seq_length, n_features)
    
    print("\nTest 1: Basic tensor shapes")
    print(f"Input shapes - True: {y_true.shape}, Pred: {y_pred.shape}")
    
    try:
        metrics = TimeSeriesMetrics.evaluate_all(y_true, y_pred)
        print("\nMetrics calculated successfully:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error in basic test: {str(e)}")
    
    # Test with known values
    print("\nTest 2: Known values")
    y_true_known = torch.tensor([[[1.0], [2.0], [3.0]], 
                               [[4.0], [5.0], [6.0]]])  # shape: [2, 3, 1]
    y_pred_known = torch.tensor([[[1.1], [2.2], [3.3]], 
                               [[3.9], [4.8], [5.7]]])  # shape: [2, 3, 1]
    
    try:
        metrics_known = TimeSeriesMetrics.evaluate_all(y_true_known, y_pred_known)
        print("\nMetrics with known values:")
        for metric_name, value in metrics_known.items():
            print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error in known values test: {str(e)}")
    
    # Test with zero values (for MAPE handling)
    print("\nTest 3: Handling zero values")
    y_true_zeros = torch.tensor([[[0.0], [2.0], [0.0]], 
                                [[4.0], [0.0], [6.0]]])
    y_pred_zeros = torch.tensor([[[0.1], [2.2], [0.3]], 
                                [[3.9], [0.1], [5.7]]])
    
    try:
        metrics_zeros = TimeSeriesMetrics.evaluate_all(y_true_zeros, y_pred_zeros)
        print("\nMetrics with zero values:")
        for metric_name, value in metrics_zeros.items():
            print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error in zero values test: {str(e)}")
    
    # Test with model-like output shapes
    print("\nTest 4: Model-like output")
    model_true = torch.randn(32, 24, 1)  # Batch:32, Seq:24, Features:1
    model_pred = torch.randn(32, 24, 1)
    
    try:
        metrics_model = TimeSeriesMetrics.evaluate_all(model_true, model_pred)
        print("\nMetrics with model-like shapes:")
        for metric_name, value in metrics_model.items():
            print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error in model-like test: {str(e)}")

if __name__ == "__main__":
    print("Starting TimeSeriesMetrics tests...")
    test_metrics()
    print("\nTests completed!")