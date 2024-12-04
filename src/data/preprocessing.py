"""
Data preprocessing module for time series forecasting.

Dependencies:
- numpy>=1.24.3
- pandas>=2.0.3
- scikit-learn>=1.3.0
- torch>=2.4.1
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
import torch

__version__ = '1.0.0'

class TimeFeatureGenerator:
    """Generates cyclic time features from timestamps"""

    @staticmethod
    def _create_cyclical_features(value: float, period: int) -> Tuple[float, float]:
        """Convert a numeric value to cyclic sin/cos features"""
        sin_value = np.sin(2 * np.pi * value / period)
        cos_value = np.cos(2 * np.pi * value / period)
        return sin_value, cos_value

    def generate_time_features(self, timestamps: pd.Series) -> np.ndarray:
        """
        Generate time-based features from timestamps

        Args:
            timestamps: Pandas Series of timestamps

        Returns:
            numpy.ndarray: Array of time features with shape [n_samples, n_features]

        Note:
            Requires pandas>=2.0.3 for proper timestamp handling
        """
        # Convert to pandas datetime if not already
        timestamps = pd.to_datetime(timestamps)

        features = []
        for ts in timestamps:
            # Hour of day features
            hour_sin, hour_cos = self._create_cyclical_features(ts.hour, 24)

            # Day of week features
            dow_sin, dow_cos = self._create_cyclical_features(ts.dayofweek, 7)

            # Month features
            month_sin, month_cos = self._create_cyclical_features(ts.month, 12)

            # Is weekend feature (binary)
            is_weekend = 1.0 if ts.dayofweek >= 5 else 0.0

            features.append([
                hour_sin, hour_cos,
                dow_sin, dow_cos,
                month_sin, month_cos,
                is_weekend
            ])

        return np.array(features)

class DataPreprocessor:
    """
    Handles data preprocessing for the Informer model

    Dependencies:
        - scikit-learn>=1.3.0 for StandardScaler
        - torch>=2.4.1 for tensor operations
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.time_feature_generator = TimeFeatureGenerator()
        self._is_fitted = False

    def preprocess(self,
                   df: pd.DataFrame,
                   timestamp_col: str,
                   target_col: str,
                   is_training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess the data for the Informer model

        Args:
            df: Input dataframe with timestamp and target columns
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            is_training: Whether this is training data (for fitting scaler)

        Returns:
            Tuple of (preprocessed_tensor, preprocessing_info)

        Raises:
            ValueError: If scaler is not fitted for validation/test data
        """
        # Generate time features
        time_features = self.time_feature_generator.generate_time_features(df[timestamp_col])

        # Scale target variable
        target_values = df[target_col].values.reshape(-1, 1)
        if is_training:
            scaled_target = self.scaler.fit_transform(target_values)
            self._is_fitted = True
        else:
            if not self._is_fitted:
                raise ValueError("Scaler must be fitted before transforming validation/test data")
            scaled_target = self.scaler.transform(target_values)

        # Combine features
        combined_features = np.hstack([scaled_target, time_features])

        # Convert to tensor
        feature_tensor = torch.FloatTensor(combined_features)

        # Return tensor and preprocessing info
        preprocessing_info = {
            'feature_dims': feature_tensor.shape[1],
            'scaler_mean': float(self.scaler.mean_[0]),
            'scaler_scale': float(self.scaler.scale_[0])
        }

        return feature_tensor, preprocessing_info

    def inverse_transform(self, scaled_values: torch.Tensor) -> torch.Tensor:
        """
        Convert scaled values back to original scale

        Args:
            scaled_values: Tensor of scaled values

        Returns:
            Tensor of values in original scale

        Raises:
            ValueError: If scaler is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")

        # Convert to numpy for inverse transform
        numpy_values = scaled_values.cpu().numpy()
        if len(numpy_values.shape) == 1:
            numpy_values = numpy_values.reshape(-1, 1)

        # Inverse transform
        original_scale = self.scaler.inverse_transform(numpy_values)

        # Back to tensor
        return torch.FloatTensor(original_scale)