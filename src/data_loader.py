"""
Data loading and preprocessing for vessel power data.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler


# Features used by the model
MODEL_FEATURES = [
    'Bus1_Load', 'Bus1_Avail_Load', 'Bus2_Load', 'Bus2_Avail_Load',
    'BowThr1_Power', 'BowThr2_Power', 'BowThr3_Power',
    'SternThr1_Power', 'SternThr2_Power',
    'Main_Prop_PS_Drive_Power', 'Main_Prop_SB_Drive_Power',
    'Main_Prop_PS_ME1_Power', 'Main_Prop_PS_ME2_Power',
    'Speed', 'Latitude', 'Longitude'
]

# Variable groups for UI
VARIABLE_GROUPS = {
    'electrical': ['Bus1_Load', 'Bus1_Avail_Load', 'Bus2_Load', 'Bus2_Avail_Load'],
    'maneuver': ['BowThr1_Power', 'BowThr2_Power', 'BowThr3_Power', 'SternThr1_Power', 'SternThr2_Power'],
    'propulsion': ['Main_Prop_PS_Drive_Power', 'Main_Prop_SB_Drive_Power', 'Main_Prop_PS_ME1_Power', 'Main_Prop_PS_ME2_Power'],
    'ship': ['Draft_Aft', 'Draft_Fwd', 'Speed'],
    'coordinates': ['Latitude', 'Longitude']
}


@dataclass
class VesselData:
    """Container for vessel data."""
    timestamp: pd.DatetimeIndex
    features: np.ndarray  # (n_samples, n_features)
    feature_names: List[str]
    raw_df: pd.DataFrame


class VesselDataLoader:
    """Loads and preprocesses vessel power data."""

    def __init__(self, data_path: str, scaler_path: Optional[str] = None):
        self.data_path = Path(data_path)
        self.scaler_path = scaler_path
        self.scaler: Optional[StandardScaler] = None
        self._df: Optional[pd.DataFrame] = None
        self._vessel_data: Optional[VesselData] = None

        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)

    def load_data(self, force_reload: bool = False) -> VesselData:
        """Load and preprocess the vessel data."""
        if self._vessel_data is not None and not force_reload:
            return self._vessel_data

        # Load raw data
        df = pd.read_csv(
            self.data_path,
            sep=';',
            parse_dates=['LOGTIME'],
            low_memory=False
        )

        # Preprocess
        df = self.preprocess(df)
        self._df = df

        # Extract features
        features = df[MODEL_FEATURES].values

        self._vessel_data = VesselData(
            timestamp=df.index,
            features=features,
            feature_names=MODEL_FEATURES,
            raw_df=df
        )

        return self._vessel_data

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        # Set timestamp as index
        df = df.set_index('LOGTIME')
        df = df.sort_index()

        # Filter invalid rows (Latitude < 0.01 indicates invalid GPS)
        df = df[df['Latitude'] >= 0.01]

        # Remove duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]

        # Clip negative power values to 0
        power_cols = [c for c in df.columns if 'Power' in c or 'Load' in c]
        for col in power_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Forward fill missing values (up to 5 samples)
        df = df.ffill(limit=5)

        # Drop rows with remaining NaNs in model features
        df = df.dropna(subset=[f for f in MODEL_FEATURES if f in df.columns])

        # Compute derived features
        df['Total_Power'] = (
            df['Bus1_Load'] + df['Bus2_Load']
        )
        df['Load_Ratio'] = df['Bus1_Load'] / (df['Bus1_Load'] + df['Bus2_Load'] + 1e-6)

        return df

    def fit_scaler(self, features: np.ndarray) -> None:
        """Fit the scaler on training data."""
        self.scaler = StandardScaler()
        self.scaler.fit(features)

    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler."""
        if self.scaler is not None:
            joblib.dump(self.scaler, path)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using the fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(features)

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """Denormalize features back to original scale."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")
        return self.scaler.inverse_transform(features)

    def get_sequences(
        self,
        features: np.ndarray,
        window_size: int = 120,
        stride: int = 12
    ) -> np.ndarray:
        """Create sliding window sequences from features.

        Args:
            features: (n_samples, n_features) array
            window_size: Number of timesteps per window
            stride: Step size between windows

        Returns:
            (n_windows, window_size, n_features) array
        """
        n_samples = len(features)
        n_windows = (n_samples - window_size) // stride + 1

        sequences = np.zeros((n_windows, window_size, features.shape[1]))
        for i in range(n_windows):
            start = i * stride
            sequences[i] = features[start:start + window_size]

        return sequences

    def get_time_range(
        self,
        start: datetime,
        end: datetime
    ) -> VesselData:
        """Get data for a specific time range."""
        if self._df is None:
            self.load_data()

        mask = (self._df.index >= start) & (self._df.index <= end)
        df_slice = self._df[mask]

        return VesselData(
            timestamp=df_slice.index,
            features=df_slice[MODEL_FEATURES].values,
            feature_names=MODEL_FEATURES,
            raw_df=df_slice
        )

    def get_latest(self, n_samples: int = 120) -> VesselData:
        """Get the latest n samples."""
        if self._df is None:
            self.load_data()

        df_slice = self._df.iloc[-n_samples:]

        return VesselData(
            timestamp=df_slice.index,
            features=df_slice[MODEL_FEATURES].values,
            feature_names=MODEL_FEATURES,
            raw_df=df_slice
        )

    def get_data_at_index(self, index: int, n_samples: int = 120) -> VesselData:
        """Get n samples ending at a specific index."""
        if self._df is None:
            self.load_data()

        # Ensure bounds
        end_idx = min(index + 1, len(self._df))
        start_idx = max(0, end_idx - n_samples)

        df_slice = self._df.iloc[start_idx:end_idx]

        return VesselData(
            timestamp=df_slice.index,
            features=df_slice[MODEL_FEATURES].values,
            feature_names=MODEL_FEATURES,
            raw_df=df_slice
        )

    def get_test_data_range(self) -> Tuple[int, int]:
        """Get the index range for test data (last 15%)."""
        if self._df is None:
            self.load_data()

        n_samples = len(self._df)
        test_start = int(n_samples * 0.85)  # Test data starts at 85%
        test_end = n_samples - 1
        return test_start, test_end

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[VesselData, VesselData, VesselData]:
        """Split data temporally into train/val/test sets."""
        if self._df is None:
            self.load_data()

        n_samples = len(self._df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_df = self._df.iloc[:train_end]
        val_df = self._df.iloc[train_end:val_end]
        test_df = self._df.iloc[val_end:]

        train_data = VesselData(
            timestamp=train_df.index,
            features=train_df[MODEL_FEATURES].values,
            feature_names=MODEL_FEATURES,
            raw_df=train_df
        )

        val_data = VesselData(
            timestamp=val_df.index,
            features=val_df[MODEL_FEATURES].values,
            feature_names=MODEL_FEATURES,
            raw_df=val_df
        )

        test_data = VesselData(
            timestamp=test_df.index,
            features=test_df[MODEL_FEATURES].values,
            feature_names=MODEL_FEATURES,
            raw_df=test_df
        )

        return train_data, val_data, test_data

    def get_variable_group(self, group: str) -> List[str]:
        """Get variables for a specific group."""
        return VARIABLE_GROUPS.get(group, [])

    def get_group_data(self, group: str) -> pd.DataFrame:
        """Get data for a specific variable group."""
        if self._df is None:
            self.load_data()

        cols = self.get_variable_group(group)
        available_cols = [c for c in cols if c in self._df.columns]
        return self._df[available_cols]
