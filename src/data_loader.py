"""
Hass Avocado Retail Price & Volume Data Ingestion Module.
Ingests the official Hass Avocado Board (HAB) retail dataset, validates SHA-256 checksums,
and extracts log-log elasticity and Fourier annual seasonality features.
"""

import os
import hashlib
import pandas as pd
import numpy as np


class AvocadoDataLoader:
    """
    Data ingestion and feature preparation engine for econometric demand modeling.
    """

    EXPECTED_SHA256 = "631e77010fc1d22a57b7c0600b2f19427c7ba002f077cb7c3bad532bd72f0739"

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, "avocado.csv")

    def validate_checksum(self) -> bool:
        """Validates SHA-256 integrity of the raw avocado dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        with open(self.data_path, "rb") as f:
            computed = hashlib.sha256(f.read()).hexdigest()
        return computed == self.EXPECTED_SHA256

    def load_data(self, product_type='conventional', region='California'):
        """
        Loads and prepares avocado transaction history.
        Returns:
            df (pd.DataFrame): Cleaned weekly time series.
            X (np.ndarray): Design matrix [log_price, seasonality_sin, seasonality_cos].
            y (np.ndarray): Target log_volume.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        if product_type is not None:
            df = df[df['type'] == product_type]
        if region is not None:
            df = df[df['region'] == region]

        df = df.sort_values('Date').reset_index(drop=True)
        df['Month'] = df['Date'].dt.month
        df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12.0)
        df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12.0)

        df['Log_Price'] = np.log(df['AveragePrice'])
        df['Log_Volume'] = np.log(df['Total_Volume'])

        X = df[['Log_Price', 'Sin_Month', 'Cos_Month']].values
        y = df['Log_Volume'].values

        return df, X, y
