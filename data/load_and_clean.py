"""
Load and clean power system data from Excel files.

This module provides robust Excel data loading with timestamp parsing,
data validation, and cleaning for power system analysis.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """
    Load and clean power system data from Excel file(s).

    This function handles:
    - Single Excel file or directory of files
    - Corrupted timestamp parsing (mixed date-time formats)
    - Auto-detection of Date/Time columns (case-insensitive)
    - Unit conversion (kVA -> MVA)
    - Data validation and cleaning
    - Missing data interpolation
    - Frequency detection and resampling

    Args:
        path: Path to Excel file or directory containing .xlsx files

    Returns:
        pd.DataFrame: Cleaned dataframe with datetime index and power columns

    Raises:
        FileNotFoundError: If path doesn't exist or no Excel files found
        ValueError: If no valid data could be loaded
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Collect Excel files to process
    if path.is_dir():
        excel_files = sorted(list(path.glob("*.xlsx")))
        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in directory: {path}")
    else:
        excel_files = [path]

    print(f"Found {len(excel_files)} Excel file(s) to process")

    # Load and process each file
    dataframes = []
    for file_path in excel_files:
        try:
            df = pd.read_excel(file_path)

            if df.empty:
                print(f"Warning: {file_path.name} is empty, skipping")
                continue

            # Standardize column names (case-insensitive mapping)
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if 'date' in col_lower and 'time' not in col_lower:
                    column_mapping[col] = 'Date'
                elif 'time' in col_lower and 'date' not in col_lower:
                    column_mapping[col] = 'Time'
                elif col_lower in ['s_total', 'stotal', 's total']:
                    column_mapping[col] = 'S_TOTAL'
                elif col_lower in ['p_total', 'ptotal', 'p total']:
                    column_mapping[col] = 'P_TOTAL'
                elif col_lower in ['q_total', 'qtotal', 'q total']:
                    column_mapping[col] = 'Q_TOTAL'

            if column_mapping:
                df = df.rename(columns=column_mapping)

            # Parse timestamps
            df = _parse_timestamps(df)

            if df.empty:
                print(f"Warning: {file_path.name} has no valid timestamps, skipping")
                continue

            # Convert numeric columns
            numeric_cols = ['S_TOTAL', 'P_TOTAL', 'Q_TOTAL', 'PT1', 'PT2', 'PT3',
                          'QT1', 'QT2', 'QT3', 'COS_PHI']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows where all power values are NaN
            power_cols = [col for col in numeric_cols if col in df.columns]
            if power_cols:
                df = df.dropna(subset=power_cols, how='all')

            if df.empty:
                print(f"Warning: {file_path.name} has no valid data, skipping")
                continue

            # Set timestamp as index
            df = df.set_index('timestamp')
            df = df.sort_index()

            # Keep only numeric columns
            df = df.select_dtypes(include=[np.number])

            print(f" Loaded {file_path.name}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
            dataframes.append(df)

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue

    if not dataframes:
        raise ValueError("No files could be loaded successfully")

    # Concatenate all dataframes
    df = pd.concat(dataframes, axis=0)
    df = df.sort_index()

    # Remove duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep='first')]

    # Detect and validate frequency
    df = _detect_and_resample(df)

    # Convert kVA to MVA if needed
    if 'S_TOTAL' in df.columns:
        mean_val = df['S_TOTAL'].mean()
        if mean_val > 1000:  # Likely in kVA
            print("Converting S_TOTAL from kVA to MVA")
            df['S_TOTAL'] = df['S_TOTAL'] / 1000

            # Convert other power columns if present
            for col in ['P_TOTAL', 'Q_TOTAL', 'PT1', 'PT2', 'PT3', 'QT1', 'QT2', 'QT3']:
                if col in df.columns:
                    df[col] = df[col] / 1000

    # Interpolate missing values (limit to small gaps)
    df = df.interpolate(method='linear', limit=6, limit_direction='both')

    # Final validation
    if 'S_TOTAL' not in df.columns:
        raise ValueError("S_TOTAL column not found in data")

    print(f"\n Final dataset: {len(df)} records, {len(df.columns)} columns")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Frequency: {df.index.freq}")

    return df


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps from Date/Time columns with multiple fallback strategies.

    Handles:
    - Separate Date and Time columns
    - Combined datetime columns
    - Mixed/corrupted formats (extracts time with regex)
    """
    # Strategy 1: Separate Date and Time columns
    if 'Date' in df.columns and 'Time' in df.columns:
        try:
            # Handle corrupted time strings (extract HH:MM:SS)
            def extract_time(val):
                if pd.isna(val):
                    return None
                val_str = str(val)
                # Extract time pattern HH:MM:SS
                match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', val_str)
                if match:
                    return f"{match.group(1).zfill(2)}:{match.group(2)}:{match.group(3)}"
                return val_str

            df['Time'] = df['Time'].apply(extract_time)
            df['timestamp'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )

            # Remove rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            return df

        except Exception as e:
            print(f"Warning: Error parsing Date/Time columns: {e}")

    # Strategy 2: Look for any datetime-like column
    for col in df.columns:
        col_lower = str(col).lower()
        if any(word in col_lower for word in ['date', 'time', 'timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                if not df.empty:
                    return df
            except:
                continue

    # Strategy 3: Try first two columns
    if len(df.columns) >= 2:
        try:
            df['timestamp'] = pd.to_datetime(
                df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
                errors='coerce'
            )
            df = df.dropna(subset=['timestamp'])
            if not df.empty:
                return df
        except:
            pass

    # Strategy 4: Try first column alone
    try:
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if not df.empty:
            return df
    except:
        pass

    raise ValueError("Could not parse timestamps from any column")


def _detect_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect data frequency and resample if needed.

    Expected frequency: 10 minutes (6 samples per hour)
    """
    # Calculate time differences
    time_diffs = df.index.to_series().diff()
    median_diff = time_diffs.median()

    # Determine frequency
    if pd.isna(median_diff):
        print("Warning: Could not determine frequency, assuming 10min")
        freq = '10min'
    else:
        minutes = median_diff.total_seconds() / 60
        if 8 <= minutes <= 12:  # Around 10 minutes
            freq = '10min'
        elif 13 <= minutes <= 17:  # Around 15 minutes
            freq = '15min'
        elif 28 <= minutes <= 32:  # Around 30 minutes
            freq = '30min'
        elif 58 <= minutes <= 62:  # Around 1 hour
            freq = '1h'
        else:
            print(f"Warning: Unusual frequency detected ({minutes:.1f} min), using 10min")
            freq = '10min'

    print(f"Detected frequency: {freq}")

    # Resample to ensure consistent frequency
    try:
        df = df.asfreq(freq)
    except:
        # If asfreq fails, try reindex with date_range
        start = df.index.min()
        end = df.index.max()
        new_index = pd.date_range(start=start, end=end, freq=freq)
        df = df.reindex(new_index)

    return df


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Testing load_and_clean with: {path}\n")

        try:
            df = load_and_clean(path)
            print(f"\n{df.shape}")
            print(f"\n{df.index.freq}")
            print(f"\nColumns: {df.columns.tolist()}")
            print(f"\nFirst few rows:")
            print(df.head())

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python load_and_clean.py <path_to_excel_or_directory>")
        sys.exit(1)
