"""
Data loading module for the Email Marketing Causal Inference project.
Handles loading and basic preprocessing of the UCI Online Retail dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def load_online_retail_data(file_path=None):
    """
    Load the UCI Online Retail dataset from Excel file.

    Parameters:
    -----------
    file_path : str, optional
        Path to the Excel file. If None, uses default path.

    Returns:
    --------
    pandas.DataFrame
        Raw dataset with columns:
        - InvoiceNo: Invoice number
        - StockCode: Product code
        - Description: Product description
        - Quantity: Number of items
        - InvoiceDate: Date and time
        - UnitPrice: Price per unit
        - CustomerID: Customer identifier
        - Country: Customer's country
    """
    if file_path is None:
        # Default path to downloaded dataset
        default_path = Path(__file__).parent.parent.parent / "data" / "raw" / "online_retail.xlsx"
        file_path = str(default_path)

    print(f"Loading data from: {file_path}")

    # Load the Excel file
    df = pd.read_excel(file_path)

    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def clean_data(df):
    """
    Perform basic cleaning on the online retail dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw online retail dataset

    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()

    # Remove rows where CustomerID is missing (required for our analysis)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['CustomerID'])
    print(f"Removed {initial_rows - len(df_clean)} rows with missing CustomerID")

    # Remove negative quantities (returns)
    df_clean = df_clean[df_clean['Quantity'] > 0]

    # Remove negative prices
    df_clean = df_clean[df_clean['UnitPrice'] > 0]

    # Calculate total amount per transaction
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # Create date-only column for analysis
    df_clean['Date'] = df_clean['InvoiceDate'].dt.date

    print(f"Cleaned dataset shape: {df_clean.shape}")

    return df_clean


def get_date_range(df):
    """
    Get the date range of the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset

    Returns:
    --------
    tuple
        (min_date, max_date) as datetime.date objects
    """
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    return min_date, max_date


if __name__ == "__main__":
    # Load and clean data
    raw_data = load_online_retail_data()
    clean_data_df = clean_data(raw_data)

    # Show date range
    min_date, max_date = get_date_range(clean_data_df)
    print(f"\nDate range: {min_date} to {max_date}")
    print(f"Total days: {(max_date - min_date).days}")

    # Save cleaned data
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cleaned_online_retail.csv"
    clean_data_df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")