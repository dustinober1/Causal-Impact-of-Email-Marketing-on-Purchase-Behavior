"""
Feature Engineering: Customer-Week Panel Dataset
Transforms transaction-level data into customer-week observations for causal analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from load_data import load_online_retail_data, clean_data, get_date_range
from pathlib import Path


def calculate_rfm_scores(df, reference_date):
    """
    Calculate RFM (Recency, Frequency, Monetary) scores for each customer.

    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction-level data
    reference_date : datetime
        Reference date for recency calculation

    Returns:
    --------
    pandas.DataFrame
        RFM scores by customer
    """
    # Handle edge case: empty dataframe
    if len(df) == 0:
        return pd.DataFrame(columns=['Recency', 'Frequency', 'Monetary', 'RFM_Score'])

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).round(2)

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Handle edge case: single customer or insufficient variation
    if len(rfm) < 5:
        rfm['RFM_Score'] = 7  # Neutral score
        rfm['R_Quartile'] = 3
        rfm['F_Quartile'] = 3
        rfm['M_Quartile'] = 3
        return rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score']]

    # Create quartiles with duplicate handling
    try:
        rfm['R_Quartile'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['F_Quartile'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_Quartile'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    except ValueError:
        # Fallback: assign neutral scores if quartiles can't be created
        rfm['R_Quartile'] = 3
        rfm['F_Quartile'] = 3
        rfm['M_Quartile'] = 3

    # Calculate composite RFM score
    rfm['RFM_Score'] = (
        rfm['R_Quartile'].astype(int) +
        rfm['F_Quartile'].astype(int) +
        rfm['M_Quartile'].astype(int)
    )

    return rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score']]


def create_customer_week_panel(df):
    """
    Create customer-week panel dataset with engineered features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction-level data

    Returns:
    --------
    pandas.DataFrame
        Customer-week panel with features
    """
    print("Creating customer-week panel dataset...")
    print("=" * 60)

    # Get date range
    min_date, max_date = get_date_range(df)
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

    print(f"Date range: {min_date.date()} to {max_date.date()}")

    # Convert to week format
    df['week_start'] = df['InvoiceDate'].dt.to_period('W').dt.start_time

    # Calculate week numbers from start
    df['week_number'] = ((df['week_start'] - min_date).dt.days / 7).astype(int) + 1

    # Get all weeks in the dataset
    total_weeks = df['week_number'].max()
    print(f"Total weeks in dataset: {total_weeks}")

    # Filter customers with at least 3 purchases
    customer_purchase_counts = df.groupby('CustomerID').size()
    eligible_customers = customer_purchase_counts[customer_purchase_counts >= 3].index

    print(f"Total customers: {df['CustomerID'].nunique():,}")
    print(f"Customers with ≥3 purchases: {len(eligible_customers):,}")

    df_filtered = df[df['CustomerID'].isin(eligible_customers)].copy()
    df_filtered['CustomerID'] = df_filtered['CustomerID'].astype(int)

    # Calculate customer-level metadata (first purchase, etc.)
    customer_metadata = df_filtered.groupby('CustomerID').agg({
        'InvoiceDate': ['min', 'max'],
        'TotalAmount': ['sum', 'mean', 'count'],
        'InvoiceNo': 'nunique',
        'week_number': 'min'
    }).round(2)

    customer_metadata.columns = [
        'first_purchase_date', 'last_purchase_date',
        'total_spent', 'avg_order_value', 'total_transactions',
        'unique_orders', 'first_week'
    ]

    # Calculate overall RFM scores (for reference)
    print("\nCalculating RFM scores...")
    rfm_scores = calculate_rfm_scores(df_filtered, max_date + pd.Timedelta(days=1))

    # Aggregate by customer-week
    print("\nAggregating to customer-week level...")
    customer_week_agg = df_filtered.groupby(['CustomerID', 'week_start']).agg({
        'TotalAmount': 'sum',
        'Quantity': 'sum',
        'InvoiceNo': 'nunique',
        'InvoiceDate': 'count'
    }).round(2)

    customer_week_agg.columns = [
        'revenue_this_week', 'quantity_this_week', 'orders_this_week', 'transactions_this_week'
    ]
    customer_week_agg = customer_week_agg.reset_index()

    # Add week number
    customer_week_agg['week_number'] = ((customer_week_agg['week_start'] - min_date).dt.days / 7).astype(int) + 1

    print(f"Customer-week observations: {len(customer_week_agg):,}")

    # Create complete panel (all customers, all weeks)
    print("\nCreating complete panel dataset...")
    all_customers = customer_week_agg['CustomerID'].unique()
    all_week_nums = range(1, total_weeks + 1)

    # Create full combination of customers and weeks
    from itertools import product
    panel_index = pd.DataFrame(
        list(product(all_customers, all_week_nums)),
        columns=['CustomerID', 'week_number']
    )

    print(f"Complete panel size: {len(panel_index):,} observations")

    # Merge with customer-week data
    panel = panel_index.merge(
        customer_week_agg,
        on=['CustomerID', 'week_number'],
        how='left'
    )

    # Fill in missing values for weeks without purchases
    panel['purchase_this_week'] = (~panel['revenue_this_week'].isna()).astype(int)
    panel['revenue_this_week'] = panel['revenue_this_week'].fillna(0)
    panel['quantity_this_week'] = panel['quantity_this_week'].fillna(0)
    panel['orders_this_week'] = panel['orders_this_week'].fillna(0)
    panel['transactions_this_week'] = panel['transactions_this_week'].fillna(0)

    # Merge customer metadata
    panel = panel.merge(
        customer_metadata.reset_index()[['CustomerID', 'first_week']],
        on='CustomerID',
        how='left'
    )

    # Only include weeks from customer's first purchase onwards
    panel = panel[panel['week_number'] >= panel['first_week']].copy()

    print(f"Panel after filtering to customer start dates: {len(panel):,}")

    # Engineer time-dependent features
    print("\nEngineering time-dependent features...")
    print("Using optimized vectorized operations...")

    # Sort by customer and week for time-aware calculations
    panel = panel.sort_values(['CustomerID', 'week_number']).reset_index(drop=True)

    # Customer tenure in weeks (vectorized)
    panel['customer_tenure_weeks'] = panel['week_number'] - panel['first_week']

    # Total past purchases (cumulative sum per customer)
    panel['total_past_purchases'] = panel.groupby('CustomerID')['purchase_this_week'].cumsum() - panel['purchase_this_week']

    # Average order value (cumulative average per customer)
    panel['cumulative_revenue'] = panel.groupby('CustomerID')['revenue_this_week'].cumsum()
    panel['cumulative_purchases'] = panel.groupby('CustomerID')['purchase_this_week'].cumsum()

    # Shift by 1 to exclude current week
    panel['avg_order_value'] = (
        (panel['cumulative_revenue'].shift(1) / panel['cumulative_purchases'].shift(1)).fillna(0)
    )

    # Days since last purchase (efficient calculation)
    print("  Calculating days since last purchase...")
    panel = panel.sort_values(['CustomerID', 'week_number']).reset_index(drop=True)

    # Create an empty column
    panel['days_since_last_purchase'] = 999.0

    # Process in chunks to avoid memory issues
    chunk_size = 1000
    total_customers = panel['CustomerID'].nunique()
    num_chunks = (total_customers + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_customers)
        customer_chunk = panel['CustomerID'].unique()[start_idx:end_idx]

        for customer_id in customer_chunk:
            mask = panel['CustomerID'] == customer_id
            customer_data = panel[mask].copy()

            # Calculate days since last purchase for this customer
            last_purchase_week = None

            for idx in customer_data.index:
                week = panel.loc[idx, 'week_number']
                has_purchase = panel.loc[idx, 'purchase_this_week']

                if has_purchase == 1:
                    # Purchase this week
                    panel.loc[idx, 'days_since_last_purchase'] = 0
                    last_purchase_week = week
                else:
                    # No purchase this week
                    if last_purchase_week is None:
                        # No prior purchase
                        panel.loc[idx, 'days_since_last_purchase'] = 999
                    else:
                        # Days since last purchase
                        panel.loc[idx, 'days_since_last_purchase'] = (week - last_purchase_week) * 7

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{num_chunks} customer chunks...")

    # RFM scores (calculated once per customer from all historical data)
    print("  Calculating RFM scores...")
    ref_date = max_date + pd.Timedelta(days=1)
    rfm_all = calculate_rfm_scores(df_filtered, ref_date)
    rfm_all = rfm_all.reset_index()[['CustomerID', 'RFM_Score']]

    panel = panel.merge(rfm_all, on='CustomerID', how='left')

    # Initialize rfm_score column
    panel['rfm_score'] = 7  # Default neutral score

    # Fill with calculated scores where available
    panel.loc[panel['RFM_Score'].notna(), 'rfm_score'] = panel.loc[panel['RFM_Score'].notna(), 'RFM_Score']

    # Drop temporary columns
    panel = panel.drop(columns=['cumulative_revenue', 'cumulative_purchases', 'RFM_Score'])

    # Final feature selection and ordering
    final_features = [
        'CustomerID',
        'week_number',
        'week_start',
        'purchase_this_week',
        'revenue_this_week',
        'days_since_last_purchase',
        'total_past_purchases',
        'avg_order_value',
        'customer_tenure_weeks',
        'rfm_score',
        'quantity_this_week',
        'orders_this_week',
        'transactions_this_week'
    ]

    panel = panel[final_features]

    # Round numerical columns
    numeric_columns = [
        'revenue_this_week', 'days_since_last_purchase', 'avg_order_value',
        'customer_tenure_weeks', 'rfm_score'
    ]
    for col in numeric_columns:
        panel[col] = panel[col].round(2)

    print("\n" + "=" * 60)
    print("Panel Dataset Summary")
    print("=" * 60)
    print(f"Total observations: {len(panel):,}")
    print(f"Unique customers: {panel['CustomerID'].nunique():,}")
    print(f"Week range: {panel['week_number'].min()} to {panel['week_number'].max()}")
    print(f"Total weeks: {panel['week_number'].nunique()}")
    print(f"Weeks with purchases: {panel['purchase_this_week'].sum():,}")
    print(f"Purchase rate: {panel['purchase_this_week'].mean():.3f}")

    return panel


def analyze_panel(panel):
    """Analyze the panel dataset and print summary statistics."""
    print("\n" + "=" * 60)
    print("Feature Statistics")
    print("=" * 60)

    print("\nNumerical Features:")
    numeric_cols = [
        'revenue_this_week', 'days_since_last_purchase', 'total_past_purchases',
        'avg_order_value', 'customer_tenure_weeks', 'rfm_score'
    ]
    print(panel[numeric_cols].describe())

    print("\nPurchase Behavior:")
    print(f"Customers with purchases in week 1: {panel[panel['week_number'] == 1]['purchase_this_week'].sum()}")
    print(f"Customers with purchases in last week: {panel[panel['week_number'] == panel['week_number'].max()]['purchase_this_week'].sum()}")

    print("\nTop 10 customers by total purchases:")
    top_customers = panel.groupby('CustomerID')['purchase_this_week'].sum().sort_values(ascending=False).head(10)
    print(top_customers)

    print("\nFeature correlations with purchase_this_week:")
    correlation_cols = [
        'purchase_this_week', 'days_since_last_purchase', 'total_past_purchases',
        'avg_order_value', 'customer_tenure_weeks', 'rfm_score'
    ]
    print(panel[correlation_cols].corr()['purchase_this_week'].sort_values(ascending=False))


if __name__ == "__main__":
    # Load and clean data
    print("Loading transaction data...")
    raw_data = load_online_retail_data()
    df = clean_data(raw_data)

    # Create customer-week panel
    panel_df = create_customer_week_panel(df)

    # Analyze panel
    analyze_panel(panel_df)

    # Save panel dataset
    output_path = Path(__file__).parent.parent / "data" / "processed" / "customer_week_panel.csv"
    panel_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Panel dataset saved to: {output_path}")
    print("=" * 60)
    print("\nPanel dataset is ready for causal inference modeling!")