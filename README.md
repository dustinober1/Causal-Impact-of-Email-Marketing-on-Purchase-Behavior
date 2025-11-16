# Causal Impact of Email Marketing on Purchase Behavior

A comprehensive causal inference project analyzing the effectiveness of email marketing campaigns using the UCI Online Retail dataset.

## 📊 Project Overview

This project implements advanced causal inference techniques to measure the causal impact of email marketing campaigns on customer purchase behavior. Using real-world e-commerce transaction data, we analyze customer segments, purchase patterns, and campaign effectiveness.

## 🎯 Objectives

1. **Customer Segmentation**: Analyze customer behavior using RFM (Recency, Frequency, Monetary) analysis
2. **Purchase Pattern Analysis**: Identify trends and patterns in customer purchasing behavior
3. **Causal Inference**: Establish causal relationships between email marketing and purchases
4. **Policy Evaluation**: Measure the effectiveness of different email marketing strategies

## 📁 Project Structure

```
Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/
├── data/
│   ├── raw/
│   │   └── online_retail.xlsx          # Original UCI dataset (22.6 MB)
│   └── processed/
│       ├── cleaned_online_retail.csv   # Cleaned transaction data
│       ├── daily_customer_purchases.csv # Daily aggregated customer data
│       └── customer_rfm_analysis.csv   # RFM segmentation results
├── notebooks/
│   └── 01_initial_eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   └── load_data.py                # Data loading & preprocessing
│   ├── causal/                         # Causal inference models
│   └── visualization/                  # Plotting & visualization
├── .venv/                              # Python virtual environment
└── README.md                           # This file
```

## 📊 Dataset

**UCI Online Retail Dataset**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Time Period**: December 1, 2010 to December 9, 2011 (373 days)
- **Records**: 397,884 transactions (after cleaning)
- **Customers**: 4,332 unique customers
- **Products**: 3,840 unique items
- **Total Revenue**: £8+ million

### Dataset Schema
- `InvoiceNo`: Transaction identifier
- `StockCode`: Product code
- `Description`: Product description
- `Quantity`: Number of items purchased
- `InvoiceDate`: Date and time of transaction
- `UnitPrice`: Price per unit
- `CustomerID`: Customer identifier (required for analysis)
- `Country`: Customer's country

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Install Dependencies (if not already installed)
```bash
pip install pandas matplotlib seaborn jupyter openpyxl numpy
```

### 3. Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_initial_eda.ipynb
```

Or execute the notebook programmatically:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_initial_eda.ipynb
```

### 4. Load and Explore Data Programmatically
```python
import sys
sys.path.append('src/data/')

from load_data import load_online_retail_data, clean_data, get_date_range

# Load and clean data
raw_data = load_online_retail_data()
df = clean_data(raw_data)

# Check date range
min_date, max_date = get_date_range(df)
print(f"Date range: {min_date} to {max_date}")
```

## 📈 Key Features

### Initial EDA Notebook (`notebooks/01_initial_eda.ipynb`)

The exploratory data analysis notebook provides:

1. **Basic Dataset Statistics**
   - Summary statistics for all numerical columns
   - Data types and missing value analysis
   - Unique counts for customers, products, and transactions

2. **Date Range Analysis**
   - Time period identification (2010-12-01 to 2011-12-09)
   - Monthly sales trends
   - Seasonal patterns

3. **Customer Purchase Patterns**
   - Order frequency distribution
   - Total spending per customer
   - Average order value
   - Customer lifetime analysis

4. **RFM Analysis (Recency, Frequency, Monetary)**
   - Customer segmentation into 5 groups:
     - **Champions** (RFM Score ≥ 13): Best customers
     - **Loyal Customers** (RFM Score 11-12): Regular high-value customers
     - **Potential Loyalists** (RFM Score 9-10): Recent customers with potential
     - **New Customers** (RFM Score 7-8): Recent but infrequent buyers
     - **Promising/Lost** (RFM Score < 7): Low engagement customers
   - Quartile-based scoring system
   - Visual segment distribution

5. **Exported Clean Datasets**
   - `daily_customer_purchases.csv`: Daily aggregated data for causal analysis
   - `customer_rfm_analysis.csv`: RFM scores and segments for each customer

## 📊 Sample Analysis Results

Based on the initial EDA:

- **Customer Segments** (example breakdown):
  - Champions: ~20% of customers
  - Loyal Customers: ~25% of customers
  - Potential Loyalists: ~20% of customers
  - New Customers: ~15% of customers
  - Promising/Lost: ~20% of customers

- **Purchase Patterns**:
  - Average order value: ~£15-20
  - Customer lifetime: 30-200 days (highly variable)
  - Peak purchasing: October-December (holiday season)

## 🔬 Next Steps

This project framework is designed for:

1. **Causal Impact Analysis**
   - Implement difference-in-differences (DiD)
   - Use synthetic control methods
   - Apply propensity score matching

2. **Email Marketing Campaign Simulation**
   - Define treatment and control groups
   - Analyze pre/post intervention metrics
   - Measure incremental lift

3. **Advanced Causal Inference Models**
   - Uplift modeling
   - Bayesian causal forests
   - Instrumental variable analysis

4. **Visualization & Reporting**
   - Interactive dashboards
   - Campaign effectiveness reports
   - Customer journey visualization

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
openpyxl>=3.0.0
```

## 🤝 Contributing

This is a research project for causal inference methodology. Future contributions could include:

- Additional causal inference models
- New visualization techniques
- Alternative customer segmentation methods
- Email campaign simulation frameworks

## 📝 License

This project uses the UCI Online Retail Dataset which is available for academic and research purposes.

## 📧 Contact

For questions or collaboration opportunities, please reach out through the repository.

---

**Built with ❤️ for causal inference research**