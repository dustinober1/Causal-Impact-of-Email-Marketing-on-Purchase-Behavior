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
│   │   └── online_retail.xlsx                 # Original UCI dataset (22.6 MB)
│   └── processed/
│       ├── cleaned_online_retail.csv          # Cleaned transaction data
│       ├── daily_customer_purchases.csv       # Daily aggregated customer data
│       ├── customer_rfm_analysis.csv          # RFM segmentation results
│       └── customer_week_panel.csv            # Customer-week panel for causal analysis (6.4 MB)
├── notebooks/
│   └── 01_initial_eda.ipynb                   # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   ├── load_data.py                       # Data loading & preprocessing
│   │   └── create_panel_data.py               # Feature engineering & panel creation
│   ├── causal/                                # Causal inference models
│   └── visualization/                         # Plotting & visualization
├── .venv/                                     # Python virtual environment
└── README.md                                  # This file
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

### 4. Create Customer-Week Panel Dataset
```bash
# Generate the panel dataset with engineered features
.venv/bin/python src/data/create_panel_data.py
```

This script will:
- Convert transactions to customer-week format
- Engineer time-dependent features
- Create a panel dataset ready for causal inference
- Save as `data/processed/customer_week_panel.csv`

### 5. Load and Explore Data Programmatically
```python
import sys
sys.path.append('src/data/')

from load_data import load_online_retail_data, clean_data
from create_panel_data import create_customer_week_panel

# Option 1: Load transaction data
raw_data = load_online_retail_data()
df = clean_data(raw_data)

# Option 2: Create customer-week panel
panel_df = create_customer_week_panel(df)

# Option 3: Load existing panel
import pandas as pd
panel = pd.read_csv('data/processed/customer_week_panel.csv')
print(f"Panel shape: {panel.shape}")
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

### Customer-Week Panel Dataset (`data/processed/customer_week_panel.csv`)

This is the **main dataset for causal inference analysis**, transformed from transaction-level to customer-week observations.

#### Features Created:

**1. Core Identifiers**
   - `CustomerID`: Unique customer identifier
   - `week_number`: Week index (1-53, where week 1 = Dec 1-7, 2010)
   - `week_start`: Start date of the week

**2. Outcome Variables (Target for Causal Analysis)**
   - `purchase_this_week`: Binary indicator (1 = customer made purchase, 0 = no purchase)
   - `revenue_this_week`: Total revenue in that week (£)

**3. Engineered Features (Predictors)**
   - `days_since_last_purchase`: Days elapsed since customer's most recent purchase (0-999)
   - `total_past_purchases`: Cumulative number of purchases up to previous week (0-52)
   - `avg_order_value`: Running average order value based on past purchases (£0-1000+)
   - `customer_tenure_weeks`: Number of weeks since customer's first purchase (0-52)
   - `rfm_score`: Composite RFM score based on customer behavior (3-15, higher = better)

**4. Additional Metrics**
   - `quantity_this_week`: Total quantity of items purchased
   - `orders_this_week`: Number of orders placed
   - `transactions_this_week`: Number of individual transactions

#### Dataset Statistics:

- **Shape**: 137,888 observations × 13 features
- **Time Coverage**: 53 weeks (Dec 2010 - Dec 2011)
- **Customers**: 4,213 unique customers (all with ≥3 purchases)
- **Purchase Rate**: 11.4% (15,787 purchase weeks out of 137,888 total)
- **Average Revenue (when purchased)**: £556.95

#### Feature Correlations with Purchase:

| Feature | Correlation | Interpretation |
|---------|------------|----------------|
| `days_since_last_purchase` | -0.336 | **Strong negative**: More recent purchases → higher likelihood to buy |
| `rfm_score` | +0.223 | **Moderate positive**: Higher RFM score → more likely to buy |
| `total_past_purchases` | +0.175 | **Moderate positive**: More purchase history → higher engagement |
| `customer_tenure_weeks` | -0.070 | **Weak negative**: Longer tenure slightly → less likely (attrition) |
| `avg_order_value` | +0.039 | **Weak positive**: Higher AOV → more likely to buy |

#### Example Usage for Causal Inference:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load panel data
panel = pd.read_csv('data/processed/customer_week_panel.csv')

# Prepare features (exclude weeks where customer just started to avoid bias)
analysis_panel = panel[panel['customer_tenure_weeks'] >= 2].copy()

# Define features and target
features = [
    'days_since_last_purchase',
    'total_past_purchases',
    'avg_order_value',
    'customer_tenure_weeks',
    'rfm_score'
]

X = analysis_panel[features]
y = analysis_panel['purchase_this_week']

# Example: Logistic regression model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

print("Feature importance:", dict(zip(features, model.coef_[0])))
```

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

The panel dataset is now **ready for causal inference modeling**. Here are the next steps:

1. **Causal Impact Analysis** (Ready to implement!)
   - **Difference-in-Differences (DiD)**: Compare customers who received email campaigns vs. control groups over time
   - **Synthetic Control Method**: Create synthetic counterfactuals for customers who received campaigns
   - **Propensity Score Matching**: Match customers based on observable characteristics before analyzing treatment effects

2. **Email Marketing Campaign Simulation**
   - **Treatment Definition**: Define email marketing exposure (e.g., promotional emails, newsletter campaigns)
   - **Pre/Post Analysis**: Analyze purchase behavior before and after campaign exposure
   - **Incremental Lift Measurement**: Quantify the additional purchases attributable to email marketing

3. **Advanced Causal Inference Models**
   - **Uplift Modeling**: Predict the incremental effect of email campaigns on individual customers
   - **Bayesian Causal Forests**: Flexible non-parametric approach for heterogeneous treatment effects
   - **Instrumental Variable Analysis**: If instrumental variables can be identified (e.g., send time randomness)
   - **Regression Discontinuity**: If campaigns target customers based on thresholds (e.g., RFM score)

4. **Feature Engineering for Causal Analysis**
   - **Lagged Features**: Add more lag variables (e.g., 2-week, 4-week rolling averages)
   - **Seasonal Features**: Add week-of-year indicators to control for seasonality
   - **Customer Segmentation**: Include RFM segments as covariates
   - **Interaction Terms**: Test interactions between features (e.g., days_since_last_purchase × rfm_score)

5. **Visualization & Reporting**
   - **Treatment Effect Plots**: Visualize causal effects across customer segments
   - **Cohort Analysis**: Track customer behavior over time
   - **Campaign ROI Dashboards**: Show return on investment for different email strategies

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