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
│   │   └── online_retail.xlsx                       # Original UCI dataset (22.6 MB)
│   └── processed/
│       ├── cleaned_online_retail.csv                # Cleaned transaction data
│       ├── daily_customer_purchases.csv             # Daily aggregated customer data
│       ├── customer_rfm_analysis.csv                # RFM segmentation results
│       ├── customer_week_panel.csv                  # Customer-week panel for causal analysis (6.4 MB)
│       ├── simulated_email_campaigns.csv            # Simulated email campaigns with confounding (17 MB)
│       ├── ground_truth.json                        # True causal effect parameters
│       └── simulation_summary.json                  # Simulation statistics
├── notebooks/
│   ├── 01_initial_eda.ipynb                         # Exploratory Data Analysis
│   └── 02_email_campaign_simulation.ipynb           # Email campaign simulation with confounding
├── src/
│   ├── data/
│   │   ├── load_data.py                             # Data loading & preprocessing
│   │   ├── create_panel_data.py                     # Feature engineering & panel creation
│   │   └── simulate_email_campaigns.py              # Email campaign simulation
│   ├── causal/                                      # Causal inference models
│   └── visualization/                               # Plotting & visualization
├── .venv/                                           # Python virtual environment
└── README.md                                        # This file
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

### 5. Simulate Email Campaigns with Realistic Confounding
```bash
# Generate simulated email campaigns with true causal effect
.venv/bin/python src/data/simulate_email_campaigns.py
```

This script will:
- Create realistic email assignment based on customer characteristics (confounding!)
- Embed a TRUE causal effect (+10% base, varying by RFM score)
- Generate outcomes with both observed and counterfactual states
- Save ground truth for validation
- Output: `simulated_email_campaigns.csv` (17 MB, ready for causal inference!)

**Why simulate?** Real email campaigns are confounded. This simulation teaches you:
- How confounding creates bias (naive effect: 16.0% vs true: 9.5%)
- Why you need causal inference methods
- How to validate methods against known truth

### 6. Load and Explore Data Programmatically
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

### Email Campaign Simulation with Confounding (`src/data/simulate_email_campaigns.py`)

This simulation creates **realistic email marketing scenarios** with confounding and a true causal effect, perfect for learning and testing causal inference methods.

#### Why This Simulation Matters

In the real world, email campaigns are **NOT sent randomly**. Companies target customers based on:
- Recent purchase history
- Customer value (RFM score)
- Engagement levels
- Risk of churning

This creates **CONFOUNDING** - systematic differences between customers who receive emails and those who don't. Naive comparisons will be **severely biased**!

#### Simulation Design

**Confounding Rules (Email Assignment):**
Customers are more likely to receive emails if they are:
- **Recent purchasers** (bought in last 2 weeks): 60% chance
- **Frequent buyers** (>10 past purchases): 50% chance
- **High-value customers** (AOV > £20): 55% chance
- **Lapsed customers** (30-60 days since purchase): 40% chance
- **Base rate**: 15% for everyone

**True Causal Effect:**
- **Base effect**: +10 percentage points increase in purchase probability
- **Interaction**: Stronger effect for medium RFM scores (8-12): +5 pp
- **Heterogeneity**: Weaker for high RFM (>12): +2 pp, Negative for low RFM (<8): -3 pp
- **Random noise**: Realistic variation in effects

#### Key Results from Simulation

**Confounding Verification:**
- Email send rate: **81.7%** (highly selective!)
- Recent buyers: 95.1% receive emails
- Frequent buyers: 95.0% receive emails
- Other customers: only 18.2% receive emails

**Naive vs True Effect:**
- **Naive observed effect**: 16.0% (BIASED!)
- **True causal effect**: 9.5% (close to ground truth)
- **Bias**: 6.5 percentage points (68% overestimate!)

**Why the Bias?**
- Email recipients already have higher baseline purchase probability
- Comparing email vs no-email compares different customer types
- This is **selection bias**, not a valid test of email effectiveness

**Heterogeneous Effects (By RFM Score):**
- Low RFM (3-7): 5.4% effect
- **Medium RFM (8-12): 12.2% effect** ← Strongest!
- High RFM (13-15): 10.5% effect

**Business Insight:** Customers with medium engagement (familiar but not yet loyal) respond best to email marketing!

#### Files Created

1. **`simulated_email_campaigns.csv`** (17 MB, 137,888 observations)
   - Contains: email assignments, observed outcomes, counterfactuals, and true treatment effects
   - Columns: `received_email`, `email_assignment_probability`, `purchased_this_week_observed`, `individual_treatment_effect`, etc.

2. **`ground_truth.json`**
   - True parameters for validation
   - Base effect: 10%, interaction effects by RFM segment

3. **`simulation_summary.json`**
   - Quick statistics: confounding detected, email rates, effect sizes

4. **`notebooks/02_email_campaign_simulation.ipynb`**
   - Comprehensive explanation with visualizations
   - Demonstrates confounding, naive vs true effects
   - Shows how to recover causal effect from confounded data

#### Example Usage

```python
import pandas as pd
import json

# Load simulated data
sim_data = pd.read_csv('data/processed/simulated_email_campaigns.csv')

# Show confounding
print(f"Email recipients: {sim_data['received_email'].mean():.1%}")
print(f"RFM (email group): {sim_data[sim_data['received_email']]['rfm_score'].mean():.2f}")
print(f"RFM (no email): {sim_data[~sim_data['received_email']]['rfm_score'].mean():.2f}")

# Naive vs true effect
naive_effect = (
    sim_data[sim_data['received_email']]['purchased_this_week_observed'].mean() -
    sim_data[~sim_data['received_email']]['purchased_this_week_observed'].mean()
)
true_effect = sim_data['individual_treatment_effect'].mean()

print(f"\nNaive effect: {naive_effect:.1%} (biased)")
print(f"True effect: {true_effect:.1%} (causal)")
print(f"Bias: {naive_effect - true_effect:.1%}")

# Load ground truth
with open('data/processed/ground_truth.json', 'r') as f:
    ground_truth = json.load(f)
    print(f"Ground truth: +{ground_truth['base_email_effect']*100:.1f}%")
```

#### Running the Simulation

```bash
# Generate simulated email campaigns
.venv/bin/python src/data/simulate_email_campaigns.py

# Explore the simulation in notebook
jupyter notebook notebooks/02_email_campaign_simulation.ipynb
```

#### Learning Objectives

This simulation teaches:
1. **Confounding is everywhere** in marketing data
2. **Naive comparisons are dangerously biased**
3. **Causal inference methods** can recover true effects
4. **Heterogeneous treatment effects** matter for targeting
5. **Counterfactuals** are the foundation of causal inference

#### Next Steps for Causal Inference

Now that we have realistic confounding and a known true effect, we can test methods to recover the 9.5% causal effect:

1. **Propensity Score Matching**: Match similar customers across email/no-email groups
2. **Inverse Probability Weighting**: Weight observations by inverse propensity to receive email
3. **Regression Adjustment**: Control for confounding variables
4. **Double Machine Learning**: ML-based causal inference
5. **Difference-in-Differences**: Use before/after variations

Each method should recover the true effect (9.5%) from the biased naive estimate (16.0%)!

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

The project now includes **realistic email campaign simulation with confounding**, ready for causal inference learning and testing!

### ✅ **Completed: Email Campaign Simulation**

We've created a **realistic simulation** where:
- Email assignment is **confounded** (based on customer characteristics)
- True causal effect is **known** (10% base + RFM interactions)
- Ground truth is **saved** for validation
- **Naive analysis is biased** (16.0% observed vs 9.5% true effect)

This is perfect for **learning causal inference methods** without the complexity of real-world data!

---

### 🚀 **Ready to Implement: Causal Inference Methods**

Test methods to recover the **true 9.5% effect** from confounded data:

1. **Propensity Score Matching** (Easiest to start!)
   - Match customers with similar characteristics
   - Compare email recipients to similar non-recipients
   - Should recover ~9.5% effect

2. **Inverse Probability Weighting (IPW)**
   - Weight observations by inverse propensity to receive email
   - Corrects for selection bias
   - Robust to model misspecification

3. **Regression Adjustment**
   - Include confounding variables as controls
   - Linear/logistic regression with treatment indicator
   - Simple but relies on correct model

4. **Double Machine Learning (DML)**
   - Use ML to control for confounding
   - Residuals-on-residuals approach
   - Flexible, handles non-linearities

5. **Difference-in-Differences**
   - Use before/after variation
   - Compare changes over time
   - Controls for time-invariant confounders

### 📊 **Validation Approach**

For each method:
1. Apply to `simulated_email_campaigns.csv`
2. Compare estimate to **ground truth** (9.5%)
3. Measure bias and variance
4. Test on different customer segments
5. Analyze heterogeneous effects

### 💡 **Business Applications**

Once methods are validated:
- **Measure true ROI** of email campaigns
- **Optimize targeting** based on heterogeneous effects
- **Personalize frequency** by customer segment
- **A/B test strategies** with causal inference

### 🎯 **Learning Path**

1. **Understand confounding** (Notebook 02)
2. **Implement PSM** (start here!)
3. **Try IPW and regression**
4. **Advance to DML**
5. **Apply to real data**

### 📈 **Key Insights from Simulation**

- **Medium RFM customers** respond best (12.2% effect)
- **Confounding creates 68% bias** (16.0% vs 9.5%)
- **Targeting strategies** matter for ROI
- **Causal inference is essential** for marketing measurement

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