# Visualization Gallery

This directory contains all visualizations generated from the causal inference project.

## 📊 Total Visualizations: 19+ plots

---

## Python Scripts (9+ plots)

### 1. Naive Analysis (`src/data/naive_analysis_save.py`)
Generated when running the naive analysis to demonstrate confounding bias.

**Plot 1: 01_naive_comparison.png**
- Naive purchase rate comparison (Email vs No Email)
- Sample sizes by group
- Observed effect size visualization
- Direct comparison of purchase rates

**Plot 2: 02_confounding_visualizations.png**
- RFM score distribution by email status
- Days since last purchase (capped at 100)
- Total past purchases comparison
- Average order value (capped at £100)
- Customer tenure comparison
- Correlation with email receipt
- Standardized differences before matching
- RFM score histogram comparison
- Days since purchase distribution

**Plot 3: 03_naive_vs_true_comparison.png**
- Side-by-side comparison: Naive vs True effect
- Bias decomposition visualization
- Shows 16.0% naive vs 9.5% true effect

### 2. Propensity Score Estimation (`src/causal/estimate_propensity_scores.py`)
Complete propensity score estimation workflow with diagnostics.

**Plot 4: propensity_score_diagnostics.png**
- 12-panel comprehensive diagnostic plots:
  - Propensity score distributions by treatment group
  - ROC curve (AUC = 0.661)
  - Box plots and violin plots
  - Q-Q plots and cumulative distributions
  - Standardized differences (before matching)
  - Feature vs propensity scatter plots

**Plot 5: propensity_score_summary.png**
- 4-panel summary visualization:
  - Distribution comparison (histograms)
  - Box plots with statistics
  - Coefficient importance (bar chart)
  - Summary statistics by group

### 3. Quick Start Guide (`src/causal/quick_start_propensity_scores.py`)
Quick visualization of propensity score usage and interpretation.

**Plot 6: propensity_scores_quick_start.png**
- 3-panel quick reference:
  - Distribution by treatment group
  - Box plots comparison
  - Propensity vs outcome scatter plot

### 4. Propensity Score Matching v2 (`src/causal/propensity_score_matching_v2.py`)
Comprehensive PSM implementation with Love plots and bootstrap CI.

**Plot 7: love_plot_balance.png**
- Love plot showing standardized differences before/after matching
- Horizontal bar chart with good balance threshold (±0.1)
- Clear visualization of balance improvement for all covariates

**Plot 8: psm_results_comprehensive.png**
- 6-panel comprehensive results:
  1. Covariate balance (Love plot)
  2. Treatment effect estimates (Naive vs PSM vs True)
  3. 95% confidence interval with true effect marker
  4. Bootstrap distribution histogram
  5. Balance improvement by covariate
  6. Summary statistics panel

### 5. Legacy Propensity Score Matching (`src/causal/propensity_score_matching_save.py`)
Original PSM implementation (superseded by v2).

**Plot 9: 04_propensity_scores.png**
- Propensity score distribution by group
- ROC curve for propensity model (AUC = 0.661)

**Plot 10: 05_covariate_balance.png**
- Before vs After matching standardized differences
- Absolute balance improvement comparison
- Shows all 5 features improved after matching

**Plot 11: 06_psm_results_summary.png**
- Effect estimates comparison (Naive vs PSM vs True)
- Absolute bias comparison
- 95% confidence interval
- Covariate balance improvement

---

## Jupyter Notebooks (9 plots)

### 1. Initial EDA (`notebooks/01_initial_eda.ipynb`)
Exploratory data analysis on UCI Online Retail dataset.

**Plot 7: 01_initial_eda_plot_001.png**
- [Generated from notebook -EDA analysis 1]

**Plot 8: 01_initial_eda_plot_002.png**
- [Generated from notebook -EDA analysis 2]

**Plot 9: 01_initial_eda_plot_003.png**
- [Generated from notebook -EDA analysis 3]

**Plot 10: 01_initial_eda_plot_004.png**
- [Generated from notebook -EDA analysis 4]

### 2. Email Campaign Simulation (`notebooks/02_email_campaign_simulation.ipynb`)
Demonstrates confounding in realistic email campaign simulation.

**Plot 11: 02_email_campaign_simulation_plot_001.png**
- [Generated from notebook -Confounding visualization 1]

**Plot 12: 02_email_campaign_simulation_plot_002.png**
- [Generated from notebook -Confounding visualization 2]

### 3. Naive Analysis Fails (`notebooks/03_naive_analysis_fails.ipynb`)
Demonstrates why naive comparisons fail with confounding.

**Plot 13: 03_naive_analysis_fails_plot_001.png**
- [Generated from notebook -Naive comparison 1]

**Plot 14: 03_naive_analysis_fails_plot_002.png**
- [Generated from notebook -Confounding analysis 1]

**Plot 15: 03_naive_analysis_fails_plot_003.png**
- [Generated from notebook -Naive vs True 1]

---

## 🎯 Key Visual Insights

### Naive Analysis (Bias Demonstration)
- **Problem**: Naive comparison shows 16.0% effect
- **Reality**: True causal effect is only 9.5%
- **Bias**: 6.5 percentage points (68% overestimate!)
- **Cause**: Email recipients are systematically different (higher RFM, more recent purchases)

### Propensity Score Estimation (Foundation)
- **Model**: Logistic regression P(email | customer features)
- **Performance**: AUC = 0.661 (moderate predictive power)
- **Key Driver**: Days since last purchase (coef = -0.422)
- **Common Support**: Excellent overlap (99.98% of units)
- **Features**: 5 confounders (recency, frequency, monetary, tenure, RFM)

### Propensity Score Matching v2 (Solution)
- **Method**: 1:1 nearest neighbor with caliper = 0.0078
- **Matched Pairs**: 112,722 (100% match rate)
- **Result**: Recover 11.2% effect (close to true 9.5%!)
- **Bias Reduction**: From 6.5% to 1.7% (74% improvement!)
- **Balance Improvement**:
  - Before: 1/8 covariates well-balanced
  - After: 6/8 covariates well-balanced
  - Mean |Std Diff| reduced by 67.3%
- **Statistical Significance**: p < 0.0001, 95% CI: [10.8%, 11.5%]
- **Key Feature**: `days_since_last_purchase` had largest improvement (0.506 → 0.040)

---

## 📂 Directory Structure

```
src/visualization/
├── README.md                           # This file
├── extract_notebook_plots.py           # Script to extract plots from notebooks
├── save_plots.py                       # Utility for saving matplotlib plots
│
├── Python Scripts Output (11+ plots):
│   ├── 01_naive_comparison.png                 # Naive analysis
│   ├── 02_confounding_visualizations.png       # Confounding demo
│   ├── 03_naive_vs_true_comparison.png         # Bias visualization
│   ├── propensity_score_diagnostics.png        # 12-panel PS diagnostics
│   ├── propensity_score_summary.png            # 4-panel PS summary
│   ├── propensity_scores_quick_start.png       # Quick reference
│   ├── love_plot_balance.png                   # Love plot (balance)
│   ├── psm_results_comprehensive.png           # 6-panel PSM results
│   ├── 04_propensity_scores.png                # Legacy PSM
│   ├── 05_covariate_balance.png                # Legacy balance
│   └── 06_psm_results_summary.png              # Legacy summary
│
└── Notebook Plots (9 plots):
    ├── 01_initial_eda_plot_001.png
    ├── 01_initial_eda_plot_002.png
    ├── 01_initial_eda_plot_003.png
    ├── 01_initial_eda_plot_004.png
    ├── 02_email_campaign_simulation_plot_001.png
    ├── 02_email_campaign_simulation_plot_002.png
    ├── 03_naive_analysis_fails_plot_001.png
    ├── 03_naive_analysis_fails_plot_002.png
    └── 03_naive_analysis_fails_plot_003.png
```

---

## 🔬 How to Regenerate Plots

### Python Scripts
```bash
# Naive Analysis
source .venv/bin/activate
python src/data/naive_analysis_save.py

# Propensity Score Estimation
source .venv/bin/activate
python src/causal/estimate_propensity_scores.py

# Quick Start Guide
source .venv/bin/activate
python src/causal/quick_start_propensity_scores.py

# Propensity Score Matching v2 (Recommended)
source .venv/bin/activate
python src/causal/propensity_score_matching_v2.py

# Legacy Propensity Score Matching (Superseded)
source .venv/bin/activate
python src/causal/propensity_score_matching_save.py
```

### Notebooks
```bash
# Execute all notebooks
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/01_initial_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_email_campaign_simulation.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_naive_analysis_fails.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_propensity_score_matching.ipynb

# Extract plots from notebooks
python src/visualization/extract_notebook_plots.py
```

---

## 💡 Learning Progression

1. **Start Here**: Understand the problem (Naive Analysis plots)
   - See 01_naive_comparison.png
   - See 02_confounding_visualizations.png
   - See 03_naive_vs_true_comparison.png

2. **Foundation**: Learn Propensity Score Estimation
   - See propensity_score_diagnostics.png (12-panel diagnostics)
   - See propensity_score_summary.png (4-panel summary)
   - See propensity_scores_quick_start.png (quick reference)

3. **Solution**: See PSM in action (v2 - recommended)
   - See love_plot_balance.png (balance visualization)
   - See psm_results_comprehensive.png (6-panel results)
   - Key: 74% bias reduction achieved!

4. **Deep Dive**: Explore notebooks for detailed analysis
   - See 01_initial_eda plots (data exploration)
   - See 02_email_campaign_simulation plots (confounding)
   - See 03_naive_analysis_fails plots (problem demonstration)

5. **Legacy Methods**: Original PSM implementation
   - See 04_propensity_scores.png
   - See 05_covariate_balance.png
   - See 06_psm_results_summary.png

---

## 📊 Summary Statistics

**Total Visualizations**: 20+
- Python scripts: 11+ plots
- Jupyter notebooks: 9 plots

**Key Results Visualized**:
- Naive effect: 16.0% (biased, 6.5% absolute bias)
- PSM v2 effect: 11.2% (causal, 1.7% absolute bias)
- True effect: 9.5% (ground truth)
- **Bias reduction: 74.1%** ✅

**Propensity Score Model**:
- AUC: 0.661
- Key predictor: Days since last purchase (coef = -0.422)
- Common support: 99.98% of units

**PSM Matching Results**:
- Matched pairs: 112,722 (100% match rate)
- Caliper: 0.0078
- Balance: 6/8 covariates well-balanced (vs 1/8 before)
- Mean |Std Diff| reduction: 67.3%

**Visualization Methods**:
- Bar charts
- Histograms
- Violin plots
- Box plots
- ROC curves
- Confidence intervals
- Love plots
- Balance comparisons
- Bootstrap distributions
- Comprehensive 6-panel results

---

Generated: 2025-11-16
Project: Causal Impact of Email Marketing on Purchase Behavior
