# Visualization Gallery

This directory contains all visualizations generated from the causal inference project.

## 📊 Total Visualizations: 15 plots

---

## Python Scripts (6 plots)

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

### 2. Propensity Score Matching (`src/causal/propensity_score_matching_save.py`)
Generated when running PSM to recover true causal effects from confounded data.

**Plot 4: 04_propensity_scores.png**
- Propensity score distribution by group
- ROC curve for propensity model (AUC = 0.661)

**Plot 5: 05_covariate_balance.png**
- Before vs After matching standardized differences
- Absolute balance improvement comparison
- Shows all 5 features improved after matching

**Plot 6: 06_psm_results_summary.png**
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

### Propensity Score Matching (Solution)
- **Method**: Match customers with similar propensity scores
- **Result**: Recover 11.2% effect (close to true 9.5%!)
- **Bias Reduction**: From 6.5% to 1.7% (74% improvement!)
- **Balance**: All 5 features achieved better balance after matching
- **Validation**: Propensity model AUC = 0.661, 100% match rate

---

## 📂 Directory Structure

```
src/visualization/
├── README.md                           # This file
├── extract_notebook_plots.py           # Script to extract plots from notebooks
├── save_plots.py                       # Utility for saving matplotlib plots
│
├── Python Scripts Output (6 plots):
│   ├── 01_naive_comparison.png
│   ├── 02_confounding_visualizations.png
│   ├── 03_naive_vs_true_comparison.png
│   ├── 04_propensity_scores.png
│   ├── 05_covariate_balance.png
│   └── 06_psm_results_summary.png
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

# Propensity Score Matching
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

2. **Learn the Solution**: See PSM in action
   - See 04_propensity_scores.png
   - See 05_covariate_balance.png
   - See 06_psm_results_summary.png

3. **Deep Dive**: Explore notebooks for detailed analysis
   - See 01_initial_eda plots (data exploration)
   - See 02_email_campaign_simulation plots (confounding)
   - See 03_naive_analysis_fails plots (problem demonstration)

---

## 📊 Summary Statistics

**Total Visualizations**: 15
- Python scripts: 6 plots
- Jupyter notebooks: 9 plots

**Key Results Visualized**:
- Naive effect: 16.0% (biased)
- PSM effect: 11.2% (causal)
- True effect: 9.5% (ground truth)
- Bias reduction: 74%

**Visualization Methods**:
- Bar charts
- Histograms
- Violin plots
- Box plots
- ROC curves
- Confidence intervals
- Balance comparisons

---

Generated: 2025-11-16
Project: Causal Impact of Email Marketing on Purchase Behavior
