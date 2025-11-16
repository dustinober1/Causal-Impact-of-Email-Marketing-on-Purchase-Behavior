# Visualization Gallery

This directory contains all visualizations generated from the causal inference project.

## 📊 Total Visualizations: 28+ plots

---

## Python Scripts (17+ plots)

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

### 6. Difference-in-Differences (`src/causal/difference_in_differences.py`)
Complete DiD implementation with parallel trends and event study.

**Plot 12: did_parallel_trends.png**
- 2-panel parallel trends visualization:
  - Pre-treatment trends by group (weeks 1-9)
  - Difference in purchase rates over time
  - Treatment start line at week 10
  - Parallel trends test result (p=0.9495)

**Plot 13: did_event_study.png**
- Event study plot with dynamic treatment effects
- Leads and lags around treatment week
- Pre-treatment period highlighted
- Shows treatment effect evolution

**Plot 14: did_results_comprehensive.png**
- 4-panel comprehensive DiD results:
  1. Mean outcomes by group and time period
  2. DiD coefficient with confidence interval
  3. Parallel trends visualization
  4. Summary statistics

**Key Results**:
- DiD estimate: 0.5% (vs true 9.5%)
- Parallel trends: Satisfied (p=0.9495)
- Note: Wrong method for this data (no true policy change)

### 7. Inverse Probability Weighting (`src/causal/inverse_probability_weighting.py`)
IPW implementation with weight diagnostics and bootstrap CI.

**Plot 15: ipw_diagnostics.png**
- 4-panel IPW diagnostic plots:
  1. Weight distribution by treatment group
  2. Propensity score distribution
  3. Weight vs propensity score relationship
  4. Summary statistics (mean, max weights, trimming %)

**Plot 16: ipw_results_comprehensive.png**
- 4-panel comprehensive results:
  1. Method comparison (Naive vs IPW)
  2. Bootstrap distribution of ATE
  3. IPW weight distribution
  4. Complete summary statistics

**Key Results**:
- IPW estimate: 13.6% (bias: +4.1 pp)
- Weight issue: Control weights unstable (max=13.07)
- Trimmed: 2.0% of extreme propensity scores
- Shows IPW works but requires careful weight management

### 8. Doubly Robust Methods (`src/causal/doubly_robust.py`)
Complete AIPW and T-Learner implementation with heterogeneous effects.

**Plot 17: doubly_robust_results.png**
- 4-panel comprehensive results:
  1. CATE by RFM segment (with error bars)
  2. Distribution of CATE (individual treatment effects)
  3. CATE vs RFM score (scatter plot with trend line)
  4. Summary statistics (AIPW, T-Learner, model performance)

**Key Results**:
- AIPW ATE: 12.7% (bias: +3.2 pp)
- T-Learner mean CATE: 12.8%
- CATE range: [-3.3%, +22.6%] (significant heterogeneity!)
- All RFM segments show positive effects (~12.8%)
- Demonstrates doubly robust properties (works despite poor outcome models)

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
├── Python Scripts Output (17+ plots):
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
│   ├── 06_psm_results_summary.png              # Legacy summary
│   ├── did_parallel_trends.png                 # DiD parallel trends
│   ├── did_event_study.png                     # DiD event study
│   ├── did_results_comprehensive.png           # DiD 4-panel results
│   ├── ipw_diagnostics.png                     # IPW diagnostics
│   ├── ipw_results_comprehensive.png           # IPW 4-panel results
│   └── doubly_robust_results.png               # AIPW/T-Learner results
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
# 1. Naive Analysis (shows the problem)
source .venv/bin/activate
python src/data/naive_analysis.py

# 2. Propensity Score Estimation
source .venv/bin/activate
python src/causal/estimate_propensity_scores.py

# 3. Quick Start Guide
source .venv/bin/activate
python src/causal/quick_start_propensity_scores.py

# 4. Propensity Score Matching v2 (Recommended)
source .venv/bin/activate
python src/causal/propensity_score_matching_v2.py

# 5. Difference-in-Differences (wrong for this data)
source .venv/bin/activate
python src/causal/difference_in_differences.py

# 6. Inverse Probability Weighting
source .venv/bin/activate
python src/causal/inverse_probability_weighting.py

# 7. Doubly Robust (AIPW + T-Learner)
source .venv/bin/activate
python src/causal/doubly_robust.py
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
   - **Key**: Naive = 16.0% vs True = 9.5% (68% bias!)

2. **Foundation**: Learn Propensity Score Estimation
   - See propensity_score_diagnostics.png (12-panel diagnostics)
   - See propensity_score_summary.png (4-panel summary)
   - See propensity_scores_quick_start.png (quick reference)
   - **Key**: AUC = 0.661, common support verified

3. **Solution**: See PSM in action (v2 - recommended)
   - See love_plot_balance.png (balance visualization)
   - See psm_results_comprehensive.png (6-panel results)
   - **Key**: 74% bias reduction (11.2% vs 9.5% true)

4. **Time-Based Methods**: Learn DiD
   - See did_parallel_trends.png (parallel trends test)
   - See did_event_study.png (dynamic effects)
   - See did_results_comprehensive.png (complete analysis)
   - **Key**: Parallel trends satisfied, but wrong method (0.5% estimate)

5. **Weighting Methods**: Learn IPW
   - See ipw_diagnostics.png (weight quality checks)
   - See ipw_results_comprehensive.png (results with uncertainty)
   - **Key**: Weight instability issue (control weights up to 13.07)

6. **Modern Methods**: Learn Doubly Robust
   - See doubly_robust_results.png (AIPW + T-Learner)
   - **Key**: AIPW = 12.7%, T-Learner shows heterogeneity (-3.3% to +22.6%)

7. **Deep Dive**: Explore notebooks for detailed analysis
   - See 01_initial_eda plots (data exploration)
   - See 02_email_campaign_simulation plots (confounding)
   - See 03_naive_analysis_fails plots (problem demonstration)

8. **Legacy Methods**: Original PSM implementation
   - See 04_propensity_scores.png
   - See 05_covariate_balance.png
   - See 06_psm_results_summary.png

---

## 📊 Summary Statistics

**Total Visualizations**: 28+
- Python scripts: 17+ plots
- Jupyter notebooks: 9 plots
- Total file size: ~3.5 MB

**Key Results Visualized**:
- Naive effect: 16.0% (biased, +6.5 pp bias) ❌
- PSM v2 effect: 11.2% (causal, +1.7 pp bias) 🥇 BEST
- DiD estimate: 0.5% (wrong method, -9.3 pp bias) ❌
- IPW estimate: 13.6% (causal, +4.1 pp bias) ⚠️
- AIPW estimate: 12.7% (causal, +3.2 pp bias) ✅
- T-Learner mean: 12.8% (heterogeneous, +3.3 pp bias) ✅
- True effect: 9.5% (ground truth)

**Method Rankings**:
1. 🥇 PSM: 11.2% (±1.7 pp bias) - Lowest bias
2. 🥈 AIPW: 12.7% (±3.2 pp bias) - Modern, doubly robust
3. 🥉 T-Learner: 12.8% (±3.3 pp bias) - Individual effects
4. IPW: 13.6% (±4.1 pp bias) - Weight issues
5. Naive: 16.0% (±6.5 pp bias) - Baseline only
6. DiD: 0.5% (±9.3 pp bias) - Wrong method

**Propensity Score Model**:
- AUC: 0.661 (moderate predictive power)
- Key predictor: Days since last purchase (coef = -0.422)
- Common support: 99.98% of units
- Sample size: 137,888 observations

**PSM Matching Results**:
- Matched pairs: 112,722 (100% match rate)
- Caliper: 0.0078
- Balance: 6/8 covariates well-balanced (vs 1/8 before)
- Mean |Std Diff| reduction: 67.3%
- Bootstrap CI: [10.8%, 11.5%]

**IPW Weight Diagnostics**:
- Mean weight (treated): 1.22
- Mean weight (control): 5.36 ⚠️
- Max weight (control): 13.07 ⚠️ (unstable)
- Trimmed: 2.0% of extreme propensity scores

**DiD Parallel Trends**:
- Test statistic: F = 14.72
- P-value: 0.9495 (satisfied)
- Treatment group: 668 customers
- Control group: 705 customers
- Note: Satisfies assumption but wrong for this data

**Doubly Robust Results**:
- AIPW ATE: 12.7% with 95% CI [12.0%, 13.3%]
- T-Learner mean CATE: 12.8%
- CATE range: [-3.3%, +22.6%] (significant heterogeneity!)
- RFM segments: All show ~12.8% (small differences)
- Bootstrap SE: 0.32 percentage points

**Visualization Methods**:
- Bar charts and histograms
- Violin plots and box plots
- ROC curves and QQ plots
- Confidence intervals and bootstrap distributions
- Love plots (balance visualization)
- Event study plots
- Scatter plots with trend lines
- Comprehensive multi-panel results
- Parallel trends visualizations
- Weight diagnostic plots
- Heterogeneous effects analysis

---

Generated: 2025-11-16
Project: Causal Impact of Email Marketing on Purchase Behavior
