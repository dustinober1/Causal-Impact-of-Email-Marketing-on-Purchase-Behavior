# 📊 Inverse Probability Weighting (IPW) - Complete Implementation

**Date**: 2025-11-16
**Project**: Causal Impact of Email Marketing on Purchase Behavior

---

## ✅ What Was Implemented

I've created a comprehensive **Inverse Probability Weighting (IPW)** implementation that uses propensity scores to create a pseudo-population where treatment assignment is as-if random.

### 📁 Files Created

1. **Main Implementation**: `src/causal/inverse_probability_weighting.py` (21 KB)
   - Complete `InverseProbabilityWeighting` class
   - IPW estimator with trimming and stabilization options
   - Bootstrap standard errors
   - Diagnostic plots for weight quality
   - Comparison to naive and true effects

2. **Visualizations Created**:
   - `src/visualization/ipw_diagnostics.png` (203 KB)
     - Weight distributions by treatment group
     - Propensity score distributions
     - Weight vs propensity score relationship
     - Summary statistics panel
   - `src/visualization/ipw_results_comprehensive.png` (203 KB)
     - Method comparison (Naive vs IPW)
     - Bootstrap distribution of ATE
     - Weight distributions
     - Complete summary statistics

---

## 🎯 Key Results

### IPW Estimates
- **Average Treatment Effect**: 13.6% (13.56 percentage points)
- **E[Y(1)]**: 33.92%
- **E[Y(0)]**: 20.37%
- **Naive Estimate**: 16.0%

### Bootstrap Standard Errors (200 samples)
- **Point Estimate**: 13.6%
- **Bootstrap SE**: 0.39 percentage points
- **95% CI**: [12.8%, 14.3%]
- **Z-statistic**: 35.19
- **P-value**: < 0.0001 (highly significant)

### Weight Quality Diagnostics
- **Mean Weight (Treated)**: 1.22
- **Mean Weight (Control)**: 5.36
- **Max Weight (Treated)**: 1.84
- **Max Weight (Control)**: 13.07
- **% Trimmed**: 2.0% (extreme propensity scores)

### Validation Against True Effect

| Metric | Value |
|--------|-------|
| **IPW Estimate** | 13.6% |
| **True Effect** | 9.5% |
| **Expected (Ground Truth)** | 10.0% |
| **IPW Bias** | +4.1 percentage points |

**Method Comparison:**
- **Naive**: 16.0% (bias: +6.5 pp)
- **IPW**: 13.6% (bias: +4.1 pp) ✅
- **PSM**: 11.2% (bias: +1.7 pp) ✅
- **AIPW**: 12.7% (bias: +3.2 pp) ✅
- **DiD**: 0.5% (bias: -9.3 pp) ❌

**Validation Status:**
- 95% CI does NOT include true effect (12.8%-14.3% vs 9.5%)
- IPW overestimates the effect but is better than naive
- 37% bias reduction compared to naive (4.1pp vs 6.5pp)

---

## 📋 IPW Implementation Details

### 1. **IPW Weight Calculation** ✅
```python
def calculate_ipw_weights(self, data, treatment_col='received_email'):
    T = data[treatment_col].values
    e = self.propensity_scores

    # Trim extreme propensity scores
    if self.trim_percentile > 0:
        trim_lower = np.percentile(e, self.trim_percentile)
        trim_upper = np.percentile(e, 100 - self.trim_percentile)
        e_trimmed = np.clip(e, trim_lower, trim_upper)
    else:
        e_trimmed = e

    # Calculate IPW weights
    # w_i = T_i / e_i for treated, (1-T_i) / (1-e_i) for control
    weights_treated = T / e_trimmed
    weights_control = (1 - T) / (1 - e_trimmed)

    weights = weights_treated + weights_control

    return weights, e_trimmed
```

### 2. **IPW Estimator** ✅
```python
def estimate_ipw(self, data, outcome_col='purchased_this_week_observed'):
    Y = data[outcome_col].values
    T = data['received_email'].values

    # Calculate IPW weights
    weights, e_trimmed = self.calculate_ipw_weights(data)

    # IPW estimators
    # E[Y(1)] = E[T*Y/e(X)]
    # E[Y(0)] = E[(1-T)*Y/(1-e(X))]

    treated_weighted = np.sum(T * Y / e_trimmed) / len(data)
    control_weighted = np.sum((1 - T) * Y / (1 - e_trimmed)) / len(data)
    ipw_ate = treated_weighted - control_weighted

    return {
        'ate': ipw_ate,
        'e_y1': treated_weighted,
        'e_y0': control_weighted,
        'weights': weights
    }
```

### 3. **Bootstrap Standard Errors** ✅
```python
def bootstrap_se(self, data, outcome_col='purchased_this_week_observed', n_bootstrap=500):
    n = len(data)
    bootstrap_ates = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_data = data.iloc[boot_idx].copy()
        boot_propensity = self.propensity_scores[boot_idx]

        Y = boot_data[outcome_col].values
        T = boot_data['received_email'].values

        # Calculate weights
        if self.trim_percentile > 0:
            trim_lower = np.percentile(boot_propensity, self.trim_percentile)
            trim_upper = np.percentile(boot_propensity, 100 - self.trim_percentile)
            e_trimmed = np.clip(boot_propensity, trim_lower, trim_upper)
        else:
            e_trimmed = boot_propensity

        # IPW estimate
        treated_weighted = np.sum(T * Y / e_trimmed) / len(data)
        control_weighted = np.sum((1 - T) * Y / (1 - e_trimmed)) / len(data)
        ipw_ate = treated_weighted - control_weighted

        bootstrap_ates.append(ipw_ate)

    # Calculate statistics
    se_bootstrap = np.std(bootstrap_ates)
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)

    return {
        'ate_bootstrap': bootstrap_ates,
        'se': se_bootstrap,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
```

---

## 📊 What the Results Tell Us

### 1. **IPW Performance**
- **IPW ATE**: 13.6% vs true 9.5%
- **Bias**: +4.1 percentage points
- **Improvement over naive**: 37% bias reduction (6.5pp → 4.1pp)

### 2. **Weight Quality**
**Concerning Findings:**
- Control group has much higher mean weights (5.36 vs 1.22 for treated)
- Some extreme weights (max control weight = 13.07)
- High variance in weights indicates potential issues

**Why This Happens:**
- Control group is small (18.3% of data)
- Many control observations have very low propensity scores
- Low propensity → high weights (1/(1-e) can be very large)
- Weight instability leads to high variance estimates

### 3. **Comparison to Other Methods**

| Method | Estimate | Bias | Variance | Valid Here |
|--------|----------|------|----------|------------|
| **Naive** | 16.0% | +6.5 pp | Low | ❌ No (confounded) |
| **IPW** | 13.6% | +4.1 pp | Medium | ✅ Yes, but unstable |
| **PSM** | 11.2% | +1.7 pp | Low | ✅ Yes |
| **AIPW** | 12.7% | +3.2 pp | Medium | ✅ Yes |
| **DiD** | 0.5% | -9.3 pp | Low | ❌ No (wrong design) |

**Key Insight**: IPW works but has higher variance than matching due to extreme weights. This is a classic example of the trade-off between methods.

### 4. **Why IPW Overestimated**
1. **Extreme weights**: Control weights up to 13.07 create instability
2. **Trimmed only 2%**: More aggressive trimming might help
3. **No stabilization**: Stabilized weights could reduce variance
4. **Propensity model**: Moderate AUC (0.659) limits performance

### 5. **When IPW Works Well**
- Good overlap in propensity scores (no extreme values)
- Adequate sample size in both groups
- Good propensity model (high AUC > 0.7)
- Balance between treated and control groups

### 6. **IPW Limitations Demonstrated**
- **Weight Instability**: Small sample in control group
- **High Variance**: Extreme weights inflate uncertainty
- **Sensitivity**: Results depend heavily on trimming decisions
- **Not Robust**: Requires correct propensity model specification

---

## 💡 Key Learnings

### IPW Theory
1. **Identification**: Requires unconfoundedness and positivity
2. **Efficiency**: More efficient than matching when models are correct
3. **Variance**: Can be high with extreme weights
4. **Trimming**: Essential for practical implementation

### Practical Considerations
1. **Weight Diagnostics**: Always check weight distributions
2. **Trimming**: Remove extreme propensity scores (1-5%)
3. **Stabilization**: Use stabilized weights to reduce variance
4. **Overlap**: Ensure good common support
5. **Sample Size**: Need adequate observations in both groups

### Method Selection Guide

**Use IPW when**:
- ✅ You have good propensity model (AUC > 0.7)
- ✅ Balanced treatment groups
- ✅ Good overlap in propensity scores
- ✅ Large sample size
- ❌ Avoid with extreme weights or small control groups

**Use PSM instead**:
- ✅ Small samples
- ✅ Extreme propensity scores
- ✅ Need transparent diagnostics
- ✅ Want to visualize matched pairs

**Use AIPW instead**:
- ✅ Want robustness to model misspecification
- ✅ Have both propensity and outcome data
- ✅ Want efficiency gains

---

## 📈 Visualizations

### 1. Diagnostic Plots (`ipw_diagnostics.png`)

**Panel 1: Weight Distribution by Treatment**
- Treated (green) vs Control (red) IPW weights
- Shows much higher variance in control weights
- Some extreme values in control group

**Panel 2: Propensity Score Distribution**
- Overlap between treated and control
- Most control observations have low propensity
- Explains why control weights are high

**Panel 3: Weight vs Propensity Score**
- Negative relationship (expected)
- Control weights blow up as propensity → 0
- Visual demonstration of the problem

**Panel 4: Summary Statistics**
- Weight quality metrics
- Trimming information
- Effective sample sizes

### 2. Comprehensive Results (`ipw_results_comprehensive.png`)

**Panel 1: Method Comparison**
- Naive (16.0%) vs IPW (13.6%)
- Shows bias reduction but still overestimation

**Panel 2: Bootstrap Distribution**
- Distribution of 200 bootstrap samples
- True effect (9.5%) shown for comparison
- Shows IPW overestimation

**Panel 3: Weight Distributions**
- Highlights weight instability issue
- Control group weights more variable

**Panel 4: Complete Summary**
- All key metrics and diagnostics

---

## 🔍 Technical Implementation

### Complete Usage Example
```python
from src.causal.inverse_probability_weighting import InverseProbabilityWeighting
import pandas as pd

# Load data
data = pd.read_csv('data/processed/data_with_propensity_scores.csv')

# Initialize IPW estimator
ipw = InverseProbabilityWeighting(
    trim_percentile=1,  # Remove extreme 1% of propensity scores
    stabilize=False,    # Don't use stabilized weights
    random_state=42
)

# Set propensity scores (from previous estimation)
ipw.propensity_scores = data['propensity_score'].values

# Estimate IPW
ipw_result = ipw.estimate_ipw(data)

# Bootstrap standard errors
bootstrap_results = ipw.bootstrap_se(data, n_bootstrap=200)

# Create diagnostic plots
ipw.diagnostic_plots(data)

# Compare to true effect
comparison = ipw.compare_to_true_effect()

# View results
print(f"IPW ATE: {ipw_result['ate']:.4f}")
print(f"95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
```

### Full Script Execution
```bash
source .venv/bin/activate
python src/causal/inverse_probability_weighting.py
```

---

## 🎓 Learning Outcomes

### What You Learned
1. ✅ **IPW Theory**: Weighting by inverse propensity scores
2. ✅ **Weight Diagnostics**: Checking for extreme values
3. ✅ **Trimming**: Removing problematic observations
4. ✅ **Stabilization**: Reducing weight variance
5. ✅ **Bootstrap Methods**: Robust standard error estimation
6. ✅ **Method Trade-offs**: IPW vs PSM vs AIPW

### Key Concepts Mastered
- ✅ **IPW Estimator**: E[T*Y/e(X)] - E[(1-T)*Y/(1-e(X))]
- ✅ **Positivity**: P(T=1|X) > 0 for all X
- ✅ **Weight Instability**: Problem with extreme weights
- ✅ **Effective Sample Size**: How weights reduce precision
- ✅ **Trimming**: Practical solution for weight issues

### Advanced Topics Covered
- **IPW Math**: Weight each observation by inverse propensity score
- **Stabilized Weights**: w*_i = w_i * P(T=t) for variance reduction
- **Effective N**: 1/(sum(w_i^2)) measures information loss
- **Weight Diagnostics**: Visual and numerical checks

---

## 🚀 Next Steps

### Recommended Extensions
1. **Stabilized IPW**
   - Use marginal treatment probability
   - Reduces variance with extreme weights
   - Compare performance

2. **Doubly Robust IPW (AIPW)**
   - Already implemented!
   - Compare directly to pure IPW
   - Shows robustness benefits

3. **Trimmed IPW**
   - More aggressive trimming (5%, 10%)
   - Find optimal trimming level
   - Bias-variance trade-off

4. **Weight Averaging**
   - Average multiple IPW estimates
   - Different trimming levels
   - More robust inference

### Weight Quality Improvements
```python
# Try different configurations
configs = [
    {'trim': 0, 'stabilize': False},    # Raw IPW
    {'trim': 1, 'stabilize': False},    # Trimmed IPW
    {'trim': 1, 'stabilize': True},     # Stabilized IPW
    {'trim': 5, 'stabilize': True},     # Aggressive trimming
]

# Compare performance across configs
```

---

## 📝 Summary

**IPW Successfully Implemented with Important Lessons on Weight Stability!**

✅ **IPW Implementation**:
- Estimate: 13.6% (vs true 9.5%)
- Bias: 4.1 pp (37% better than naive)
- 95% CI: [12.8%, 14.3%]
- Highly significant (p < 0.0001)

⚠️ **Weight Quality Issues**:
- High variance in control weights (mean=5.36, max=13.07)
- Only 2% trimmed (insufficient)
- Weight instability creates variance inflation

✅ **Key Advantages Demonstrated**:
- Uses all observations (unlike matching)
- Simple and transparent
- Valid when propensity model is good
- Can be stabilized and trimmed

❌ **Limitations Shown**:
- Sensitive to extreme weights
- Higher variance than matching
- Requires good overlap
- Not robust to model misspecification

💡 **Key Insight**:
**IPW is powerful but fragile** - works well with good overlap and balanced groups, but PSM and AIPW are more robust in practice. This implementation demonstrates both the potential and limitations of IPW - essential knowledge for causal inference!

---

**Generated**: 2025-11-16
**Project**: Causal Impact of Email Marketing on Purchase Behavior
**Status**: ✅ Complete - IPW implemented with comprehensive diagnostics and weight quality analysis!