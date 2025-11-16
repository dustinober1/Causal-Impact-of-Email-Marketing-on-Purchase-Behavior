"""
Propensity Score Estimation: P(email | customer features)

This script estimates propensity scores using logistic regression and creates
comprehensive diagnostics to assess the quality of propensity score modeling.

The propensity score e(x) = P(T=1 | X) is the probability of receiving treatment
(email) given customer characteristics (X). It's a crucial tool for:
- Propensity Score Matching
- Inverse Probability Weighting
- Stratification
- Covariate adjustment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_simulated_data():
    """Load the simulated email campaign data."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'processed'

    sim_data = pd.read_csv(data_dir / 'simulated_email_campaigns.csv')

    with open(data_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)

    return sim_data, ground_truth


def estimate_propensity_scores(data, features):
    """
    Estimate propensity scores using logistic regression.

    Parameters:
    -----------
    data : DataFrame
        Dataset with features and treatment indicator
    features : list
        List of feature column names to use in propensity model

    Returns:
    --------
    data : DataFrame
        Original data with added propensity scores
    model : LogisticRegression
        Fitted logistic regression model
    scaler : StandardScaler
        Fitted scaler for features
    """
    print("=" * 70)
    print("STEP 1: ESTIMATING PROPENSITY SCORES")
    print("=" * 70)

    # Prepare features and treatment
    X = data[features].copy()
    treatment = data['received_email'].values

    # Check for missing values
    print(f"\n📊 Data Summary:")
    print(f"   Sample size: {len(data):,}")
    print(f"   Treatment rate: {treatment.mean():.1%}")
    print(f"   Features: {len(features)}")

    # Standardize features
    print(f"\n📋 Feature List:")
    for i, feature in enumerate(features, 1):
        print(f"   {i}. {feature}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    print(f"\n🔄 Fitting logistic regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, treatment)

    # Predict propensity scores
    propensity_scores = model.predict_proba(X_scaled)[:, 1]

    # Add to dataframe
    data = data.copy()
    data['propensity_score'] = propensity_scores

    # Model performance
    auc = roc_auc_score(treatment, propensity_scores)
    print(f"   ✅ Model fitted successfully")
    print(f"   📈 AUC: {auc:.3f}")

    # Interpret coefficients
    print(f"\n📊 Model Coefficients (odds ratios):")
    print("-" * 70)
    print(f"{'Feature':<30} {'Coefficient':<15} {'OR':<10} {'Interpretation'}")
    print("-" * 70)

    for feature, coef in zip(features, model.coef_[0]):
        odds_ratio = np.exp(coef)
        direction = "↑" if coef > 0 else "↓"
        if odds_ratio > 1.5:
            interpretation = "Strong positive"
        elif odds_ratio > 1.2:
            interpretation = "Moderate positive"
        elif odds_ratio > 0.8:
            interpretation = "Weak effect"
        elif odds_ratio > 0.67:
            interpretation = "Moderate negative"
        else:
            interpretation = "Strong negative"

        print(f"{feature:<30} {coef:<+15.4f} {odds_ratio:<10.3f} {direction} {interpretation}")

    print("\n💡 Interpretation:")
    print("   OR > 1: Higher feature value → Higher probability of receiving email")
    print("   OR < 1: Higher feature value → Lower probability of receiving email")
    print("   OR = 1: No effect")

    return data, model, scaler


def check_common_support(data):
    """
    Check for common support (overlap) in propensity scores.

    Parameters:
    -----------
    data : DataFrame
        Data with propensity scores and treatment indicator

    Returns:
    --------
    dict : Common support diagnostics
    """
    print("\n" + "=" * 70)
    print("STEP 2: CHECKING COMMON SUPPORT")
    print("=" * 70)

    treated = data[data['received_email']]['propensity_score']
    control = data[~data['received_email']]['propensity_score']

    # Calculate overlap statistics
    min_treated = treated.min()
    max_treated = treated.max()
    min_control = control.min()
    max_control = control.max()

    overlap_min = max(min_treated, min_control)
    overlap_max = min(max_treated, max_control)

    print(f"\n📏 Propensity Score Ranges:")
    print(f"   Treated (email):     [{min_treated:.3f}, {max_treated:.3f}]")
    print(f"   Control (no email):  [{min_control:.3f}, {max_control:.3f}]")
    print(f"   Overlap:             [{overlap_min:.3f}, {overlap_max:.3f}]")

    # Check for lack of support
    no_support_low = (data['propensity_score'] < overlap_min).sum()
    no_support_high = (data['propensity_score'] > overlap_max).sum()
    no_support_total = no_support_low + no_support_high

    print(f"\n⚠️  Lack of Support:")
    print(f"   Below overlap: {no_support_low:,} units ({no_support_low/len(data):.1%})")
    print(f"   Above overlap: {no_support_high:,} units ({no_support_high/len(data):.1%})")
    print(f"   Total:         {no_support_total:,} units ({no_support_total/len(data):.1%})")

    if no_support_total > 0:
        print(f"\n   ⚠️  WARNING: {no_support_total} units lack common support!")
        print(f"      These units cannot be matched and should be excluded.")
    else:
        print(f"\n   ✅ Perfect common support - all units have overlap!")

    # Calculate percentiles
    print(f"\n📊 Propensity Score Percentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"{'Percentile':<12} {'Treated':<15} {'Control':<15} {'Difference'}")
    print("-" * 70)

    for p in percentiles:
        t_val = np.percentile(treated, p)
        c_val = np.percentile(control, p)
        diff = abs(t_val - c_val)
        print(f"{p:>7}%:      {t_val:>8.4f}      {c_val:>8.4f}      {diff:>8.4f}")

    return {
        'min_treated': min_treated,
        'max_treated': max_treated,
        'min_control': min_control,
        'max_control': max_control,
        'overlap_min': overlap_min,
        'overlap_max': overlap_max,
        'no_support_count': no_support_total,
        'no_support_pct': no_support_total / len(data)
    }


def identify_extreme_scores(data, thresholds=[0.01, 0.05, 0.95, 0.99]):
    """
    Identify units with extreme propensity scores.

    Parameters:
    -----------
    data : DataFrame
        Data with propensity scores
    thresholds : list
        Percentile thresholds to check

    Returns:
    --------
    dict : Extreme score diagnostics
    """
    print("\n" + "=" * 70)
    print("STEP 3: IDENTIFYING EXTREME PROPENSITY SCORES")
    print("=" * 70)

    propensity = data['propensity_score']

    # Calculate threshold values
    low_thresholds = {p: np.percentile(propensity, p) for p in thresholds[:2]}
    high_thresholds = {p: np.percentile(propensity, p) for p in thresholds[2:]}

    print(f"\n📊 Extreme Score Thresholds:")
    print(f"{'Threshold':<15} {'Value':<15} {'Count':<15} {'Percentage'}")
    print("-" * 70)

    extreme_stats = {}

    # Low scores
    for p, val in low_thresholds.items():
        count = (propensity <= val).sum()
        pct = count / len(propensity) * 100
        extreme_stats[f'low_{p}'] = {'value': val, 'count': count, 'pct': pct}
        print(f"{p}th percentile:    {val:<15.4f} {count:<15,} {pct:<15.2f}%")

    # High scores
    for p, val in high_thresholds.items():
        count = (propensity >= val).sum()
        pct = count / len(propensity) * 100
        extreme_stats[f'high_{p}'] = {'value': val, 'count': count, 'pct': pct}
        print(f"{p}th percentile:    {val:<15.4f} {count:<15,} {pct:<15.2f}%")

    # Units with scores near 0 or 1
    near_zero = (propensity < 0.01).sum()
    near_one = (propensity > 0.99).sum()

    print(f"\n🎯 Units Near Boundaries:")
    print(f"   Score < 0.01:  {near_zero:,} units ({near_zero/len(propensity):.2%})")
    print(f"   Score > 0.99:  {near_one:,} units ({near_one/len(propensity):.2%})")
    print(f"   Near boundaries: {near_zero + near_one:,} units ({(near_zero + near_one)/len(propensity):.2%})")

    extreme_stats['near_zero'] = near_zero
    extreme_stats['near_one'] = near_one

    # Recommendations
    print(f"\n💡 Recommendations:")
    if near_zero + near_one > len(propensity) * 0.05:
        print(f"   ⚠️  Many extreme scores ({((near_zero + near_one)/len(propensity)*100):.1f}%)")
        print(f"      Consider trimming or excluding extreme units")
    else:
        print(f"   ✅ Reasonable number of extreme scores ({((near_zero + near_one)/len(propensity)*100):.1f}%)")
        print(f"      No trimming needed for most analyses")

    if near_zero > 0:
        print(f"   ℹ️  Units with score ~0: Unlikely to receive email (even with high X)")
    if near_one > 0:
        print(f"   ℹ️  Units with score ~1: Very likely to receive email (regardless of X)")

    return extreme_stats


def create_diagnostic_plots(data, model, features, common_support_stats):
    """
    Create comprehensive diagnostic plots for propensity scores.

    Parameters:
    -----------
    data : DataFrame
        Data with propensity scores and treatment
    model : LogisticRegression
        Fitted model
    features : list
        Feature names
    common_support_stats : dict
        Common support statistics
    """
    print("\n" + "=" * 70)
    print("STEP 4: CREATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))

    # Plot 1: Propensity score distribution by treatment
    plt.subplot(4, 3, 1)
    treated = data[data['received_email']]['propensity_score']
    control = data[~data['received_email']]['propensity_score']

    plt.hist(control, bins=50, alpha=0.7, label='No Email (Control)',
             color='lightcoral', edgecolor='black', density=True)
    plt.hist(treated, bins=50, alpha=0.7, label='Received Email (Treated)',
             color='lightgreen', edgecolor='black', density=True)

    plt.axvline(common_support_stats['overlap_min'], color='blue',
                linestyle='--', alpha=0.7, label='Overlap Region')
    plt.axvline(common_support_stats['overlap_max'], color='blue',
                linestyle='--', alpha=0.7)

    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Distribution by Treatment', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: ROC Curve
    plt.subplot(4, 3, 2)
    treatment = data['received_email'].values
    propensity = data['propensity_score'].values

    fpr, tpr, _ = roc_curve(treatment, propensity)
    auc = roc_auc_score(treatment, propensity)

    plt.plot(fpr, tpr, color='darkgreen', linewidth=3,
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7,
             label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Model Performance (AUC = {auc:.3f})', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Box plots by treatment
    plt.subplot(4, 3, 3)
    plot_data = pd.DataFrame({
        'propensity_score': np.concatenate([treated, control]),
        'treatment': ['No Email'] * len(control) + ['Email'] * len(treated)
    })

    sns.boxplot(data=plot_data, x='treatment', y='propensity_score',
                palette=['lightcoral', 'lightgreen'])
    plt.title('Propensity Score Box Plots', fontweight='bold')
    plt.ylabel('Propensity Score')

    # Plot 4: Violin plots by treatment
    plt.subplot(4, 3, 4)
    sns.violinplot(data=plot_data, x='treatment', y='propensity_score',
                   palette=['lightcoral', 'lightgreen'])
    plt.title('Propensity Score Violin Plots', fontweight='bold')
    plt.ylabel('Propensity Score')

    # Plot 5: QQ plot comparing distributions
    plt.subplot(4, 3, 5)
    from scipy import stats
    treated_quantiles = np.percentile(treated, np.linspace(0, 100, 100))
    control_quantiles = np.percentile(control, np.linspace(0, 100, 100))

    plt.scatter(control_quantiles, treated_quantiles, alpha=0.6, s=30)
    min_val = min(treated_quantiles.min(), control_quantiles.min())
    max_val = max(treated_quantiles.max(), control_quantiles.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Control Quantiles')
    plt.ylabel('Treated Quantiles')
    plt.title('Q-Q Plot: Treated vs Control', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 6: Cumulative distribution
    plt.subplot(4, 3, 6)
    plt.hist(control, bins=50, cumulative=True, density=True, alpha=0.7,
             label='No Email', color='lightcoral', histtype='step', linewidth=2)
    plt.hist(treated, bins=50, cumulative=True, density=True, alpha=0.7,
             label='Email', color='lightgreen', histtype='step', linewidth=2)
    plt.xlabel('Propensity Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: Standardized differences before matching (raw covariates)
    plt.subplot(4, 3, 7)
    std_diffs = []
    for feature in features:
        mean_treated = data[data['received_email']][feature].mean()
        mean_control = data[~data['received_email']][feature].mean()
        pooled_std = np.sqrt((data[data['received_email']][feature].var() +
                             data[~data['received_email']][feature].var()) / 2)
        std_diff = (mean_treated - mean_control) / pooled_std
        std_diffs.append(std_diff)

    colors = ['red' if abs(d) > 0.1 else 'orange' if abs(d) > 0.05 else 'green' for d in std_diffs]
    bars = plt.barh(features, std_diffs, color=colors, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Threshold (0.1)')
    plt.axvline(-0.1, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Standardized Difference')
    plt.title('Covariate Imbalance (Before Matching)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 8: Propensity score by treatment status (scatter)
    plt.subplot(4, 3, 8)
    # Sample for visualization
    sample_size = min(10000, len(data))
    sample = data.sample(n=sample_size, random_state=42)

    control_sample = sample[~sample['received_email']]['propensity_score']
    treated_sample = sample[sample['received_email']]['propensity_score']

    # Create jittered strip plot
    np.random.seed(42)
    control_y = np.random.normal(0, 0.05, len(control_sample))
    treated_y = np.random.normal(1, 0.05, len(treated_sample))

    plt.scatter(control_sample, control_y, alpha=0.3, s=10, color='lightcoral', label='No Email')
    plt.scatter(treated_sample, treated_y, alpha=0.3, s=10, color='lightgreen', label='Email')

    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1], ['No Email', 'Email'])
    plt.xlabel('Propensity Score')
    plt.title(f'Distribution Comparison (n={sample_size:,})', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 9: Love plot style visualization
    plt.subplot(4, 3, 9)
    # This shows how different propensity scores are between groups
    percentiles = np.arange(5, 100, 5)
    diffs = []
    for p in percentiles:
        t_val = np.percentile(treated, p)
        c_val = np.percentile(control, p)
        diffs.append(abs(t_val - c_val))

    plt.plot(percentiles, diffs, 'o-', color='purple', linewidth=2, markersize=6)
    plt.xlabel('Percentile')
    plt.ylabel('|Difference in Propensity Score|')
    plt.title('Percentile Differences', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 10-12: Individual feature vs propensity score
    for i, feature in enumerate(features[:3]):
        plt.subplot(4, 3, 10 + i)
        sample = data.sample(n=min(5000, len(data)), random_state=42)
        control = sample[~sample['received_email']]
        treated = sample[sample['received_email']]

        plt.scatter(control[feature], control['propensity_score'],
                   alpha=0.4, s=10, color='lightcoral', label='No Email')
        plt.scatter(treated[feature], treated['propensity_score'],
                   alpha=0.4, s=10, color='lightgreen', label='Email')

        plt.xlabel(feature.replace('_', ' ').title())
        plt.ylabel('Propensity Score')
        plt.title(f'{feature.replace("_", " ").title()} vs Propensity', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/propensity_score_diagnostics.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Diagnostic plots saved to: propensity_score_diagnostics.png")


def print_model_summary(data, model, features):
    """
    Print comprehensive model summary.
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    treatment = data['received_email'].values
    propensity = data['propensity_score'].values

    # Basic statistics
    print(f"\n📊 Data Summary:")
    print(f"   Total observations: {len(data):,}")
    print(f"   Treated (email): {treatment.sum():,} ({treatment.mean():.1%})")
    print(f"   Control (no email): {(~treatment).sum():,} ({(~treatment).mean():.1%})")

    # Model performance
    auc = roc_auc_score(treatment, propensity)
    print(f"\n📈 Model Performance:")
    print(f"   AUC: {auc:.3f}")
    if auc > 0.9:
        perf = "Excellent"
    elif auc > 0.8:
        perf = "Good"
    elif auc > 0.7:
        perf = "Fair"
    else:
        perf = "Poor"
    print(f"   Performance: {perf}")

    # Propensity score summary
    print(f"\n📏 Propensity Score Summary:")
    print(f"   Min: {propensity.min():.4f}")
    print(f"   25th percentile: {np.percentile(propensity, 25):.4f}")
    print(f"   Median: {np.percentile(propensity, 50):.4f}")
    print(f"   75th percentile: {np.percentile(propensity, 75):.4f}")
    print(f"   Max: {propensity.max():.4f}")

    # By treatment group
    treated_prop = propensity[treatment == 1]
    control_prop = propensity[treatment == 0]

    print(f"\nBy Treatment Group:")
    print(f"   Treated: mean={treated_prop.mean():.4f}, std={treated_prop.std():.4f}")
    print(f"   Control: mean={control_prop.mean():.4f}, std={control_prop.std():.4f}")
    print(f"   Difference: {treated_prop.mean() - control_prop.mean():.4f}")

    # Feature importance
    print(f"\n🎯 Feature Importance (by |coefficient|):")
    feature_importance = sorted(zip(features, abs(model.coef_[0]), model.coef_[0]),
                               key=lambda x: x[1], reverse=True)
    print(f"{'Feature':<30} {'Abs Coef':<12} {'Coefficient':<15} {'Importance Rank'}")
    print("-" * 70)
    for i, (feature, abs_coef, coef) in enumerate(feature_importance, 1):
        print(f"{feature:<30} {abs_coef:<12.4f} {coef:<+15.4f} #{i}")


def main():
    """
    Run complete propensity score estimation workflow.
    """
    print("\n" + "=" * 70)
    print("PROPENSITY SCORE ESTIMATION AND DIAGNOSTICS")
    print("=" * 70)

    # Load data
    print("\nLoading simulated email campaign data...")
    data, ground_truth = load_simulated_data()
    print(f"✅ Data loaded: {data.shape}")

    # Define features for propensity model
    # These are the confounders (variables that affect both treatment and outcome)
    features = [
        'days_since_last_purchase',  # Recency
        'total_past_purchases',      # Frequency
        'avg_order_value',           # Monetary
        'customer_tenure_weeks',     # Tenure
        'rfm_score'                  # Composite RFM
    ]

    # Estimate propensity scores
    data, model, scaler = estimate_propensity_scores(data, features)

    # Check common support
    common_support_stats = check_common_support(data)

    # Identify extreme scores
    extreme_stats = identify_extreme_scores(data)

    # Create diagnostic plots
    create_diagnostic_plots(data, model, features, common_support_stats)

    # Print model summary
    print_model_summary(data, model, features)

    # Save propensity scores to file
    output_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
    data.to_csv(output_path, index=False)
    print(f"\n💾 Propensity scores saved to:")
    print(f"   {output_path}")

    # Save model parameters for reuse
    model_info = {
        'features': features,
        'model_coefficients': model.coef_[0].tolist(),
        'model_intercept': model.intercept_[0],
        'auc': roc_auc_score(data['received_email'].values, data['propensity_score'].values),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    model_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/propensity_model.json'
    with open(model_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"💾 Model parameters saved to:")
    print(f"   {model_path}")

    print("\n" + "=" * 70)
    print("✅ PROPENSITY SCORE ESTIMATION COMPLETE!")
    print("=" * 70)

    print(f"\n🎯 Next Steps:")
    print(f"   1. ✅ Propensity scores estimated and saved")
    print(f"   2. ✅ Diagnostic plots created")
    print(f"   3. ✅ Common support verified")
    print(f"   4. ✅ Ready for matching, weighting, or stratification!")

    return data, model, scaler, common_support_stats


if __name__ == "__main__":
    data, model, scaler, support_stats = main()
