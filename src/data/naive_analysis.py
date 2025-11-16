"""
Naive Analysis: Demonstrating Why Simple Comparisons Fail

This script shows the problem with naive comparisons when there's confounding.
We compare email recipients to non-recipients and show the bias.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from pathlib import Path

# Try to import seaborn, fall back to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib-only plots")


def load_data():
    """Load the simulated email campaign data."""
    # Get the project root directory (go up 3 levels: src/data/__file__ -> project_root)
    project_root = Path(__file__).parent.parent.parent

    # Construct paths to data files
    data_dir = project_root / 'data' / 'processed'
    sim_data_path = data_dir / 'simulated_email_campaigns.csv'
    ground_truth_path = data_dir / 'ground_truth.json'

    # Verify files exist
    if not sim_data_path.exists():
        raise FileNotFoundError(f"Simulated data not found at {sim_data_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth not found at {ground_truth_path}")

    sim_data = pd.read_csv(sim_data_path)

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    return sim_data, ground_truth


def calculate_naive_effect(sim_data):
    """
    Calculate the naive effect estimate.

    Returns:
    --------
    dict: Naive comparison results
    """
    # Split into groups
    email_group = sim_data[sim_data['received_email']]
    no_email_group = sim_data[~sim_data['received_email']]

    # Calculate purchase rates
    purchase_rate_email = email_group['purchased_this_week_observed'].mean()
    purchase_rate_no_email = no_email_group['purchased_this_week_observed'].mean()

    # Naive effect
    naive_effect = purchase_rate_email - purchase_rate_no_email

    return {
        'email_purchase_rate': purchase_rate_email,
        'no_email_purchase_rate': purchase_rate_no_email,
        'naive_effect': naive_effect,
        'email_n': len(email_group),
        'no_email_n': len(no_email_group)
    }


def create_naive_comparison_plot(naive_results):
    """
    Create visualization of naive comparison.
    """
    plt.figure(figsize=(12, 8))

    # Plot 1: Purchase rates
    plt.subplot(2, 2, 1)
    groups = ['No Email', 'Received Email']
    rates = [
        naive_results['no_email_purchase_rate'] * 100,
        naive_results['email_purchase_rate'] * 100
    ]
    colors = ['lightcoral', 'lightgreen']

    bars = plt.bar(groups, rates, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Naive Comparison: Purchase Rates', fontweight='bold', fontsize=14)
    plt.ylabel('Purchase Rate (%)')
    plt.ylim(0, max(rates) * 1.3)

    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Plot 2: Sample sizes
    plt.subplot(2, 2, 2)
    sizes = [naive_results['no_email_n'], naive_results['email_n']]
    plt.bar(groups, sizes, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Sample Sizes', fontweight='bold', fontsize=14)
    plt.ylabel('Number of Observations')

    for i, (group, size) in enumerate(zip(groups, sizes)):
        plt.text(i, size + max(sizes)*0.02, f'{size:,}',
                 ha='center', va='bottom', fontweight='bold')

    # Plot 3: Effect size
    plt.subplot(2, 2, 3)
    plt.barh(['Observed Effect'], [naive_results['naive_effect'] * 100],
             color='gold', edgecolor='black', linewidth=1.5)
    plt.title('Naive Effect Size', fontweight='bold', fontsize=14)
    plt.xlabel('Effect Size (Percentage Points)')
    plt.text(naive_results['naive_effect']*100/2, 0, f'{naive_results["naive_effect"]:.1%}',
             ha='center', va='center', fontweight='bold', fontsize=12)

    # Plot 4: Direct comparison
    plt.subplot(2, 2, 4)
    x = np.arange(2)
    width = 0.35

    plt.bar(x - width/2, [naive_results['no_email_purchase_rate'] * 100,
                          naive_results['email_purchase_rate'] * 100],
            width, label='Purchase Rate', color=['lightcoral', 'lightgreen'],
            edgecolor='black', linewidth=1.5)

    plt.title('Direct Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Purchase Rate (%)')
    plt.xticks(x, ['No Email', 'Email'])
    plt.legend()

    plt.tight_layout()
    plt.show()


def analyze_covariate_imbalance(sim_data):
    """
    Analyze imbalance between treatment and control groups.
    """
    features_to_compare = [
        'rfm_score',
        'days_since_last_purchase',
        'total_past_purchases',
        'avg_order_value',
        'customer_tenure_weeks'
    ]

    # Split into groups
    email_group = sim_data[sim_data['received_email']]
    no_email_group = sim_data[~sim_data['received_email']]

    # Create imbalance table
    imbalance_results = []

    for feature in features_to_compare:
        treated = email_group[feature]
        control = no_email_group[feature]

        mean_treated = treated.mean()
        mean_control = control.mean()

        std_treated = treated.std()
        std_control = control.std()

        # Standardized difference
        pooled_std = np.sqrt((std_treated**2 + std_control**2) / 2)
        std_diff = (mean_treated - mean_control) / pooled_std if pooled_std > 0 else 0

        # P-value
        t_stat, p_value = stats.ttest_ind(treated, control)

        imbalance_results.append({
            'Feature': feature,
            'Mean (No Email)': mean_control,
            'Mean (Email)': mean_treated,
            'Difference': mean_treated - mean_control,
            'Std. Diff.': std_diff,
            'P-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

    return pd.DataFrame(imbalance_results)


def create_confounding_visualizations(sim_data):
    """
    Create comprehensive confounding visualizations.
    """
    features_to_compare = [
        'rfm_score',
        'days_since_last_purchase',
        'total_past_purchases',
        'avg_order_value',
        'customer_tenure_weeks'
    ]

    plt.figure(figsize=(16, 12))

    # Plot 1: RFM Score
    plt.subplot(3, 3, 1)
    # Create boxplot manually
    email_vals = sim_data[sim_data['received_email']]['rfm_score'].values
    no_email_vals = sim_data[~sim_data['received_email']]['rfm_score'].values
    
    # Create boxplot
    parts = plt.violinplot([no_email_vals, email_vals], positions=[0, 1])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('lightcoral' if i == 0 else 'lightgreen')
        pc.set_alpha(0.7)
    plt.title('RFM Score by Email Status', fontweight='bold')
    plt.xlabel('Received Email')

    no_email_rfm = sim_data[~sim_data['received_email']]['rfm_score'].mean()
    email_rfm = sim_data[sim_data['received_email']]['rfm_score'].mean()
    plt.text(0.5, plt.ylim()[1]*0.9, f'Email: {email_rfm:.2f}\nNo Email: {no_email_rfm:.2f}',
             ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Days since last purchase (capped for visualization)
    plt.subplot(3, 3, 2)
    # Create boxplot manually for days
    email_days = np.minimum(sim_data[sim_data['received_email']]['days_since_last_purchase'], 100).values
    no_email_days = np.minimum(sim_data[~sim_data['received_email']]['days_since_last_purchase'], 100).values
    
    parts = plt.violinplot([no_email_days, email_days], positions=[0, 1])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('lightcoral' if i == 0 else 'lightgreen')
        pc.set_alpha(0.7)
    plt.title('Days Since Last Purchase', fontweight='bold')
    plt.xlabel('Received Email')
    plt.ylabel('Days (capped at 100)')

    # Plot 3: Total past purchases
    plt.subplot(3, 3, 3)
    email_purchases = sim_data[sim_data['received_email']]['total_past_purchases'].values
    no_email_purchases = sim_data[~sim_data['received_email']]['total_past_purchases'].values

    parts = plt.violinplot([no_email_purchases, email_purchases], positions=[0, 1])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('lightcoral' if i == 0 else 'lightgreen')
        pc.set_alpha(0.7)
    plt.title('Total Past Purchases', fontweight='bold')
    plt.xlabel('Received Email')

    # Plot 4: Average order value (capped)
    plt.subplot(3, 3, 4)
    email_aov = sim_data[(sim_data['received_email']) & (sim_data['avg_order_value'] < 100)]['avg_order_value'].values
    no_email_aov = sim_data[(~sim_data['received_email']) & (sim_data['avg_order_value'] < 100)]['avg_order_value'].values

    parts = plt.violinplot([no_email_aov, email_aov], positions=[0, 1])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('lightcoral' if i == 0 else 'lightgreen')
        pc.set_alpha(0.7)
    plt.title('Average Order Value', fontweight='bold')
    plt.xlabel('Received Email')
    plt.ylabel('AOV (capped at £100)')

    # Plot 5: Customer tenure
    plt.subplot(3, 3, 5)
    email_tenure = sim_data[sim_data['received_email']]['customer_tenure_weeks'].values
    no_email_tenure = sim_data[~sim_data['received_email']]['customer_tenure_weeks'].values

    parts = plt.violinplot([no_email_tenure, email_tenure], positions=[0, 1])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('lightcoral' if i == 0 else 'lightgreen')
        pc.set_alpha(0.7)
    plt.title('Customer Tenure (weeks)', fontweight='bold')
    plt.xlabel('Received Email')

    # Plot 6: Correlations
    plt.subplot(3, 3, 6)
    correlations = []
    for feature in features_to_compare:
        corr = sim_data['received_email'].corr(sim_data[feature])
        correlations.append(corr)

    colors = ['red' if c < 0 else 'green' for c in correlations]
    bars = plt.barh(features_to_compare, correlations, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Correlation with Email Receipt', fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)

    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        plt.text(corr + (0.01 if corr >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                 f'{corr:.3f}', ha='left' if corr >= 0 else 'right', va='center', fontweight='bold')

    # Plot 7: Standardized differences
    plt.subplot(3, 3, 7)
    std_diffs = []
    for feature in features_to_compare:
        mean_treated = sim_data[sim_data['received_email']][feature].mean()
        mean_control = sim_data[~sim_data['received_email']][feature].mean()
        pooled_std = np.sqrt((sim_data[sim_data['received_email']][feature].var() +
                             sim_data[~sim_data['received_email']][feature].var()) / 2)
        std_diff = (mean_treated - mean_control) / pooled_std if pooled_std > 0 else 0
        std_diffs.append(std_diff)

    colors = ['red' if abs(d) > 0.1 else 'orange' if abs(d) > 0.05 else 'green' for d in std_diffs]
    bars = plt.barh(features_to_compare, std_diffs, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Standardized Differences', fontweight='bold')
    plt.xlabel('Standardized Difference')
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Threshold (0.1)')
    plt.axvline(-0.1, color='red', linestyle='--', alpha=0.5)

    for i, (bar, diff) in enumerate(zip(bars, std_diffs)):
        plt.text(diff + (0.01 if diff >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                 f'{diff:.3f}', ha='left' if diff >= 0 else 'right', va='center', fontweight='bold')

    plt.legend()

    # Plot 8: RFM distribution
    plt.subplot(3, 3, 8)
    no_email_rfm_dist = sim_data[~sim_data['received_email']]['rfm_score']
    email_rfm_dist = sim_data[sim_data['received_email']]['rfm_score']
    plt.hist(no_email_rfm_dist, bins=15, alpha=0.7, label='No Email',
             color='lightcoral', edgecolor='black')
    plt.hist(email_rfm_dist, bins=15, alpha=0.7, label='Email',
             color='lightgreen', edgecolor='black')
    plt.title('RFM Score Distribution', fontweight='bold')
    plt.xlabel('RFM Score')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 9: Days since purchase distribution
    plt.subplot(3, 3, 9)
    no_email_recency = sim_data[~sim_data['received_email']]['days_since_last_purchase']
    email_recency = sim_data[sim_data['received_email']]['days_since_last_purchase']
    no_email_recency = np.minimum(no_email_recency, 100)
    email_recency = np.minimum(email_recency, 100)

    plt.hist(no_email_recency, bins=20, alpha=0.7, label='No Email',
             color='lightcoral', edgecolor='black')
    plt.hist(email_recency, bins=20, alpha=0.7, label='Email',
             color='lightgreen', edgecolor='black')
    plt.title('Days Since Purchase Distribution', fontweight='bold')
    plt.xlabel('Days (capped at 100)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


def compare_naive_to_true(sim_data, ground_truth, naive_results):
    """
    Compare naive estimate to true causal effect.
    """
    # Calculate true effect
    true_effect = sim_data['individual_treatment_effect'].mean()
    expected_effect = ground_truth['base_email_effect']

    # Print comparison
    print("\n" + "="*70)
    print("NAIVE vs TRUE EFFECT COMPARISON")
    print("="*70)

    print(f"\nNaive (Observed): {naive_results['naive_effect']:.1%}")
    print(f"True (Causal):    {true_effect:.1%}")
    print(f"Expected (GT):    {expected_effect:.1%}")

    bias = naive_results['naive_effect'] - true_effect
    relative_bias = (bias / true_effect) * 100

    print(f"\nBias: {bias:.1%} ({relative_bias:.0f}% overestimate)")

    # Visualize
    plt.figure(figsize=(14, 6))

    # Plot 1: Side by side comparison
    plt.subplot(1, 2, 1)
    effects = ['Naive\n(Observed)', 'True\n(Causal)']
    effect_values = [naive_results['naive_effect'] * 100, true_effect * 100]
    colors = ['lightcoral', 'lightgreen']

    bars = plt.bar(effects, effect_values, color=colors, edgecolor='black', linewidth=2, width=0.6)
    plt.title('Naive vs True Effect', fontweight='bold', fontsize=14)
    plt.ylabel('Effect Size (Percentage Points)')

    for i, (bar, val) in enumerate(zip(bars, effect_values)):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.ylim(0, max(effect_values) * 1.2)

    # Add bias annotation
    plt.annotate('', xy=(1.2, true_effect * 100), xytext=(0.8, naive_results['naive_effect'] * 100),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    plt.text(1, (naive_results['naive_effect'] + true_effect) * 50,
             f'Bias:\n{bias:.1%}',
             ha='center', va='center', color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))

    # Plot 2: Bias decomposition
    plt.subplot(1, 2, 2)
    baseline_diff = (sim_data[sim_data['received_email']]['purchased_this_week_observed'].mean() -
                    sim_data[~sim_data['received_email']]['purchased_this_week_observed'].mean() -
                    true_effect)

    components = ['Baseline\nDifference', 'True Email\nEffect', 'Naive\nEstimate']
    values = [baseline_diff * 100, true_effect * 100, naive_results['naive_effect'] * 100]
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    bars = plt.bar(components, values, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Bias Decomposition', fontweight='bold', fontsize=14)
    plt.ylabel('Percentage Points')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 val + (0.5 if val >= 0 else -0.5),
                 f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                 fontweight='bold')

    plt.tight_layout()
    plt.show()

    return {
        'true_effect': true_effect,
        'bias': bias,
        'relative_bias': relative_bias
    }


def print_summary(sim_data, naive_results, bias_analysis):
    """
    Print comprehensive summary.
    """
    print("\n" + "="*70)
    print("SUMMARY: WHY NAIVE ANALYSIS FAILS")
    print("="*70)

    print(f"\n1. NAIVE EFFECT: {naive_results['naive_effect']:.1%}")
    print(f"   - Purchase rate (email): {naive_results['email_purchase_rate']:.1%}")
    print(f"   - Purchase rate (no email): {naive_results['no_email_purchase_rate']:.1%}")

    print(f"\n2. TRUE EFFECT: {bias_analysis['true_effect']:.1%}")
    print(f"   - This is the actual causal effect of emails")

    print(f"\n3. BIAS: {bias_analysis['bias']:.1%}")
    print(f"   - Naive overestimates by {bias_analysis['relative_bias']:.0f}%")
    print(f"   - This is CONFOUNDING BIAS!")

    print(f"\n4. WHY?")
    email_group = sim_data[sim_data['received_email']]
    no_email_group = sim_data[~sim_data['received_email']]

    print(f"   - Email recipients have HIGHER baseline purchase probability")
    print(f"   - RFM score: {email_group['rfm_score'].mean():.2f} vs {no_email_group['rfm_score'].mean():.2f}")
    print(f"   - Days since last purchase: {email_group['days_since_last_purchase'].mean():.1f} vs {no_email_group['days_since_last_purchase'].mean():.1f}")

    print(f"\n5. SOLUTION:")
    print(f"   - Need causal inference methods to correct for confounding")
    print(f"   - Match, weight, or adjust for customer characteristics")
    print(f"   - Recover the true {bias_analysis['true_effect']:.1%} effect")


def main():
    """
    Run complete naive analysis.
    """
    print("="*70)
    print("NAIVE ANALYSIS: DEMONSTRATING CONFOUNDING BIAS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    sim_data, ground_truth = load_data()
    print(f"Data loaded: {sim_data.shape}")

    # Calculate naive effect
    print("\n" + "="*70)
    print("STEP 1: NAIVE COMPARISON")
    print("="*70)

    naive_results = calculate_naive_effect(sim_data)

    print(f"\nEmail Group:")
    print(f"  Sample size: {naive_results['email_n']:,} ({naive_results['email_n']/len(sim_data):.1%})")
    print(f"  Purchase rate: {naive_results['email_purchase_rate']:.1%}")

    print(f"\nNo Email Group:")
    print(f"  Sample size: {naive_results['no_email_n']:,} ({naive_results['no_email_n']/len(sim_data):.1%})")
    print(f"  Purchase rate: {naive_results['no_email_purchase_rate']:.1%}")

    print(f"\nNaive Effect Estimate: {naive_results['naive_effect']:.1%}")

    # Visualize naive comparison
    create_naive_comparison_plot(naive_results)

    # Analyze confounding
    print("\n" + "="*70)
    print("STEP 2: COVARIATE IMBALANCE")
    print("="*70)

    imbalance_df = analyze_covariate_imbalance(sim_data)

    print("\nCovariate Imbalance Table:")
    display_df = imbalance_df.copy()
    for col in ['Mean (No Email)', 'Mean (Email)', 'Difference']:
        display_df[col] = display_df[col].round(2)
    display_df['Std. Diff.'] = display_df['Std. Diff.'].round(3)
    display_df['P-value'] = display_df['P-value'].apply(
        lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}"
    )

    print(display_df.to_string(index=False))

    # Visualize confounding
    create_confounding_visualizations(sim_data)

    # Compare to true effect
    print("\n" + "="*70)
    print("STEP 3: NAIVE vs TRUE")
    print("="*70)

    bias_analysis = compare_naive_to_true(sim_data, ground_truth, naive_results)

    # Print summary
    print_summary(sim_data, naive_results, bias_analysis)

    print("\n" + "="*70)
    print("CONCLUSION: NAIVE ANALYSIS IS BIASED!")
    print("="*70)
    print("\nWe need causal inference methods to recover the true effect.")


if __name__ == "__main__":
    main()
