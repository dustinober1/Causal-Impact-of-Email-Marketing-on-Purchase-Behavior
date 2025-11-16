"""
Propensity Score Summary Visualization

Quick visualization of key propensity score insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data with propensity scores
data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
data = pd.read_csv(data_path)

# Create summary visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Distribution by treatment group
ax1 = axes[0, 0]
treated = data[data['received_email']]['propensity_score']
control = data[~data['received_email']]['propensity_score']

ax1.hist(control, bins=50, alpha=0.7, label='No Email (Control)', color='lightcoral', density=True)
ax1.hist(treated, bins=50, alpha=0.7, label='Received Email (Treated)', color='lightgreen', density=True)
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Propensity Score Distribution by Treatment', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Box plots
ax2 = axes[0, 1]
plot_data = pd.DataFrame({
    'propensity_score': np.concatenate([treated, control]),
    'treatment': ['No Email'] * len(control) + ['Email'] * len(treated)
})

bp = ax2.boxplot([control, treated], labels=['No Email', 'Email'],
                 patch_artist=True, notch=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax2.set_ylabel('Propensity Score')
ax2.set_title('Propensity Score Box Plots', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Feature importance
ax3 = axes[1, 0]
features = ['days_since_last_purchase', 'total_past_purchases', 'avg_order_value',
            'customer_tenure_weeks', 'rfm_score']
coefficients = [-0.4219, 0.1210, 0.0021, 0.0197, 0.0181]

colors = ['red' if c < 0 else 'green' for c in coefficients]
bars = ax3.barh(features, coefficients, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Coefficient Value')
ax3.set_title('Logistic Regression Coefficients', fontweight='bold')
ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
ax3.grid(True, alpha=0.3)

# Add value labels
for i, (bar, coef) in enumerate(zip(bars, coefficients)):
    ax3.text(coef + (0.01 if coef >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
             f'{coef:+.3f}', ha='left' if coef >= 0 else 'right', va='center', fontweight='bold')

# Plot 4: Score summary statistics
ax4 = axes[1, 1]
stats = {
    'Min': data['propensity_score'].min(),
    '25th': np.percentile(data['propensity_score'], 25),
    'Median': data['propensity_score'].median(),
    '75th': np.percentile(data['propensity_score'], 75),
    'Max': data['propensity_score'].max()
}

treated_stats = {
    'Min': treated.min(),
    '25th': np.percentile(treated, 25),
    'Median': treated.median(),
    '75th': np.percentile(treated, 75),
    'Max': treated.max()
}

control_stats = {
    'Min': control.min(),
    '25th': np.percentile(control, 25),
    'Median': control.median(),
    '75th': np.percentile(control, 75),
    'Max': control.max()
}

x = np.arange(len(stats))
width = 0.25

ax4.bar(x - width, [stats[k] for k in stats.keys()], width,
        label='Overall', color='lightblue', alpha=0.8, edgecolor='black')
ax4.bar(x, [treated_stats[k] for k in stats.keys()], width,
        label='Treated (Email)', color='lightgreen', alpha=0.8, edgecolor='black')
ax4.bar(x + width, [control_stats[k] for k in stats.keys()], width,
        label='Control (No Email)', color='lightcoral', alpha=0.8, edgecolor='black')

ax4.set_xlabel('Percentile')
ax4.set_ylabel('Propensity Score')
ax4.set_title('Propensity Score Summary Statistics', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(stats.keys())
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/propensity_score_summary.png',
            dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "=" * 70)
print("PROPENSITY SCORE SUMMARY")
print("=" * 70)

print(f"\n📊 Overall Statistics:")
print(f"   Mean: {data['propensity_score'].mean():.4f}")
print(f"   Std:  {data['propensity_score'].std():.4f}")
print(f"   Min:  {data['propensity_score'].min():.4f}")
print(f"   Max:  {data['propensity_score'].max():.4f}")

print(f"\n📧 Treated Group (received email):")
print(f"   Mean: {treated.mean():.4f}")
print(f"   Std:  {treated.std():.4f}")
print(f"   Min:  {treated.min():.4f}")
print(f"   Max:  {treated.max():.4f}")

print(f"\n🚫 Control Group (no email):")
print(f"   Mean: {control.mean():.4f}")
print(f"   Std:  {control.std():.4f}")
print(f"   Min:  {control.min():.4f}")
print(f"   Max:  {control.max():.4f}")

print(f"\n🎯 Difference:")
print(f"   Mean difference: {treated.mean() - control.mean():.4f}")

print("\n" + "=" * 70)
print("✅ Summary plot saved: propensity_score_summary.png")
print("=" * 70)
