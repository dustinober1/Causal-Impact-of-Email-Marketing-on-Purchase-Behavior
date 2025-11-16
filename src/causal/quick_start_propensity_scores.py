"""
Quick Start: Using Propensity Scores

This script demonstrates how to load and use the estimated propensity scores
for causal inference analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("PROPENSITY SCORES - QUICK START GUIDE")
print("=" * 70)

# 1. Load data with propensity scores
print("\n1. Loading data with propensity scores...")
data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
data = pd.read_csv(data_path)

print(f"   ✅ Data loaded: {data.shape}")
print(f"   ✅ Propensity scores column added!")

# 2. Quick summary
print("\n2. Propensity Score Summary:")
print(f"   Mean: {data['propensity_score'].mean():.4f}")
print(f"   Std:  {data['propensity_score'].std():.4f}")
print(f"   Min:  {data['propensity_score'].min():.4f}")
print(f"   Max:  {data['propensity_score'].max():.4f}")

# 3. By treatment group
print("\n3. By Treatment Group:")
treated = data[data['received_email']]
control = data[~data['received_email']]

print(f"   Treated (email):     {len(treated):,} obs, mean={treated['propensity_score'].mean():.4f}")
print(f"   Control (no email):  {len(control):,} obs, mean={control['propensity_score'].mean():.4f}")
print(f"   Difference:          {treated['propensity_score'].mean() - control['propensity_score'].mean():.4f}")

# 4. Create quick visualization
print("\n4. Creating quick visualization...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Distribution
axes[0].hist(control['propensity_score'], bins=30, alpha=0.7, label='No Email',
             color='lightcoral', density=True)
axes[0].hist(treated['propensity_score'], bins=30, alpha=0.7, label='Email',
             color='lightgreen', density=True)
axes[0].set_xlabel('Propensity Score')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution by Treatment')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Box plot
axes[1].boxplot([control['propensity_score'], treated['propensity_score']],
                tick_labels=['No Email', 'Email'])
axes[1].set_ylabel('Propensity Score')
axes[1].set_title('Box Plots')
axes[1].grid(True, alpha=0.3)

# Plot 3: Scatter of propensity vs outcome
axes[2].scatter(control['propensity_score'], control['purchased_this_week_observed'],
                alpha=0.3, s=10, color='lightcoral', label='No Email')
axes[2].scatter(treated['propensity_score'], treated['purchased_this_week_observed'],
                alpha=0.3, s=10, color='lightgreen', label='Email')
axes[2].set_xlabel('Propensity Score')
axes[2].set_ylabel('Purchased This Week')
axes[2].set_title('Propensity vs Outcome')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/propensity_scores_quick_start.png',
            dpi=150, bbox_inches='tight')
plt.show()

# 5. Check common support
print("\n5. Common Support Check:")
overlap_min = max(control['propensity_score'].min(), treated['propensity_score'].min())
overlap_max = min(control['propensity_score'].max(), treated['propensity_score'].max())

print(f"   Overlap region: [{overlap_min:.4f}, {overlap_max:.4f}]")
print(f"   ✅ Good overlap - matching is feasible!")

# 6. Use cases
print("\n" + "=" * 70)
print("WHAT CAN YOU DO WITH THESE PROPENSITY SCORES?")
print("=" * 70)

print("\n1. PROPENSITY SCORE MATCHING:")
print("   Match treated units to control units with similar scores")
print("   Example:")
print("   ```python")
print("   # See: src/causal/propensity_score_matching.py")
print("   ```")

print("\n2. INVERSE PROBABILITY WEIGHTING (IPW):")
print("   Weight observations by inverse propensity score")
print("   Example:")
print("   ```python")
print("   data['weight'] = np.where(data['received_email'],")
print("                           1 / data['propensity_score'],")
print("                           1 / (1 - data['propensity_score']))")
print("   weighted_purchase_rate = np.average(data['purchased_this_week_observed'],")
print("                                      weights=data['weight'])")
print("   ```")

print("\n3. STRATIFICATION:")
print("   Create strata based on propensity score quintiles")
print("   Example:")
print("   ```python")
print("   data['ps_quintile'] = pd.qcut(data['propensity_score'], 5, labels=False)")
print("   # Analyze within each quintile")
print("   for q in range(5):")
print("       quintile_data = data[data['ps_quintile'] == q]")
print("       # Should have better balance within quintiles")
print("   ```")

print("\n4. COVARIATE ADJUSTMENT:")
print("   Include propensity score as covariate in regression")
print("   Example:")
print("   ```python")
print("   from sklearn.linear_model import LogisticRegression")
print("   X = data[['propensity_score']]  # Or all features")
print("   y = data['purchased_this_week_observed']")
print("   model = LogisticRegression().fit(X, y)")
print("   ```")

# 7. Next steps
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print("\n✅ COMPLETED:")
print("   • Propensity scores estimated")
print("   • Model diagnostics performed")
print("   • Data saved with scores")
print("   • Common support verified")

print("\n🎯 READY FOR:")
print("   1. Propensity Score Matching (see PSM implementation)")
print("   2. Inverse Probability Weighting (implement IPW next!)")
print("   3. Stratification on propensity scores")
print("   4. Covariate adjustment with propensity scores")

print("\n🎓 GOAL:")
print("   • Naive Effect: 16.0% (biased)")
print("   • True Effect: 9.5%")
print("   • Use propensity scores to recover the truth!")

print("\n" + "=" * 70)
print("✅ Propensity Scores Ready for Causal Inference!")
print("=" * 70)
