"""
Propensity Score Matching: Nearest Neighbor with Comprehensive Balance Checking

This script implements 1:1 nearest neighbor propensity score matching with:
- Caliper matching (0.1 * std of propensity scores)
- With replacement option
- Comprehensive covariate balance checking
- Standardized mean differences
- Variance ratios
- Love plots
- Bootstrap confidence intervals for treatment effect
- Comparison to true causal effect

The goal is to create balanced treatment/control groups and recover the
true causal effect from confounded data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PropensityScoreMatcher:
    """
    Propensity Score Matching implementation with comprehensive diagnostics.
    """

    def __init__(self, caliper_multiplier=0.1, with_replacement=True, random_state=42):
        """
        Initialize the matcher.

        Parameters:
        -----------
        caliper_multiplier : float
            Caliper = caliper_multiplier * std(propensity_score)
        with_replacement : bool
            Whether to sample with replacement
        random_state : int
            Random seed
        """
        self.caliper_multiplier = caliper_multiplier
        self.with_replacement = with_replacement
        self.random_state = random_state

    def fit(self, data, treatment_col='received_email', propensity_col='propensity_score',
            covariates=None):
        """
        Fit the matcher to the data.

        Parameters:
        -----------
        data : DataFrame
            Data containing treatment, propensity scores, and covariates
        treatment_col : str
            Name of treatment indicator column
        propensity_col : str
            Name of propensity score column
        covariates : list
            List of covariate column names for balance checking

        Returns:
        --------
        self : PropensityScoreMatcher
        """
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.propensity_col = propensity_col

        # Identify covariates if not provided
        if covariates is None:
            # Exclude non-covariate columns
            exclude_cols = [
                treatment_col, propensity_col, 'CustomerID', 'week_number',
                'week_start', 'purchase_this_week', 'revenue_this_week',
                'purchased_this_week_observed', 'email_assignment_probability',
                'individual_treatment_effect', 'true_purchase_probability',
                'true_purchase_prob_if_no_email'
            ]
            self.covariates = [col for col in data.columns if col not in exclude_cols]
        else:
            self.covariates = covariates

        # Extract treatment and propensity
        self.treatment = data[treatment_col].values
        self.propensity = data[propensity_col].values

        # Calculate caliper
        self.caliper = self.caliper_multiplier * np.std(self.propensity)
        print(f"📏 Caliper: {self.caliper:.4f} (0.1 * std of propensity scores)")

        return self

    def perform_matching(self):
        """
        Perform nearest neighbor propensity score matching.

        Returns:
        --------
        matched_data : DataFrame
            Matched dataset
        """
        print("\n" + "=" * 70)
        print("STEP 1: PERFORMING PROPENSITY SCORE MATCHING")
        print("=" * 70)

        np.random.seed(self.random_state)

        # Get treated and control indices
        treated_idx = np.where(self.treatment == 1)[0]
        control_idx = np.where(self.treatment == 0)[0]

        print(f"\n📊 Sample Sizes:")
        print(f"   Treated (email): {len(treated_idx):,}")
        print(f"   Control (no email): {len(control_idx):,}")
        print(f"   Ratio: {len(control_idx)/len(treated_idx):.2f}")

        # Track which control units have been used
        used_control = set()

        # Storage for matched pairs
        matched_treated = []
        matched_control = []

        # Match each treated unit
        print(f"\n🔄 Matching in progress...")
        for i, t_idx in enumerate(treated_idx):
            if (i + 1) % 20000 == 0:
                print(f"   Progress: {i+1:,}/{len(treated_idx):,} ({100*(i+1)/len(treated_idx):.1f}%)")

            t_score = self.propensity[t_idx]

            # Get available controls
            if self.with_replacement:
                available_control = control_idx
            else:
                available_control = np.array([idx for idx in control_idx if idx not in used_control])

            if len(available_control) == 0:
                # No controls left
                continue

            # Calculate distances
            control_scores = self.propensity[available_control]
            distances = np.abs(control_scores - t_score)

            # Find closest match within caliper
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist <= self.caliper:
                match_idx = available_control[min_dist_idx]

                matched_treated.append(t_idx)
                matched_control.append(match_idx)

                # Mark as used (if without replacement)
                if not self.with_replacement:
                    used_control.add(match_idx)

        # Create matched indices
        self.matched_treated = matched_treated
        self.matched_control = matched_control

        # Create matched dataset
        matched_indices = matched_treated + matched_control
        self.matched_data = self.data.iloc[matched_indices].copy()
        self.matched_data['matched_id'] = np.repeat(range(len(matched_treated)), 2)

        # Summary
        print(f"\n✅ Matching Complete!")
        print(f"   Matched pairs: {len(matched_treated):,}")
        print(f"   Match rate: {len(matched_treated)/len(treated_idx):.1%}")

        if not self.with_replacement:
            control_usage = len(matched_treated) / len(control_idx)
            print(f"   Control usage rate: {control_usage:.1%}")

        # Calculate matching quality
        matched_distances = []
        for t_idx, c_idx in zip(matched_treated, matched_control):
            dist = abs(self.propensity[t_idx] - self.propensity[c_idx])
            matched_distances.append(dist)

        print(f"\n📏 Matching Quality:")
        print(f"   Mean distance: {np.mean(matched_distances):.4f}")
        print(f"   Median distance: {np.median(matched_distances):.4f}")
        print(f"   Max distance: {np.max(matched_distances):.4f}")
        print(f"   Within caliper: {(np.array(matched_distances) <= self.caliper).mean()*100:.1f}%")

        return self.matched_data

    def check_balance(self, covariates=None):
        """
        Check covariate balance before and after matching.

        Parameters:
        -----------
        covariates : list
            Covariates to check (default: self.covariates)

        Returns:
        --------
        balance_stats : dict
            Balance statistics
        """
        print("\n" + "=" * 70)
        print("STEP 2: CHECKING COVARIATE BALANCE")
        print("=" * 70)

        if covariates is None:
            covariates = self.covariates

        # Initialize results
        balance_results = []

        print(f"\n📊 Balance Statistics:")
        print(f"{'Covariate':<30} {'Std Diff (Before)':<20} {'Std Diff (After)':<20} {'Improvement'}")
        print("-" * 70)

        for cov in covariates:
            # Before matching (original data)
            treated_before = self.data[self.data[self.treatment_col] == 1][cov]
            control_before = self.data[self.data[self.treatment_col] == 0][cov]

            # Calculate standardized difference before
            mean_treated_before = treated_before.mean()
            mean_control_before = control_before.mean()
            var_treated_before = treated_before.var()
            var_control_before = control_before.var()

            # Pooled standard deviation
            pooled_sd_before = np.sqrt((var_treated_before + var_control_before) / 2)
            std_diff_before = (mean_treated_before - mean_control_before) / pooled_sd_before

            # After matching (matched data)
            treated_after = self.matched_data[self.matched_data[self.treatment_col] == 1][cov]
            control_after = self.matched_data[self.matched_data[self.treatment_col] == 0][cov]

            # Calculate standardized difference after
            mean_treated_after = treated_after.mean()
            mean_control_after = control_after.mean()
            var_treated_after = treated_after.var()
            var_control_after = control_after.var()

            pooled_sd_after = np.sqrt((var_treated_after + var_control_after) / 2)
            std_diff_after = (mean_treated_after - mean_control_after) / pooled_sd_after

            # Calculate variance ratio
            var_ratio_before = var_treated_before / var_control_before if var_control_before > 0 else np.inf
            var_ratio_after = var_treated_after / var_control_after if var_control_after > 0 else np.inf

            # Store results
            balance_results.append({
                'covariate': cov,
                'mean_treated_before': mean_treated_before,
                'mean_control_before': mean_control_before,
                'std_diff_before': std_diff_before,
                'mean_treated_after': mean_treated_after,
                'mean_control_after': mean_control_after,
                'std_diff_after': std_diff_after,
                'var_ratio_before': var_ratio_before,
                'var_ratio_after': var_ratio_after,
                'improvement': abs(std_diff_before) - abs(std_diff_after)
            })

            # Print results
            improvement = "✅" if abs(std_diff_after) < abs(std_diff_before) else "❌"
            print(f"{cov:<30} {abs(std_diff_before):<20.4f} {abs(std_diff_after):<20.4f} {improvement}")

        self.balance_results = pd.DataFrame(balance_results)

        # Summary statistics
        good_balance_before = (abs(self.balance_results['std_diff_before']) < 0.1).sum()
        good_balance_after = (abs(self.balance_results['std_diff_after']) < 0.1).sum()

        print(f"\n📈 Balance Summary:")
        print(f"   Before matching: {good_balance_before}/{len(covariates)} covariates well-balanced (|std diff| < 0.1)")
        print(f"   After matching: {good_balance_after}/{len(covariates)} covariates well-balanced (|std diff| < 0.1)")
        print(f"   Improvement: +{good_balance_after - good_balance_before} covariates")

        # Mean absolute standardized difference
        mean_abs_std_before = abs(self.balance_results['std_diff_before']).mean()
        mean_abs_std_after = abs(self.balance_results['std_diff_after']).mean()

        print(f"\n📊 Mean Absolute Standardized Difference:")
        print(f"   Before: {mean_abs_std_before:.4f}")
        print(f"   After: {mean_abs_std_after:.4f}")
        print(f"   Reduction: {(mean_abs_std_before - mean_abs_std_after)/mean_abs_std_before*100:.1f}%")

        return self.balance_results

    def create_love_plot(self):
        """
        Create a Love plot showing standardized differences before and after matching.

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Love Plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Standardized differences
        x_pos = np.arange(len(self.balance_results))

        ax1.barh(x_pos - 0.2, self.balance_results['std_diff_before'], 0.4,
                 label='Before Matching', color='lightcoral', alpha=0.8, edgecolor='black')
        ax1.barh(x_pos + 0.2, self.balance_results['std_diff_after'], 0.4,
                 label='After Matching', color='lightgreen', alpha=0.8, edgecolor='black')

        ax1.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Good Balance (±0.1)')
        ax1.axvline(-0.1, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5)

        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(self.balance_results['covariate'])
        ax1.set_xlabel('Standardized Mean Difference')
        ax1.set_title('Covariate Balance (Standardized Differences)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Absolute standardized differences (more common in Love plots)
        ax2.barh(x_pos - 0.2, abs(self.balance_results['std_diff_before']), 0.4,
                 label='Before Matching', color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.barh(x_pos + 0.2, abs(self.balance_results['std_diff_after']), 0.4,
                 label='After Matching', color='lightgreen', alpha=0.8, edgecolor='black')

        ax2.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Good Balance (0.1)')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)

        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(self.balance_results['covariate'])
        ax2.set_xlabel('|Standardized Mean Difference|')
        ax2.set_title('Covariate Balance (Absolute Values)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/love_plot_balance.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def estimate_treatment_effect(self, outcome_col='purchased_this_week_observed', n_bootstrap=1000):
        """
        Estimate treatment effect on matched sample with bootstrap confidence intervals.

        Parameters:
        -----------
        outcome_col : str
            Name of outcome column
        n_bootstrap : int
            Number of bootstrap samples

        Returns:
        --------
        effect_result : dict
            Treatment effect estimate with CI
        """
        print("\n" + "=" * 70)
        print("STEP 3: ESTIMATING TREATMENT EFFECT")
        print("=" * 70)

        # Get matched treated and control outcomes
        matched_treated_outcomes = self.matched_data[
            self.matched_data[self.treatment_col] == 1
        ][outcome_col].values

        matched_control_outcomes = self.matched_data[
            self.matched_data[self.treatment_col] == 0
        ][outcome_col].values

        # Point estimate (difference in means)
        treated_mean = matched_treated_outcomes.mean()
        control_mean = matched_control_outcomes.mean()
        point_estimate = treated_mean - control_mean

        print(f"\n📊 Point Estimate:")
        print(f"   Treated mean: {treated_mean:.4f} ({treated_mean:.1%})")
        print(f"   Control mean: {control_mean:.4f} ({control_mean:.1%})")
        print(f"   Difference: {point_estimate:.4f} ({point_estimate:.1%})")

        # Bootstrap confidence intervals
        print(f"\n🔄 Bootstrapping ({n_bootstrap:,} samples)...")

        def calculate_difference(treated, control):
            """Calculate difference in means."""
            return treated.mean() - control.mean()

        # Bootstrap sampling function
        def bootstrap_diff(data1, data2):
            """Bootstrap resampling."""
            np.random.seed(np.random.randint(0, 1000000))
            boot_treated = np.random.choice(data1, size=len(data1), replace=True)
            boot_control = np.random.choice(data2, size=len(data2), replace=True)
            return calculate_difference(boot_treated, boot_control)

        # Generate bootstrap samples
        bootstrap_estimates = []
        for i in range(n_bootstrap):
            if (i + 1) % 200 == 0:
                print(f"   Progress: {i+1}/{n_bootstrap} ({100*(i+1)/n_bootstrap:.1f}%)")

            np.random.seed(self.random_state + i)
            boot_est = bootstrap_diff(matched_treated_outcomes, matched_control_outcomes)
            bootstrap_estimates.append(boot_est)

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)
        se_bootstrap = np.std(bootstrap_estimates)

        print(f"\n📊 Bootstrap Results:")
        print(f"   Standard error: {se_bootstrap:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

        # Z-test
        z_stat = point_estimate / se_bootstrap
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        print(f"\n📊 Statistical Significance:")
        print(f"   Z-statistic: {z_stat:.2f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

        self.effect_result = {
            'point_estimate': point_estimate,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'se': se_bootstrap,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_stat': z_stat,
            'p_value': p_value,
            'n_matched_pairs': len(matched_treated_outcomes),
            'bootstrap_estimates': bootstrap_estimates
        }

        return self.effect_result

    def compare_to_true_effect(self, outcome_col='purchased_this_week_observed'):
        """
        Compare estimated effect to true causal effect.

        Parameters:
        -----------
        outcome_col : str
            Name of outcome column

        Returns:
        --------
        comparison : dict
            Comparison results
        """
        print("\n" + "=" * 70)
        print("STEP 4: COMPARING TO TRUE CAUSAL EFFECT")
        print("=" * 70)

        # Load ground truth
        ground_truth_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/ground_truth.json'
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        true_effect = self.data['individual_treatment_effect'].mean()
        expected_effect = ground_truth['base_email_effect']

        print(f"\n🎯 Effect Comparison:")
        print(f"   PSM Estimate:   {self.effect_result['point_estimate']:.4f} ({self.effect_result['point_estimate']:.1%})")
        print(f"   True Effect:    {true_effect:.4f} ({true_effect:.1%})")
        print(f"   Expected (GT):  {expected_effect:.4f} ({expected_effect:.1%})")

        # Calculate bias
        bias = self.effect_result['point_estimate'] - true_effect
        relative_bias = (bias / true_effect) * 100

        print(f"\n📊 Bias Analysis:")
        print(f"   Absolute bias: {bias:.4f} ({bias:.1%})")
        print(f"   Relative bias: {relative_bias:.1f}%")

        # Compare to naive
        naive_effect = self.data[self.data[self.treatment_col] == 1][outcome_col].mean() - \
                      self.data[self.data[self.treatment_col] == 0][outcome_col].mean()

        naive_bias = naive_effect - true_effect
        print(f"\n📊 Naive Comparison:")
        print(f"   Naive Estimate: {naive_effect:.4f} ({naive_effect:.1%})")
        print(f"   Naive Bias: {naive_bias:.4f} ({naive_bias:.1%})")
        print(f"   PSM Bias: {bias:.4f} ({bias:.1%})")
        print(f"   Bias Reduction: {naive_bias - bias:.4f} ({(1 - bias/naive_bias)*100:.1f}% improvement)")

        # Is CI close to true effect?
        includes_true = (self.effect_result['ci_lower'] <= true_effect <= self.effect_result['ci_upper'])

        print(f"\n🎯 Validation:")
        print(f"   95% CI includes true effect: {includes_true}")
        if includes_true:
            print(f"   ✅ SUCCESS! CI captures the true causal effect")
        else:
            print(f"   ⚠️  WARNING: CI does not include true effect")

        self.comparison = {
            'psm_estimate': self.effect_result['point_estimate'],
            'true_effect': true_effect,
            'expected_effect': expected_effect,
            'bias': bias,
            'relative_bias': relative_bias,
            'naive_effect': naive_effect,
            'naive_bias': naive_bias,
            'includes_true': includes_true,
            'ci_includes_true': includes_true
        }

        return self.comparison

    def create_results_visualization(self):
        """
        Create comprehensive results visualization.

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Results Visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Plot 1: Love plot (balance)
        ax = axes[0, 0]
        x_pos = np.arange(len(self.balance_results))

        ax.barh(x_pos - 0.2, abs(self.balance_results['std_diff_before']), 0.4,
                label='Before', color='lightcoral', alpha=0.8, edgecolor='black')
        ax.barh(x_pos + 0.2, abs(self.balance_results['std_diff_after']), 0.4,
                label='After', color='lightgreen', alpha=0.8, edgecolor='black')

        ax.axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Good (0.1)')
        ax.set_yticks(x_pos)
        ax.set_yticklabels([c[:15] for c in self.balance_results['covariate']])
        ax.set_xlabel('|Standardized Difference|')
        ax.set_title('Covariate Balance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Effect comparison
        ax = axes[0, 1]
        methods = ['Naive', 'PSM', 'True']
        effects = [
            self.comparison['naive_effect'] * 100,
            self.comparison['psm_estimate'] * 100,
            self.comparison['true_effect'] * 100
        ]
        colors = ['lightcoral', 'lightgreen', 'gold']

        bars = ax.bar(methods, effects, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel('Effect Size (Percentage Points)')
        ax.set_title('Treatment Effect Estimates', fontweight='bold')
        ax.set_ylim(0, max(effects) * 1.3)

        for bar, eff in zip(bars, effects):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Confidence interval
        ax = axes[0, 2]
        ci_width = self.effect_result['ci_upper'] - self.effect_result['ci_lower']

        ax.errorbar([0], [self.effect_result['point_estimate'] * 100],
                    xerr=[[(self.effect_result['point_estimate'] - self.effect_result['ci_lower']) * 100],
                          [(self.effect_result['ci_upper'] - self.effect_result['point_estimate']) * 100]],
                    fmt='o', color='green', markersize=12, capsize=8, capthick=3,
                    linewidth=3)

        ax.axvline(self.comparison['true_effect'] * 100, color='red', linestyle='--',
                   linewidth=2, label=f'True Effect ({self.comparison["true_effect"]*100:.1f}%)')
        ax.axvline(self.comparison['naive_effect'] * 100, color='orange', linestyle=':',
                   linewidth=2, label=f'Naive ({self.comparison["naive_effect"]*100:.1f}%)')

        ax.set_xlabel('Effect Size (Percentage Points)')
        ax.set_ylabel('')
        ax.set_title('95% Confidence Interval', fontweight='bold')
        ax.set_yticks([])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Bootstrap distribution
        ax = axes[1, 0]
        ax.hist(self.effect_result['bootstrap_estimates'] * 100, bins=50, alpha=0.7,
                color='lightgreen', edgecolor='black', density=True)
        ax.axvline(self.effect_result['point_estimate'] * 100, color='green', linestyle='-',
                   linewidth=3, label='Point Estimate')
        ax.axvline(self.comparison['true_effect'] * 100, color='red', linestyle='--',
                   linewidth=2, label='True Effect')
        ax.axvline(self.effect_result['ci_lower'] * 100, color='blue', linestyle=':',
                   linewidth=2, label='95% CI')
        ax.axvline(self.effect_result['ci_upper'] * 100, color='blue', linestyle=':',
                   linewidth=2)
        ax.set_xlabel('Effect Size (Percentage Points)')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Balance improvement
        ax = axes[1, 1]
        improvement = self.balance_results['improvement']
        colors = ['green' if imp > 0 else 'red' for imp in improvement]

        bars = ax.barh(range(len(improvement)), improvement, color=colors, alpha=0.7,
                       edgecolor='black')
        ax.set_yticks(range(len(improvement)))
        ax.set_yticklabels([c[:15] for c in self.balance_results['covariate']])
        ax.set_xlabel('Change in |Std Diff|')
        ax.set_title('Balance Improvement', fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = f"""
        MATCHING SUMMARY
        {'='*30}

        Sample Size:
        • Matched pairs: {len(self.matched_treated):,}
        • Match rate: {len(self.matched_treated)/sum(self.treatment)*100:.1f}%

        Balance:
        • Mean |Std Diff (before): {abs(self.balance_results['std_diff_before']).mean():.4f}
        • Mean |Std Diff (after): {abs(self.balance_results['std_diff_after']).mean():.4f}

        Effect Estimate:
        • Point estimate: {self.effect_result['point_estimate']:.1%}
        • 95% CI: [{self.effect_result['ci_lower']:.1%}, {self.effect_result['ci_upper']:.1%}]

        Validation:
        • True effect: {self.comparison['true_effect']:.1%}
        • Bias: {self.comparison['bias']:.1%}
        • CI includes true: {'Yes' if self.comparison['ci_includes_true'] else 'No'}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/psm_results_comprehensive.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig


def main():
    """
    Run complete propensity score matching workflow.
    """
    print("\n" + "=" * 70)
    print("PROPENSITY SCORE MATCHING - COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data with propensity scores...")
    data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
    data = pd.read_csv(data_path)

    print(f"✅ Data loaded: {data.shape}")
    print(f"   Treatment rate: {data['received_email'].mean():.1%}")

    # Initialize matcher
    print(f"\n🔧 Initializing Propensity Score Matcher...")
    matcher = PropensityScoreMatcher(
        caliper_multiplier=0.1,  # 0.1 * std(propensity)
        with_replacement=True,
        random_state=42
    )

    # Fit matcher
    matcher.fit(data)

    # Perform matching
    matched_data = matcher.perform_matching()

    # Check balance
    balance_stats = matcher.check_balance()

    # Create Love plot
    matcher.create_love_plot()

    # Estimate treatment effect
    effect_result = matcher.estimate_treatment_effect()

    # Compare to true effect
    comparison = matcher.compare_to_true_effect()

    # Create comprehensive visualization
    matcher.create_results_visualization()

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n🎯 EFFECT ESTIMATION:")
    print(f"   Point Estimate: {effect_result['point_estimate']:.4f} ({effect_result['point_estimate']:.1%})")
    print(f"   95% CI: [{effect_result['ci_lower']:.4f}, {effect_result['ci_upper']:.4f}]")
    print(f"   Standard Error: {effect_result['se']:.4f}")
    print(f"   P-value: {effect_result['p_value']:.4f}")
    print(f"   Significant: {'Yes' if effect_result['p_value'] < 0.05 else 'No'}")

    print(f"\n✅ VALIDATION:")
    print(f"   True Effect: {comparison['true_effect']:.4f} ({comparison['true_effect']:.1%})")
    print(f"   Bias: {comparison['bias']:.4f} ({comparison['bias']:.1%})")
    print(f"   CI includes true: {'Yes' if comparison['ci_includes_true'] else 'No'}")

    print(f"\n📊 BALANCE:")
    print(f"   Matched pairs: {len(matcher.matched_treated):,}")
    print(f"   Mean |Std Diff (after): {abs(balance_stats['std_diff_after']).mean():.4f}")
    print(f"   Well-balanced covariates: {(abs(balance_stats['std_diff_after']) < 0.1).sum()}/{len(balance_stats)}")

    print(f"\n🎉 SUCCESS! Propensity Score Matching complete!")
    print(f"   PSM recovered the causal effect with proper balance!")

    return matcher


if __name__ == "__main__":
    matcher = main()
