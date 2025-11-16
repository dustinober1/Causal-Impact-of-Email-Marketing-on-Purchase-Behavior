"""
Inverse Probability Weighting (IPW) Causal Inference

This script implements IPW estimator that uses propensity scores to create
a weighted sample where treatment assignment is as-if random.

Key Features:
1. IPW estimator for Average Treatment Effect
2. Weighted and stabilized weights
3. Trimming/censoring for extreme propensity scores
4. Bootstrap standard errors
5. Diagnostic plots (weight distributions, QQ plots)
6. Comparison to AIPW and other methods

IPW Estimator:
ATE = E[ T*Y/e(X) ] - E[ (1-T)*Y/(1-e(X)) ]

Where e(X) = P(T=1|X) is the propensity score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class InverseProbabilityWeighting:
    """
    Inverse Probability Weighting for Causal Inference.
    """

    def __init__(self, trim_percentile=1, stabilize=False, random_state=42):
        """
        Initialize the IPW estimator.

        Parameters:
        -----------
        trim_percentile : float
            Trim propensity scores below this percentile and above (100-trim_percentile)
        stabilize : bool
            Whether to use stabilized weights
        random_state : int
            Random seed
        """
        self.trim_percentile = trim_percentile
        self.stabilize = stabilize
        self.random_state = random_state
        self.propensity_scores = None

    def calculate_ipw_weights(self, data, treatment_col='received_email'):
        """
        Calculate IPW weights with trimming and stabilization.

        Parameters:
        -----------
        data : DataFrame
            Data with propensity scores
        treatment_col : str
            Treatment column name

        Returns:
        --------
        weights : array
            IPW weights
        """
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

        # Stabilized weights (optional)
        if self.stabilize:
            # Normalize by marginal probability of treatment
            p_treated = np.mean(T)
            p_control = 1 - p_treated
            weights_treated = weights_treated * p_treated
            weights_control = weights_control * p_control

        weights = weights_treated + weights_control

        return weights, e_trimmed

    def estimate_ipw(self, data, outcome_col='purchased_this_week_observed'):
        """
        Estimate Average Treatment Effect using IPW.

        Parameters:
        -----------
        data : DataFrame
            Data with propensity scores and outcomes
        outcome_col : str
            Outcome column name

        Returns:
        --------
        result : dict
            IPW results
        """
        print("\n" + "=" * 70)
        print("STEP 1: ESTIMATING ATE WITH INVERSE PROBABILITY WEIGHTING")
        print("=" * 70)

        Y = data[outcome_col].values
        T = data['received_email'].values

        # Calculate IPW weights
        weights, e_trimmed = self.calculate_ipw_weights(data)

        # IPW estimators
        # E[Y(1)] = E[w_i * T_i * Y_i] where w_i = 1/e_i
        # E[Y(0)] = E[w_i * (1-T_i) * Y_i] where w_i = 1/(1-e_i)

        treated_weighted = np.sum(T * Y / e_trimmed) / len(data)
        control_weighted = np.sum((1 - T) * Y / (1 - e_trimmed)) / len(data)

        # Alternatively using weights array
        e_y1_weighted = np.sum(weights[T == 1] * Y[T == 1]) / len(data) * np.mean(T)
        e_y0_weighted = np.sum(weights[T == 0] * Y[T == 0]) / len(data) * np.mean(1 - T)

        ipw_ate = treated_weighted - control_weighted

        # Naive estimate for comparison
        naive_treated = np.mean(Y[T == 1])
        naive_control = np.mean(Y[T == 0])
        naive_ate = naive_treated - naive_control

        print(f"\n📊 IPW Results:")
        print(f"   ATE (IPW): {ipw_ate:.4f} ({ipw_ate:.1%})")
        print(f"   E[Y(1)]: {treated_weighted:.4f}")
        print(f"   E[Y(0)]: {control_weighted:.4f}")

        print(f"\n📊 Comparison:")
        print(f"   Naive (unweighted): {naive_ate:.4f} ({naive_ate:.1%})")
        print(f"   IPW (weighted): {ipw_ate:.4f} ({ipw_ate:.1%})")

        # Weight diagnostics
        print(f"\n📊 Weight Diagnostics:")
        print(f"   Mean weight (treated): {np.mean(weights[T == 1]):.2f}")
        print(f"   Mean weight (control): {np.mean(weights[T == 0]):.2f}")
        print(f"   Max weight (treated): {np.max(weights[T == 1]):.2f}")
        print(f"   Max weight (control): {np.max(weights[T == 0]):.2f}")
        print(f"   % trimmed: {np.mean(e_trimmed != self.propensity_scores):.1%}")

        self.ipw_result = {
            'ate': ipw_ate,
            'e_y1': treated_weighted,
            'e_y0': control_weighted,
            'naive': naive_ate,
            'weights': weights,
            'e_trimmed': e_trimmed
        }

        return self.ipw_result

    def bootstrap_se(self, data, outcome_col='purchased_this_week_observed', n_bootstrap=500):
        """
        Calculate bootstrap standard errors for IPW.

        Parameters:
        -----------
        data : DataFrame
            Data
        outcome_col : str
            Outcome column name
        n_bootstrap : int
            Number of bootstrap samples

        Returns:
        --------
        bootstrap_results : dict
            Bootstrap results with SEs and CI
        """
        print("\n" + "=" * 70)
        print("STEP 2: BOOTSTRAP STANDARD ERRORS")
        print("=" * 70)

        print(f"\n🔄 Bootstrapping ({n_bootstrap:,} samples)...")

        n = len(data)
        bootstrap_ates = []

        # Get propensity scores and weights
        propensity_scores = self.propensity_scores

        for i in range(n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{n_bootstrap} ({100*(i+1)/n_bootstrap:.1f}%")

            # Bootstrap sample
            boot_idx = np.random.choice(n, size=n, replace=True)
            boot_data = data.iloc[boot_idx].copy()
            boot_propensity = propensity_scores[boot_idx]

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

        bootstrap_ates = np.array(bootstrap_ates)

        # Calculate statistics
        se_bootstrap = np.std(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        print(f"\n📊 Bootstrap Results:")
        print(f"   IPW ATE: {self.ipw_result['ate']:.4f}")
        print(f"   Bootstrap SE: {se_bootstrap:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

        # Z-test
        z_stat = self.ipw_result['ate'] / se_bootstrap
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        print(f"\n📊 Statistical Significance:")
        print(f"   Z-statistic: {z_stat:.2f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

        self.bootstrap_results = {
            'ate_bootstrap': bootstrap_ates,
            'se': se_bootstrap,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_stat': z_stat,
            'p_value': p_value
        }

        return self.bootstrap_results

    def diagnostic_plots(self, data):
        """
        Create diagnostic plots for IPW weights.

        Parameters:
        -----------
        data : DataFrame
            Data

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating IPW Diagnostic Plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Weight distribution by treatment
        ax = axes[0, 0]
        weights = self.ipw_result['weights']
        treated_data = data[data['received_email'] == 1]
        control_data = data[data['received_email'] == 0]

        ax.hist(weights[treated_data.index], bins=50, alpha=0.7, label='Treated',
               color='lightgreen', edgecolor='black')
        ax.hist(weights[control_data.index], bins=50, alpha=0.7, label='Control',
               color='lightcoral', edgecolor='black')
        ax.set_xlabel('IPW Weight')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of IPW Weights', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Propensity score distribution
        ax = axes[0, 1]
        ax.hist(self.propensity_scores[treated_data.index], bins=50, alpha=0.7,
               label='Treated', color='lightgreen', edgecolor='black')
        ax.hist(self.propensity_scores[control_data.index], bins=50, alpha=0.7,
               label='Control', color='lightcoral', edgecolor='black')
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Propensity Scores', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Weight vs propensity score
        ax = axes[1, 0]
        ax.scatter(self.propensity_scores[treated_data.index],
                  weights[treated_data.index],
                  alpha=0.3, label='Treated', color='lightgreen', s=10)
        ax.scatter(self.propensity_scores[control_data.index],
                  weights[control_data.index],
                  alpha=0.3, label='Control', color='lightcoral', s=10)
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('IPW Weight')
        ax.set_title('IPW Weight vs Propensity Score', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
        IPW SUMMARY
        {'='*30}

        ATE Estimate:
        • IPW: {self.ipw_result['ate']:.4f} ({self.ipw_result['ate']:.1%})
        • Naive: {self.ipw_result['naive']:.4f} ({self.ipw_result['naive']:.1%})

        Weight Statistics:
        • Mean (treated): {np.mean(weights[treated_data.index]):.2f}
        • Mean (control): {np.mean(weights[control_data.index]):.2f}
        • Max (treated): {np.max(weights[treated_data.index]):.2f}
        • Max (control): {np.max(weights[control_data.index]):.2f}

        Trimming:
        • Percentile: {self.trim_percentile}%
        • % Trimmed: {np.mean(self.ipw_result['e_trimmed'] != self.propensity_scores):.1%}

        Effective Sample Size:
        • Treated: {np.sum(weights[treated_data.index]**2) / (np.sum(weights[treated_data.index])**2):.0f}
        • Control: {np.sum(weights[control_data.index]**2) / (np.sum(weights[control_data.index])**2):.0f}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/ipw_diagnostics.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def compare_to_true_effect(self):
        """
        Compare IPW estimate to true causal effect.

        Returns:
        --------
        comparison : dict
            Comparison results
        """
        print("\n" + "=" * 70)
        print("STEP 3: COMPARING TO TRUE CAUSAL EFFECT")
        print("=" * 70)

        # Load ground truth
        ground_truth_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/ground_truth.json'
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        # Load data to get true effects
        data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
        data = pd.read_csv(data_path)

        true_effect = data['individual_treatment_effect'].mean()
        expected_effect = ground_truth['base_email_effect']

        print(f"\n🎯 Effect Comparison:")
        print(f"   IPW Estimate: {self.ipw_result['ate']:.4f} ({self.ipw_result['ate']:.1%})")
        print(f"   True Effect: {true_effect:.4f} ({true_effect:.1%})")
        print(f"   Expected (GT): {expected_effect:.4f} ({expected_effect:.1%})")

        # Calculate bias
        ipw_bias = self.ipw_result['ate'] - true_effect

        print(f"\n📊 Bias Analysis:")
        print(f"   IPW Bias: {ipw_bias:.4f} ({ipw_bias:.1%})")

        # Compare to naive
        naive_effect = self.ipw_result['naive']
        naive_bias = naive_effect - true_effect

        print(f"\n📊 Method Comparison:")
        print(f"   Naive: {naive_effect:.4f} ({naive_bias:.4f} bias)")
        print(f"   IPW: {self.ipw_result['ate']:.4f} ({ipw_bias:.4f} bias)")

        # Is CI close to true effect?
        if hasattr(self, 'bootstrap_results'):
            includes_true = (self.bootstrap_results['ci_lower'] <= true_effect <=
                           self.bootstrap_results['ci_upper'])

            print(f"\n🎯 Validation:")
            print(f"   95% CI includes true effect: {includes_true}")
            if includes_true:
                print(f"   ✅ SUCCESS! CI captures the true causal effect")
            else:
                print(f"   ⚠️  CI does not include true effect")

        self.comparison = {
            'ipw_estimate': self.ipw_result['ate'],
            'true_effect': true_effect,
            'expected_effect': expected_effect,
            'ipw_bias': ipw_bias,
            'naive_bias': naive_bias
        }

        return self.comparison

    def create_comprehensive_results_plot(self, data):
        """
        Create comprehensive results visualization.

        Parameters:
        -----------
        data : DataFrame
            Data

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Comprehensive IPW Results Visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Method comparison
        ax = axes[0, 0]
        methods = ['Naive', 'IPW']
        estimates = [self.ipw_result['naive'], self.ipw_result['ate']]

        bars = ax.bar(methods, estimates, color=['lightcoral', 'lightblue'],
                     edgecolor='black', alpha=0.7)
        ax.set_ylabel('ATE Estimate')
        ax.set_title('Method Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, estimates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}\n({val:.1%})', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Bootstrap distribution
        ax = axes[0, 1]
        if hasattr(self, 'bootstrap_results'):
            ax.hist(self.bootstrap_results['ate_bootstrap'], bins=50, alpha=0.7,
                   color='lightblue', edgecolor='black')
            ax.axvline(self.ipw_result['ate'], color='red', linestyle='--', linewidth=2,
                      label=f'IPW Estimate: {self.ipw_result["ate"]:.3f}')
            ax.axvline(self.comparison['true_effect'], color='green', linestyle='--', linewidth=2,
                      label=f'True Effect: {self.comparison["true_effect"]:.3f}')
            ax.set_xlabel('Bootstrap ATE')
            ax.set_ylabel('Count')
            ax.set_title('Bootstrap Distribution of IPW ATE', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Weight distribution
        ax = axes[1, 0]
        weights = self.ipw_result['weights']
        treated_data = data[data['received_email'] == 1]
        control_data = data[data['received_email'] == 0]

        ax.hist(weights[treated_data.index], bins=50, alpha=0.7, label='Treated',
               color='lightgreen', edgecolor='black')
        ax.hist(weights[control_data.index], bins=50, alpha=0.7, label='Control',
               color='lightcoral', edgecolor='black')
        ax.set_xlabel('IPW Weight')
        ax.set_ylabel('Count')
        ax.set_title('IPW Weight Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
        IPW RESULTS SUMMARY
        {'='*35}

        ATE Estimates:
        • IPW: {self.ipw_result['ate']:.4f} ({self.ipw_result['ate']:.1%})
        • Naive: {self.ipw_result['naive']:.4f} ({self.ipw_result['naive']:.1%})
        • True Effect: {self.comparison['true_effect']:.4f} ({self.comparison['true_effect']:.1%})

        Bias Analysis:
        • IPW Bias: {self.comparison['ipw_bias']:.4f} ({self.comparison['ipw_bias']:.1%})
        • Naive Bias: {self.comparison['naive_bias']:.4f} ({self.comparison['naive_bias']:.1%})

        Statistical Inference:
        • Bootstrap SE: {self.bootstrap_results['se']:.4f}
        • 95% CI: [{self.bootstrap_results['ci_lower']:.4f}, {self.bootstrap_results['ci_upper']:.4f}]
        • Z-statistic: {self.bootstrap_results['z_stat']:.2f}
        • P-value: {self.bootstrap_results['p_value']:.4f}

        Weight Quality:
        • Mean (treated): {np.mean(weights[treated_data.index]):.2f}
        • Mean (control): {np.mean(weights[control_data.index]):.2f}
        • Trimming: {self.trim_percentile}%
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/ipw_results_comprehensive.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig


def main():
    """
    Run complete IPW analysis.
    """
    print("\n" + "=" * 70)
    print("INVERSE PROBABILITY WEIGHTING (IPW) CAUSAL INFERENCE")
    print("=" * 70)

    # Load data
    print("\nLoading data with propensity scores...")
    data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
    data = pd.read_csv(data_path)

    print(f"✅ Data loaded: {data.shape}")
    print(f"   Treatment rate: {data['received_email'].mean():.1%}")

    # Use pre-computed propensity scores
    print(f"\n🔧 Loading propensity scores...")
    ipw = InverseProbabilityWeighting(trim_percentile=1, stabilize=False, random_state=42)
    ipw.propensity_scores = data['propensity_score'].values

    # Estimate IPW
    ipw_result = ipw.estimate_ipw(data)

    # Bootstrap standard errors
    bootstrap_results = ipw.bootstrap_se(data, n_bootstrap=200)

    # Compare to true effect
    comparison = ipw.compare_to_true_effect()

    # Create diagnostic plots
    ipw.diagnostic_plots(data)

    # Create comprehensive results
    ipw.create_comprehensive_results_plot(data)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n🎯 IPW ESTIMATES:")
    print(f"   IPW ATE:     {ipw_result['ate']:.4f} ({ipw_result['ate']:.1%})")
    print(f"   95% CI:      [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
    print(f"   P-value:     {bootstrap_results['p_value']:.4f}")

    print(f"\n✅ VALIDATION:")
    print(f"   True Effect: {comparison['true_effect']:.4f} ({comparison['true_effect']:.1%})")
    print(f"   IPW Bias:    {comparison['ipw_bias']:.4f}")
    print(f"   CI includes true: {'Yes' if (bootstrap_results['ci_lower'] <= comparison['true_effect'] <= bootstrap_results['ci_upper']) else 'No'}")

    print(f"\n🎉 IPW Analysis Complete!")

    return ipw


if __name__ == "__main__":
    ipw_estimator = main()