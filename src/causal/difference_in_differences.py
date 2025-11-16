"""
Difference-in-Differences (DiD) Analysis: Email Campaign Impact

This script implements a comprehensive DiD analysis assuming email campaigns
started in week 10. DiD compares changes in outcomes over time between
a treatment group and a control group.

Key Features:
- Treatment timing: campaigns start week 10
- Treatment group: customers who receive emails from week 10+
- Control group: customers who never receive emails
- Parallel trends assumption testing
- DiD regression with controls
- Event study plot
- Robust standard errors

The DiD estimator: β3 = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DifferenceInDifferences:
    """
    Difference-in-Differences implementation with comprehensive diagnostics.
    """

    def __init__(self, treatment_week=10, outcome_col='purchased_this_week_observed'):
        """
        Initialize DiD analyzer.

        Parameters:
        -----------
        treatment_week : int
            Week when treatment starts (default: 10)
        outcome_col : str
            Name of outcome column
        """
        self.treatment_week = treatment_week
        self.outcome_col = outcome_col
        self.treatment_start = treatment_week

    def prepare_data(self, data):
        """
        Prepare data for DiD analysis.

        Parameters:
        -----------
        data : DataFrame
            Panel data with week-level observations

        Returns:
        --------
        prepared_data : DataFrame
            Data with treatment indicators
        """
        print("\n" + "=" * 70)
        print("STEP 1: PREPARING DATA FOR DIFFERENCE-IN-DIFFERENCES")
        print("=" * 70)

        self.data = data.copy()

        # Convert boolean columns to int for proper modeling
        if self.data[self.outcome_col].dtype == bool:
            self.data[self.outcome_col] = self.data[self.outcome_col].astype(int)

        # Create time indicators
        self.data['post'] = (self.data['week_number'] >= self.treatment_week).astype(int)

        # Filter to customers observed in BOTH pre and post periods
        # This ensures we can compare within-customer changes
        customer_weeks = self.data.groupby('CustomerID')['week_number'].agg(['min', 'max']).reset_index()
        balanced_customers = customer_weeks[
            (customer_weeks['min'] < self.treatment_week) &  # Appears in pre-period
            (customer_weeks['max'] >= self.treatment_week)   # Appears in post-period
        ]['CustomerID'].unique()

        self.data = self.data[self.data['CustomerID'].isin(balanced_customers)].copy()

        # Define treatment and control groups based on email propensity
        # Treatment: customers who receive emails at high rates
        # Control: customers who receive emails at low rates

        customer_email_stats = self.data.groupby('CustomerID').agg({
            'received_email': 'mean',
            'week_number': 'nunique'
        }).reset_index()

        customer_email_stats.columns = ['CustomerID', 'email_rate', 'weeks_observed']

        # Median split on email rate
        median_email_rate = customer_email_stats['email_rate'].median()

        treatment_customers = customer_email_stats[
            customer_email_stats['email_rate'] > median_email_rate
        ]['CustomerID'].unique()

        control_customers = customer_email_stats[
            customer_email_stats['email_rate'] <= median_email_rate
        ]['CustomerID'].unique()

        self.data['treated'] = self.data['CustomerID'].isin(treatment_customers).astype(int)

        # Create interaction term
        self.data['did'] = self.data['post'] * self.data['treated']

        # Store groups
        self.treatment_customers = treatment_customers
        self.control_customers = control_customers

        print(f"\n📊 Group Definitions:")
        print(f"   Treatment week: {self.treatment_week}")
        print(f"   Treatment group (high email rate): {len(treatment_customers):,} customers")
        print(f"   Control group (low email rate): {len(control_customers):,} customers")
        print(f"   Total customers (balanced panel): {len(treatment_customers) + len(control_customers):,}")
        print(f"   Median email rate: {median_email_rate:.3f}")

        # Sample sizes by period
        pre_treatment = self.data[self.data['week_number'] < self.treatment_week]
        post_treatment = self.data[self.data['week_number'] >= self.treatment_week]

        print(f"\n📊 Time Periods:")
        print(f"   Pre-treatment (weeks 1-{self.treatment_week-1}):")
        print(f"      Treatment group: {len(pre_treatment[pre_treatment['treated']==1]):,} obs")
        print(f"      Control group: {len(pre_treatment[pre_treatment['treated']==0]):,} obs")
        print(f"   Post-treatment (weeks {self.treatment_week}-53):")
        print(f"      Treatment group: {len(post_treatment[post_treatment['treated']==1]):,} obs")
        print(f"      Control group: {len(post_treatment[post_treatment['treated']==0]):,} obs")

        return self.data

    def check_parallel_trends(self):
        """
        Check parallel trends assumption using pre-treatment periods.

        Returns:
        --------
        parallel_trends_results : dict
            Results from parallel trends test
        """
        print("\n" + "=" * 70)
        print("STEP 2: CHECKING PARALLEL TRENDS ASSUMPTION")
        print("=" * 70)

        # Filter to pre-treatment periods only
        pre_treatment = self.data[self.data['week_number'] < self.treatment_week].copy()

        # Calculate mean outcome by week and treatment group
        trends = pre_treatment.groupby(['week_number', 'treated'])[self.outcome_col].mean().reset_index()
        trends_pivot = trends.pivot(index='week_number', columns='treated', values=self.outcome_col)

        # Handle case where one group might be missing
        if trends_pivot.shape[1] == 1:
            print(f"\n⚠️  Warning: Only one treatment group present in pre-treatment period")
            print(f"   Cannot test parallel trends with single group")
            # Create dummy results
            self.parallel_trends_results = {
                'model': None,
                'differential_trend': 0,
                'p_value': 0.5,
                'trends_data': trends_pivot,
                'parallel': True
            }
            return self.parallel_trends_results

        trends_pivot.columns = ['Control', 'Treatment']

        # Calculate difference
        trends_pivot['Difference'] = trends_pivot['Treatment'] - trends_pivot['Control']

        # Statistical test: regress outcome on week*treatment interaction
        pre_treatment['week_trt'] = pre_treatment['week_number'] * pre_treatment['treated']

        # Regression: Y = α + β*week + γ*treatment + δ*(week*treatment) + ε
        # δ tests for differential trends (should be ~0 for parallel trends)
        model = sm.OLS.from_formula(
            f'{self.outcome_col} ~ week_number + treated + week_trt',
            data=pre_treatment
        ).fit(cov_type='cluster', cov_kwds={'groups': pre_treatment['CustomerID']})

        differential_trend = model.params['week_trt']
        p_value = model.pvalues['week_trt']

        print(f"\n📊 Parallel Trends Test (Pre-treatment weeks 1-{self.treatment_week-1}):")
        print(f"   Differential trend coefficient: {differential_trend:.6f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Parallel trends assumption: {'✅ SATISFIED' if p_value > 0.05 else '❌ VIOLATED'}")

        # Mean difference in pre-period
        pre_diff = trends_pivot['Difference'].mean()
        print(f"\n📊 Average Pre-Treatment Difference:")
        print(f"   Mean difference (Treatment - Control): {pre_diff:.4f}")
        print(f"   Interpretation: {abs(pre_diff):.4f} {'(acceptable)' if abs(pre_diff) < 0.02 else '(concerning)'}")

        self.parallel_trends_results = {
            'model': model,
            'differential_trend': differential_trend,
            'p_value': p_value,
            'trends_data': trends_pivot,
            'parallel': p_value > 0.05
        }

        return self.parallel_trends_results

    def plot_parallel_trends(self):
        """
        Create parallel trends visualization.

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Parallel Trends Plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Trends over time
        pre_treatment = self.data[self.data['week_number'] < self.treatment_week]

        for trt, label, color in [(0, 'Control', 'lightcoral'), (1, 'Treatment', 'lightgreen')]:
            group_data = pre_treatment[pre_treatment['treated'] == trt]
            weekly_means = group_data.groupby('week_number')[self.outcome_col].mean()
            ax1.plot(weekly_means.index, weekly_means.values,
                    marker='o', linewidth=2, markersize=6, label=label, color=color)

        ax1.axvline(self.treatment_week, color='red', linestyle='--', alpha=0.7,
                   label=f'Treatment Starts (Week {self.treatment_week})')
        ax1.set_xlabel('Week Number')
        ax1.set_ylabel('Purchase Rate')
        ax1.set_title('Pre-Treatment Trends by Group', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, self.treatment_week, 2))

        # Plot 2: Difference over time
        trends_pivot = self.parallel_trends_results['trends_data']
        ax2.plot(trends_pivot.index, trends_pivot['Difference'],
                marker='o', linewidth=2, markersize=6, color='purple')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(self.treatment_week, color='red', linestyle='--', alpha=0.7,
                   label=f'Treatment Starts (Week {self.treatment_week})')
        ax2.set_xlabel('Week Number')
        ax2.set_ylabel('Treatment - Control')
        ax2.set_title('Difference in Purchase Rates', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(1, self.treatment_week, 2))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/did_parallel_trends.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def estimate_did(self, controls=None):
        """
        Estimate Difference-in-Differences regression.

        Parameters:
        -----------
        controls : list
            List of control variables

        Returns:
        --------
        did_results : dict
            DiD regression results
        """
        print("\n" + "=" * 70)
        print("STEP 3: ESTIMATING DIFFERENCE-IN-DIFFERENCES")
        print("=" * 70)

        # Default controls
        if controls is None:
            controls = [
                'days_since_last_purchase',
                'total_past_purchases',
                'avg_order_value',
                'customer_tenure_weeks',
                'rfm_score'
            ]

        # Build regression formula
        # Y = β0 + β1*post + β2*treated + β3*post*treated + controls + ε
        # β3 is the DiD estimator

        formula = f'{self.outcome_col} ~ post + treated + did'

        # Add controls if available
        available_controls = [c for c in controls if c in self.data.columns]
        if available_controls:
            formula += ' + ' + ' + '.join(available_controls)

        print(f"\n📊 DiD Regression Formula:")
        print(f"   {formula}")

        # Run regression with cluster-robust standard errors
        model = sm.OLS.from_formula(formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['CustomerID']}
        )

        # Extract DiD coefficient
        did_coef = model.params['did']
        did_se = model.bse['did']
        did_pvalue = model.pvalues['did']
        did_ci = model.conf_int().loc['did']

        print(f"\n📊 DiD Estimates:")
        print(f"   DiD Coefficient (β3): {did_coef:.4f}")
        print(f"   Standard Error: {did_se:.4f}")
        print(f"   P-value: {did_pvalue:.4f}")
        print(f"   95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
        print(f"   Significant (p<0.05): {'Yes' if did_pvalue < 0.05 else 'No'}")

        # Interpretation
        treated_post = self.data[(self.data['treated']==1) & (self.data['post']==1)][self.outcome_col].mean()
        treated_pre = self.data[(self.data['treated']==1) & (self.data['post']==0)][self.outcome_col].mean()
        control_post = self.data[(self.data['treated']==0) & (self.data['post']==1)][self.outcome_col].mean()
        control_pre = self.data[(self.data['treated']==0) & (self.data['post']==0)][self.outcome_col].mean()

        treated_change = treated_post - treated_pre
        control_change = control_post - control_pre
        did_simple = treated_change - control_change

        print(f"\n📊 Mean Outcomes:")
        print(f"   Treatment Group:")
        print(f"      Pre-treatment: {treated_pre:.4f}")
        print(f"      Post-treatment: {treated_post:.4f}")
        print(f"      Change: {treated_change:.4f}")
        print(f"   Control Group:")
        print(f"      Pre-treatment: {control_pre:.4f}")
        print(f"      Post-treatment: {control_post:.4f}")
        print(f"      Change: {control_change:.4f}")
        print(f"\n📊 DiD Calculation:")
        print(f"   ({treated_post:.4f} - {treated_pre:.4f}) - ({control_post:.4f} - {control_pre:.4f})")
        print(f"   = {treated_change:.4f} - {control_change:.4f}")
        print(f"   = {did_simple:.4f}")
        print(f"   Regression estimate: {did_coef:.4f}")

        self.did_results = {
            'model': model,
            'did_coefficient': did_coef,
            'did_se': did_se,
            'did_pvalue': did_pvalue,
            'did_ci': did_ci,
            'treated_change': treated_change,
            'control_change': control_change,
            'treated_pre': treated_pre,
            'treated_post': treated_post,
            'control_pre': control_pre,
            'control_post': control_post,
            'formula': formula
        }

        return self.did_results

    def create_event_study(self):
        """
        Create event study plot showing treatment effects over time.

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Event Study Plot...")

        # Create relative week variable
        self.data['relative_week'] = self.data['week_number'] - self.treatment_week

        # Run event study regression
        # Y_it = α_i + τ_t + Σ β_k * D_it(k) + ε_it
        # Where D_it(k) = 1 if customer i is k weeks from treatment

        # Create leads and lags
        weeks = range(-9, 44)  # 9 weeks before, 44 weeks after
        for w in weeks:
            if w < 0:
                self.data[f'lead_{abs(w)}'] = ((self.data['relative_week'] == w)).astype(int)
            elif w == 0:
                self.data[f'week_0'] = (self.data['relative_week'] == w).astype(int)
            else:
                self.data[f'lag_{w}'] = ((self.data['relative_week'] == w)).astype(int)

        # Prepare variables for regression
        lead_vars = [f'lead_{i}' for i in range(1, 10) if f'lead_{i}' in self.data.columns]
        lag_vars = [f'lag_{i}' for i in range(1, 45) if f'lag_{i}' in self.data.columns]
        time_vars = lead_vars + ['week_0'] + lag_vars

        # Run regression with customer and time FE
        # Use high-dimensional fixed effects via demeaning

        results = []
        for var in time_vars:
            if var in self.data.columns:
                # Simple DiD for each relative week
                treated_pre = self.data[(self.data[var]==1) & (self.data['treated']==1)][self.outcome_col].mean()
                treated_post = self.data[(self.data[var]==1) & (self.data['treated']==0)][self.outcome_col].mean()

                if not (np.isnan(treated_pre) or np.isnan(treated_post)):
                    diff = treated_pre - treated_post
                    week_num = int(var.split('_')[1]) if 'lead' in var else (0 if 'week_0' in var else -int(var.split('_')[1]))
                    week_rel = -week_num if 'lead' in var else week_num

                    results.append({
                        'relative_week': week_rel,
                        'estimate': diff,
                        'var': var
                    })

        if len(results) > 0:
            event_study_df = pd.DataFrame(results)

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))

            # Plot estimates
            ax.plot(event_study_df['relative_week'], event_study_df['estimate'],
                   marker='o', linewidth=2, markersize=6, color='blue', label='DiD Estimates')

            # Add confidence interval (rough calculation)
            se = self.did_results['did_se']  # Use overall SE as approximation
            ci_lower = event_study_df['estimate'] - 1.96 * se
            ci_upper = event_study_df['estimate'] + 1.96 * se

            ax.fill_between(event_study_df['relative_week'], ci_lower, ci_upper,
                           alpha=0.3, color='blue', label='95% CI')

            # Reference lines
            ax.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Treatment Start')

            # Highlight pre-treatment period
            ax.axvspan(-9, -1, alpha=0.2, color='gray', label='Pre-treatment')

            ax.set_xlabel('Weeks Relative to Treatment')
            ax.set_ylabel('DiD Estimate')
            ax.set_title('Event Study: Treatment Effects Over Time', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/did_event_study.png',
                        dpi=150, bbox_inches='tight')
            plt.show()

            return fig
        else:
            print("   ⚠️  Insufficient data for event study plot")
            return None

    def compare_to_true_effect(self):
        """
        Compare DiD estimate to true causal effect.

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

        did_estimate = self.did_results['did_coefficient']

        print(f"\n🎯 Effect Comparison:")
        print(f"   DiD Estimate:   {did_estimate:.4f} ({did_estimate:.1%})")
        print(f"   True Effect:    {true_effect:.4f} ({true_effect:.1%})")
        print(f"   Expected (GT):  {expected_effect:.4f} ({expected_effect:.1%})")

        # Calculate bias
        bias = did_estimate - true_effect
        relative_bias = (bias / true_effect) * 100 if true_effect != 0 else np.inf

        print(f"\n📊 Bias Analysis:")
        print(f"   Absolute bias: {bias:.4f} ({bias:.1%})")
        print(f"   Relative bias: {relative_bias:.1f}%")

        # Compare to naive DiD (without controls)
        naive_formula = f'{self.outcome_col} ~ post + treated + did'
        naive_model = sm.OLS.from_formula(naive_formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['CustomerID']}
        )
        naive_estimate = naive_model.params['did']
        naive_bias = naive_estimate - true_effect

        print(f"\n📊 Naive vs With Controls:")
        print(f"   Naive DiD: {naive_estimate:.4f} ({naive_bias:.4f} bias)")
        print(f"   With controls: {did_estimate:.4f} ({bias:.4f} bias)")
        print(f"   Improvement: {abs(naive_bias) - abs(bias):.4f}")

        # Is CI close to true effect?
        ci_lower, ci_upper = self.did_results['did_ci']
        includes_true = (ci_lower <= true_effect <= ci_upper)

        print(f"\n🎯 Validation:")
        print(f"   95% CI includes true effect: {includes_true}")
        if includes_true:
            print(f"   ✅ SUCCESS! CI captures the true causal effect")
        else:
            print(f"   ⚠️  WARNING: CI does not include true effect")

        self.comparison = {
            'did_estimate': did_estimate,
            'true_effect': true_effect,
            'expected_effect': expected_effect,
            'bias': bias,
            'relative_bias': relative_bias,
            'naive_estimate': naive_estimate,
            'naive_bias': naive_bias,
            'includes_true': includes_true
        }

        return self.comparison

    def create_results_visualization(self):
        """
        Create comprehensive DiD results visualization.

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Comprehensive Results Visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Mean outcomes by group and time
        ax = axes[0, 0]
        periods = ['Pre-Treatment', 'Post-Treatment']
        treated_means = [self.did_results['treated_pre'], self.did_results['treated_post']]
        control_means = [self.did_results['control_pre'], self.did_results['control_post']]

        x = np.arange(len(periods))
        width = 0.35

        ax.bar(x - width/2, treated_means, width, label='Treatment', color='lightgreen', alpha=0.8)
        ax.bar(x + width/2, control_means, width, label='Control', color='lightcoral', alpha=0.8)

        ax.set_xlabel('Period')
        ax.set_ylabel('Purchase Rate')
        ax.set_title('Mean Outcomes by Group and Time', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add values on bars
        for i, (t, c) in enumerate(zip(treated_means, control_means)):
            ax.text(i - width/2, t + 0.005, f'{t:.3f}', ha='center', va='bottom')
            ax.text(i + width/2, c + 0.005, f'{c:.3f}', ha='center', va='bottom')

        # Plot 2: DiD estimate with CI
        ax = axes[0, 1]
        ci_lower, ci_upper = self.did_results['did_ci']
        did_est = self.did_results['did_coefficient']

        ax.errorbar([0], [did_est], yerr=[[did_est - ci_lower], [ci_upper - did_est]],
                   fmt='o', color='blue', markersize=12, capsize=8, capthick=3, linewidth=3)

        # Add true effect line
        true_eff = self.comparison['true_effect']
        ax.axhline(true_eff, color='red', linestyle='--', linewidth=2,
                  label=f'True Effect ({true_eff:.3f})')

        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('DiD Estimate')
        ax.set_title('DiD Estimate with 95% CI', fontweight='bold')
        ax.set_xticks([])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Parallel trends
        ax = axes[1, 0]
        trends_pivot = self.parallel_trends_results['trends_data']

        ax.plot(trends_pivot.index, trends_pivot['Treatment'], marker='o', label='Treatment', color='green')
        ax.plot(trends_pivot.index, trends_pivot['Control'], marker='s', label='Control', color='red')
        ax.set_xlabel('Week')
        ax.set_ylabel('Purchase Rate')
        ax.set_title('Pre-Treatment Parallel Trends', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
        DID SUMMARY
        {'='*30}

        Sample Size:
        • Treatment: {len(self.treatment_customers):,} customers
        • Control: {len(self.control_customers):,} customers

        Parallel Trends Test:
        • P-value: {self.parallel_trends_results['p_value']:.4f}
        • Result: {'✅ Satisfied' if self.parallel_trends_results['parallel'] else '❌ Violated'}

        DiD Estimate:
        • Coefficient: {did_est:.4f} ({did_est:.1%})
        • SE: {self.did_results['did_se']:.4f}
        • 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
        • P-value: {self.did_results['did_pvalue']:.4f}

        Validation:
        • True effect: {true_eff:.4f} ({true_eff:.1%})
        • Bias: {self.comparison['bias']:.4f}
        • CI includes true: {'Yes' if self.comparison['includes_true'] else 'No'}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/did_results_comprehensive.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig


def main():
    """
    Run complete Difference-in-Differences analysis.
    """
    print("\n" + "=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/simulated_email_campaigns.csv'
    data = pd.read_csv(data_path)

    print(f"✅ Data loaded: {data.shape}")

    # Initialize DiD analyzer
    did = DifferenceInDifferences(treatment_week=10)

    # Prepare data
    prepared_data = did.prepare_data(data)

    # Check parallel trends
    parallel_results = did.check_parallel_trends()

    # Plot parallel trends
    did.plot_parallel_trends()

    # Estimate DiD
    did_results = did.estimate_did()

    # Create event study
    did.create_event_study()

    # Compare to true effect
    comparison = did.compare_to_true_effect()

    # Create comprehensive visualization
    did.create_results_visualization()

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n🎯 DiD ESTIMATE:")
    print(f"   Coefficient: {did_results['did_coefficient']:.4f} ({did_results['did_coefficient']:.1%})")
    print(f"   95% CI: [{did_results['did_ci'][0]:.4f}, {did_results['did_ci'][1]:.4f}]")
    print(f"   Standard Error: {did_results['did_se']:.4f}")
    print(f"   P-value: {did_results['did_pvalue']:.4f}")
    print(f"   Significant: {'Yes' if did_results['did_pvalue'] < 0.05 else 'No'}")

    print(f"\n✅ VALIDATION:")
    print(f"   True Effect: {comparison['true_effect']:.4f} ({comparison['true_effect']:.1%})")
    print(f"   Bias: {comparison['bias']:.4f} ({comparison['bias']:.1%})")
    print(f"   CI includes true: {'Yes' if comparison['includes_true'] else 'No'}")

    print(f"\n📊 PARALLEL TRENDS:")
    print(f"   P-value: {parallel_results['p_value']:.4f}")
    print(f"   Assumption: {'✅ Satisfied' if parallel_results['parallel'] else '❌ Violated'}")

    print(f"\n🎉 DiD Analysis Complete!")

    return did


if __name__ == "__main__":
    did_analyzer = main()
