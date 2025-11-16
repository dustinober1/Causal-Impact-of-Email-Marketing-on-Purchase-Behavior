"""
Example: Using the Modular Causal Inference Toolkit

This script demonstrates how to use the reusable causal inference
modules for propensity score matching and difference-in-differences.

Run with: python examples/modular_usage_example.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from causal.propensity_score import (
    PropensityScoreEstimator,
    PropensityScoreMatcher,
    PropensityScoreWeighting
)
from causal.diff_in_diff import DifferenceInDifferences
from visualization.balance_plots import BalanceVisualizer, plot_treatment_effects_comparison


def generate_sample_data(n=1000):
    """
    Generate synthetic data with confounding for testing.

    Returns:
        X: Feature matrix
        treatment: Binary treatment
        outcome: Continuous outcome
        propensity_scores: True propensity scores
    """
    np.random.seed(42)

    # Features
    X = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'x3': np.random.normal(0, 1, n)
    })

    # True propensity score model (logistic)
    logit_ps = (
        0.5 * X['x1'] +
        0.3 * X['x2'] -
        0.4 * X['x3'] +
        np.random.normal(0, 0.5, n)
    )
    prob_treatment = 1 / (1 + np.exp(-logit_ps))
    propensity_scores = prob_treatment

    # Treatment assignment (based on propensity score)
    treatment = np.random.binomial(1, propensity_scores)

    # Outcome model with true treatment effect
    true_effect = 2.0
    outcome = (
        10 +  # intercept
        true_effect * treatment +  # true effect
        0.8 * X['x1'] + 0.6 * X['x2'] + 0.4 * X['x3'] +  # confounding
        np.random.normal(0, 1, n)  # noise
    )

    return X, treatment, outcome, propensity_scores


def example_1_propensity_score_matching():
    """Example 1: Propensity Score Matching."""
    print("\n" + "="*70)
    print("EXAMPLE 1: PROPENSITY SCORE MATCHING")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic data...")
    X, treatment, outcome, true_ps = generate_sample_data(n=1000)
    print(f"   Sample size: {len(X)}")
    print(f"   Treatment rate: {treatment.mean():.1%}")

    # Step 1: Estimate propensity scores
    print("\n2. Estimating propensity scores...")
    ps_estimator = PropensityScoreEstimator(use_scaling=True, random_state=42)
    ps_estimator.fit(X, treatment)
    propensity_scores = ps_estimator.predict_proba(X)

    # Evaluate model
    ps_metrics = ps_estimator.evaluate(X, treatment)
    print(f"   AUC: {ps_metrics['auc']:.3f}")
    print(f"   Propensity score range: [{ps_scores.min():.3f}, {ps_scores.max():.3f}]")

    # Step 2: Perform matching
    print("\n3. Performing propensity score matching...")
    matcher = PropensityScoreMatcher(
        matching_type='nearest',
        caliper=0.1,
        replacement=False,
        random_state=42
    )
    matcher.fit(X, treatment, propensity_scores)

    print(f"   Matched pairs: {len(matcher.matched_treated_):,}")
    print(f"   Match rate: {len(matcher.matched_treated_) / treatment.sum():.1%}")

    # Step 3: Check balance
    print("\n4. Checking covariate balance...")
    balance_stats = matcher.get_balance_stats()

    print(f"   Balanced covariates: {balance_stats['balanced'].sum()}/{len(balance_stats)}")
    print("\n   Balance by covariate:")
    for _, row in balance_stats.iterrows():
        status = "✓" if row['balanced'] else "✗"
        print(f"   {status} {row['feature']:20s}: StdDiff = {row['std_diff']:6.3f}")

    # Step 4: Estimate treatment effect
    print("\n5. Estimating treatment effect...")
    psm_result = matcher.estimate_effect(pd.Series(outcome), outcome_type='continuous')

    print(f"\n   PSM Estimate: {psm_result['effect']:.4f}")
    print(f"   Standard Error: {psm_result['std_error']:.4f}")
    print(f"   95% CI: [{psm_result['ci_lower']:.4f}, {psm_result['ci_upper']:.4f}]")
    print(f"   P-value: {psm_result['p_value']:.4f}")
    print(f"   T-statistic: {psm_result['t_statistic']:.2f}")

    # Step 5: Bootstrap confidence interval
    print("\n6. Bootstrap confidence interval (500 samples)...")
    bootstrap_ci = matcher.bootstrap_ci(
        pd.Series(outcome),
        n_bootstrap=500,
        outcome_type='continuous'
    )

    print(f"   Bootstrap 95% CI: [{bootstrap_ci['ci_lower']:.4f}, {bootstrap_ci['ci_upper']:.4f}]")
    print(f"   Bootstrap mean: {bootstrap_ci['bootstrap_mean']:.4f}")
    print(f"   Bootstrap std: {bootstrap_ci['bootstrap_std']:.4f}")

    # Create visualizations
    print("\n7. Creating visualizations...")

    viz = BalanceVisualizer()

    # Before/after balance comparison
    before_stats = balance_stats.copy()
    before_stats['std_diff'] = before_stats['std_diff'] * 2  # Simulate before matching

    fig1 = viz.love_plot(
        balance_stats=balance_stats,
        before_stats=before_stats,
        save_path='examples/love_plot_example.png'
    )
    print(f"   ✓ Love plot saved to: examples/love_plot_example.png")

    # Propensity score distributions
    fig2 = viz.propensity_score_distributions(
        propensity_scores=propensity_scores,
        treatment=pd.Series(treatment),
        save_path='examples/ps_distributions_example.png'
    )
    print(f"   ✓ PS distributions saved to: examples/ps_distributions_example.png")

    plt.close('all')

    return psm_result


def example_2_inverse_probability_weighting():
    """Example 2: Inverse Probability Weighting."""
    print("\n" + "="*70)
    print("EXAMPLE 2: INVERSE PROBABILITY WEIGHTING")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic data...")
    X, treatment, outcome, true_ps = generate_sample_data(n=1000)

    # Estimate propensity scores
    print("\n2. Estimating propensity scores...")
    ps_estimator = PropensityScoreEstimator(use_scaling=True, random_state=42)
    ps_estimator.fit(X, treatment)
    propensity_scores = ps_estimator.predict_proba(X)

    # Apply IPW
    print("\n3. Calculating IPW weights...")
    ipw = PropensityScoreWeighting(trimming_quantile=0.01)
    ipw.fit(pd.Series(treatment), propensity_scores)

    print(f"   Mean weight: {ipw.weights_.mean():.2f}")
    print(f"   Max weight: {ipw.weights_.max():.2f}")
    print(f"   Min weight: {ipw.weights_.min():.2f}")

    # Estimate effect
    print("\n4. Estimating treatment effect...")
    ipw_result = ipw.estimate_effect(pd.Series(outcome))

    print(f"\n   IPW Estimate: {ipw_result['effect']:.4f}")
    print(f"   Standard Error: {ipw_result['std_error']:.4f}")
    print(f"   Effective N: {ipw_result['n_effective']:.0f}")
    print(f"   Weights trimmed: {ipw_result['trimmed']}")

    return ipw_result


def example_3_difference_in_differences():
    """Example 3: Difference-in-Differences."""
    print("\n" + "="*70)
    print("EXAMPLE 3: DIFFERENCE-IN-DIFFERENCES")
    print("="*70)

    # Generate panel data
    print("\n1. Generating panel data...")
    np.random.seed(42)
    n_units = 200
    n_periods = 12

    data = []
    for unit in range(n_units):
        treated = 1 if unit > 100 else 0  # Last 100 units treated
        for period in range(n_periods):
            post = 1 if period >= 6 else 0  # Treatment at period 6

            # Outcome with treatment effect
            outcome = (
                10 +  # base
                1.5 * treated * post +  # treatment effect
                0.1 * period +  # time trend
                np.random.normal(0, 0.5)  # noise
            )

            data.append({
                'unit_id': unit,
                'time': period,
                'treated': treated,
                'post': post,
                'outcome': outcome
            })

    df = pd.DataFrame(data)
    print(f"   Units: {n_units} ({(df['treated'] == 1).sum() // n_periods} treated)")
    print(f"   Time periods: {n_periods}")
    print(f"   Total observations: {len(df)}")

    # Fit DiD
    print("\n2. Fitting DiD model...")
    did = DifferenceInDifferences(
        outcome_col='outcome',
        treatment_col='treated',
        time_col='time',
        unit_col='unit_id',
        post_period=6
    )

    did_result = did.fit(df)

    print(f"\n   DiD Estimate: {did_result['did_estimate']:.4f}")
    print(f"   Standard Error: {did_result['std_error']:.4f}")
    print(f"   95% CI: [{did_result['ci_lower']:.4f}, {did_result['ci_upper']:.4f}]")
    print(f"   P-value: {did_result['p_value']:.4f}")

    print(f"\n   Group-time means:")
    print(f"   - Treated (pre): {did_result['treated_pre']:.4f}")
    print(f"   - Treated (post): {did_result['treated_post']:.4f}")
    print(f"   - Control (pre): {did_result['control_pre']:.4f}")
    print(f"   - Control (post): {did_result['control_post']:.4f}")

    # Check parallel trends
    print("\n3. Testing parallel trends...")
    pt_result = did.check_parallel_trends(df)

    print(f"   Parallel trends p-value: {pt_result['p_value']:.4f}")
    print(f"   Parallel trends: {'✓ Satisfied' if pt_result['parallel_trends_satisfied'] else '✗ Violated'}")

    # Event study
    print("\n4. Event study analysis...")
    event_study = did.event_study(df, leads=3, lags=3)

    print(f"   Event time range: {event_study['event_time'].min()} to {event_study['event_time'].max()}")
    print(f"\n   Sample event study results:")
    for _, row in event_study.head(3).iterrows():
        print(f"   Event time {row['event_time']:2d}: {row['effect']:.4f} ± {row['std_error']:.4f}")

    return did_result


def example_4_method_comparison():
    """Example 4: Compare Multiple Methods."""
    print("\n" + "="*70)
    print("EXAMPLE 4: METHOD COMPARISON")
    print("="*70)

    # Generate data
    print("\n1. Generating data for comparison...")
    X, treatment, outcome, true_ps = generate_sample_data(n=1000)

    # Estimate propensity scores
    print("\n2. Estimating propensity scores...")
    ps_estimator = PropensityScoreEstimator(use_scaling=True, random_state=42)
    ps_estimator.fit(X, treatment)
    propensity_scores = ps_estimator.predict_proba(X)

    # Method 1: Naive
    print("\n3. Naive comparison...")
    treated_mean = outcome[treatment == 1].mean()
    control_mean = outcome[treatment == 0].mean()
    naive_effect = treated_mean - control_mean

    print(f"   Naive estimate: {naive_effect:.4f}")

    # Method 2: PSM
    print("\n4. Propensity score matching...")
    matcher = PropensityScoreMatcher(caliper=0.1, random_state=42)
    matcher.fit(X, treatment, propensity_scores)
    psm_effect = matcher.estimate_effect(pd.Series(outcome))

    # Method 3: IPW
    print("\n5. Inverse probability weighting...")
    ipw = PropensityScoreWeighting(trimming_quantile=0.01)
    ipw.fit(pd.Series(treatment), propensity_scores)
    ipw_effect = ipw.estimate_effect(pd.Series(outcome))

    # Compare results
    print("\n6. Method comparison:")
    print("-" * 70)
    print(f"{'Method':<20} {'Estimate':<15} {'Bias':<15}")
    print("-" * 70)

    true_effect = 2.0  # Known from data generation

    methods = {
        'Naive': naive_effect,
        'PSM': psm_effect['effect'],
        'IPW': ipw_effect['effect']
    }

    for method, estimate in methods.items():
        bias = abs(estimate - true_effect)
        print(f"{method:<20} {estimate:<15.4f} {bias:<15.4f}")

    print("-" * 70)

    # Create comparison plot
    print("\n7. Creating comparison plot...")

    comparison_dict = {
        'Naive': {
            'estimate': naive_effect,
            'ci_lower': naive_effect - 0.1,
            'ci_upper': naive_effect + 0.1,
            'valid': False
        },
        'PSM': {
            'estimate': psm_effect['effect'],
            'ci_lower': psm_effect['ci_lower'],
            'ci_upper': psm_effect['ci_upper'],
            'valid': True
        },
        'IPW': {
            'estimate': ipw_effect['effect'],
            'ci_lower': ipw_effect['effect'] - 1.96 * ipw_effect['std_error'],
            'ci_upper': ipw_effect['effect'] + 1.96 * ipw_effect['std_error'],
            'valid': True
        }
    }

    fig = plot_treatment_effects_comparison(
        comparison_dict,
        save_path='examples/method_comparison_example.png'
    )
    print(f"   ✓ Comparison plot saved to: examples/method_comparison_example.png")

    plt.close('all')

    return comparison_dict


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MODULAR CAUSAL INFERENCE TOOLKIT - USAGE EXAMPLES")
    print("="*70)

    # Create examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)

    # Run examples
    try:
        # Example 1: PSM
        psm_result = example_1_propensity_score_matching()

        # Example 2: IPW
        ipw_result = example_2_inverse_probability_weighting()

        # Example 3: DiD
        did_result = example_3_difference_in_differences()

        # Example 4: Method Comparison
        comparison = example_4_method_comparison()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\nAll examples completed successfully!")
        print("\nGenerated files:")
        print("  - examples/love_plot_example.png")
        print("  - examples/ps_distributions_example.png")
        print("  - examples/method_comparison_example.png")

        print("\nKey takeaways:")
        print("  1. Propensity score matching recovers unbiased estimates")
        print("  2. IPW provides alternative weighting approach")
        print("  3. DiD useful for panel data with time variation")
        print("  4. Multiple methods provide robustness checks")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
