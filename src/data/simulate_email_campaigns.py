"""
Email Campaign Simulation with Confounding and True Causal Effect

This script simulates realistic email marketing campaigns where:
1. Email assignment is CONFOUNDED (not random) - based on customer characteristics
2. Emails have a TRUE causal effect on purchase probability
3. We track the ground truth for validation

The confounding creates selection bias that naive analyses would miss.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


def calculate_email_assignment_probability(row):
    """
    Calculate the probability that a customer receives an email in a given week.
    This creates realistic CONFOUNDING - assignment based on characteristics.

    Rules (not mutually exclusive):
    - Recent purchasers (bought in last 2 weeks): +60% chance
    - Frequent buyers (>10 past purchases): +50% chance
    - High-value customers (AOV > 75th percentile): +55% chance
    - Lapsed customers (30-60 days since purchase): +40% chance
    - Random baseline: 15% base chance

    Parameters:
    -----------
    row : pandas.Series
        Customer-week observation

    Returns:
    --------
    float
        Probability of receiving email (0-1)
    """
    probability = 0.15  # Base random probability

    # Recent purchasers (bought in last 2 weeks)
    if row['days_since_last_purchase'] <= 14:
        probability += 0.60

    # Frequent buyers
    if row['total_past_purchases'] > 10:
        probability += 0.50

    # High-value customers (we'll update AOV threshold later based on data)
    # For now, use avg_order_value > 20 as proxy
    if row['avg_order_value'] > 20:
        probability += 0.55

    # Lapsed customers (30-60 days since purchase)
    if 30 <= row['days_since_last_purchase'] <= 60:
        probability += 0.40

    # Cap at 95% to avoid deterministic assignment
    return min(probability, 0.95)


def calculate_true_purchase_probability(row, received_email, ground_truth):
    """
    Calculate the TRUE purchase probability including causal effect of email.

    Base purchase probability depends on customer characteristics.
    Email has an ADDITIVE effect of +10 percentage points.
    Effect is STRONGER for medium RFM scores (interaction).

    Parameters:
    -----------
    row : pandas.Series
        Customer-week observation
    received_email : bool
        Whether customer received email
    ground_truth : dict
        Ground truth parameters

    Returns:
    --------
    float
        True purchase probability (0-1)
    """
    # Base probability from customer characteristics
    # Higher RFM = higher base probability
    base_prob = 0.05 + (row['rfm_score'] / 15.0) * 0.15

    # Recent purchases increase base probability
    if row['days_since_last_purchase'] <= 7:
        base_prob += 0.15
    elif row['days_since_last_purchase'] <= 30:
        base_prob += 0.05

    # More past purchases = higher base probability (but diminishing returns)
    base_prob += min(row['total_past_purchases'] * 0.01, 0.10)

    # Customer tenure has U-shaped effect (new and loyal customers buy more)
    if row['customer_tenure_weeks'] <= 4:
        base_prob += 0.05
    elif row['customer_tenure_weeks'] >= 40:
        base_prob += 0.03

    # Add some random noise to base probability
    base_prob += np.random.normal(0, 0.02)
    base_prob = max(0, min(base_prob, 0.8))  # Bound between 0 and 80%

    # Causal effect of email
    # Stronger effect for medium RFM scores (8-12)
    if received_email:
        # Base email effect: +10 percentage points
        email_effect = ground_truth['base_email_effect']

        # Interaction: stronger effect for medium RFM
        if 8 <= row['rfm_score'] <= 12:
            email_effect += ground_truth['interaction_effect_medium_rfm']
        elif row['rfm_score'] > 12:
            email_effect += ground_truth['interaction_effect_high_rfm']
        else:  # Low RFM score
            email_effect += ground_truth['interaction_effect_low_rfm']

        # Add randomness to email effect
        email_effect += np.random.normal(0, ground_truth['email_effect_noise'])

        return max(0, min(base_prob + email_effect, 0.95))
    else:
        return base_prob


def simulate_email_campaigns(panel_df, random_seed=42):
    """
    Simulate email campaigns with confounding and true causal effects.

    Parameters:
    -----------
    panel_df : pandas.DataFrame
        Customer-week panel dataset
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame
        Panel with email simulation results
    ground_truth : dict
        Ground truth parameters for validation
    """
    print("=" * 70)
    print("EMAIL CAMPAIGN SIMULATION WITH CONFOUNDING")
    print("=" * 70)
    print(f"\nRandom Seed: {random_seed}")
    print(f"Input Panel Shape: {panel_df.shape}")

    # Set random seed
    np.random.seed(random_seed)

    # Define ground truth parameters (kept secret for validation!)
    ground_truth = {
        'base_email_effect': 0.10,  # Email increases purchase prob by 10 percentage points
        'interaction_effect_medium_rfm': 0.05,  # Stronger effect for RFM 8-12
        'interaction_effect_high_rfm': 0.02,    # Weaker effect for RFM > 12
        'interaction_effect_low_rfm': -0.03,    # Negative effect for low RFM (fatigue)
        'email_effect_noise': 0.02,  # Random variation in email effect
        'confounding_rules': 'assignment_based_on_customer_characteristics',
        'description': 'Email effect stronger for medium RFM scores (8-12)'
    }

    print("\n" + "=" * 70)
    print("GROUND TRUTH (for validation later)")
    print("=" * 70)
    print(f"Base Email Effect: +{ground_truth['base_email_effect']*100:.1f} percentage points")
    print(f"Interaction Effect (RFM 8-12): +{ground_truth['interaction_effect_medium_rfm']*100:.1f} pp")
    print(f"Interaction Effect (RFM >12): +{ground_truth['interaction_effect_high_rfm']*100:.1f} pp")
    print(f"Interaction Effect (RFM <8): {ground_truth['interaction_effect_low_rfm']*100:.1f} pp")

    # Create a copy for simulation
    sim_df = panel_df.copy()

    print("\n" + "=" * 70)
    print("STEP 1: Email Assignment (CONFOUNDED)")
    print("=" * 70)

    # Calculate email assignment probabilities
    print("\nCalculating email assignment probabilities...")
    print("Confounding rules:")
    print("  - Recent purchasers (≤14 days): +60%")
    print("  - Frequent buyers (>10 purchases): +50%")
    print("  - High-value customers (AOV > £20): +55%")
    print("  - Lapsed customers (30-60 days): +40%")
    print("  - Base rate: 15%")

    sim_df['email_assignment_probability'] = sim_df.apply(
        calculate_email_assignment_probability, axis=1
    )

    # Sample email assignment
    sim_df['received_email'] = (
        np.random.random(len(sim_df)) < sim_df['email_assignment_probability']
    )

    email_rate = sim_df['received_email'].mean()
    print(f"\nActual email send rate: {email_rate:.1%}")

    # Show assignment by customer characteristics
    print("\nEmail assignment by characteristics:")
    print(f"  Recent buyers: {sim_df[sim_df['days_since_last_purchase'] <= 14]['received_email'].mean():.1%}")
    print(f"  Frequent buyers: {sim_df[sim_df['total_past_purchases'] > 10]['received_email'].mean():.1%}")
    print(f"  High AOV: {sim_df[sim_df['avg_order_value'] > 20]['received_email'].mean():.1%}")
    print(f"  Lapsed: {sim_df[(sim_df['days_since_last_purchase'] >= 30) & (sim_df['days_since_last_purchase'] <= 60)]['received_email'].mean():.1%}")
    print(f"  Other: {sim_df[~sim_df.index.isin(
        sim_df[(sim_df['days_since_last_purchase'] <= 14) |
               (sim_df['total_past_purchases'] > 10) |
               (sim_df['avg_order_value'] > 20) |
               ((sim_df['days_since_last_purchase'] >= 30) & (sim_df['days_since_last_purchase'] <= 60))
    ].index)]['received_email'].mean():.1%}")

    print("\n" + "=" * 70)
    print("STEP 2: True Purchase Probability (with causal effect)")
    print("=" * 70)

    # Calculate true purchase probabilities
    print("\nCalculating purchase probabilities with causal effect...")
    sim_df['true_purchase_probability'] = sim_df.apply(
        lambda row: calculate_true_purchase_probability(
            row, row['received_email'], ground_truth
        ), axis=1
    )

    # Sample actual purchases
    print("Sampling actual purchases...")
    sim_df['purchased_this_week_observed'] = (
        np.random.random(len(sim_df)) < sim_df['true_purchase_probability']
    )

    observed_purchase_rate = sim_df['purchased_this_week_observed'].mean()
    print(f"\nObserved purchase rate: {observed_purchase_rate:.1%}")

    # Show purchase rates by email receipt (OBSERVED - confounded!)
    email_group = sim_df.groupby('received_email')['purchased_this_week_observed'].agg(['mean', 'count'])
    print("\nObserved purchase rates by email receipt (CONFOUNDED):")
    print(f"  No email: {email_group.loc[False, 'mean']:.1%} (n={email_group.loc[False, 'count']:,})")
    print(f"  Received email: {email_group.loc[True, 'mean']:.1%} (n={email_group.loc[True, 'count']:,})")

    naive_effect = email_group.loc[True, 'mean'] - email_group.loc[False, 'mean']
    print(f"\nNaive observed effect: {naive_effect:.1%} (BIASED due to confounding!)")

    print("\n" + "=" * 70)
    print("STEP 3: Ground Truth Validation")
    print("=" * 70)

    # Calculate ground truth effect (what we would see with random assignment)
    print("\nCalculating ground truth effect with random assignment...")

    # For each customer, create a counterfactual: what if they didn't receive email?
    sim_df_counterfactual = sim_df.copy()
    sim_df_counterfactual['received_email'] = False

    sim_df_counterfactual['true_purchase_prob_if_no_email'] = sim_df_counterfactual.apply(
        lambda row: calculate_true_purchase_probability(
            row, False, ground_truth
        ), axis=1
    )

    # Calculate individual treatment effects
    sim_df['true_purchase_prob_if_no_email'] = sim_df_counterfactual['true_purchase_prob_if_no_email']
    sim_df['individual_treatment_effect'] = (
        sim_df['true_purchase_probability'] - sim_df['true_purchase_prob_if_no_email']
    )

    # Average Treatment Effect (ATE)
    ate = sim_df['individual_treatment_effect'].mean()
    print(f"\nTrue Average Treatment Effect (ATE): {ate:.1%}")
    print(f"Expected ATE (from ground truth): {ground_truth['base_email_effect']:.1%}")

    # Heterogeneous effects by RFM score
    print("\nTrue effect by RFM score:")
    for rfm_range, label in [(range(3, 8), 'Low (3-7)'),
                             (range(8, 13), 'Medium (8-12)'),
                             (range(13, 16), 'High (13-15)')]:
        mask = sim_df['rfm_score'].isin(rfm_range)
        if mask.sum() > 0:
            effect = sim_df[mask]['individual_treatment_effect'].mean()
            print(f"  {label}: {effect:.1%} (n={mask.sum():,})")

    print("\n" + "=" * 70)
    print("CONFOUNDING VERIFICATION")
    print("=" * 70)

    # Show that email assignment is correlated with customer characteristics
    print("\nEmail assignment is correlated with customer characteristics:")
    print("This is CONFOUNDING - makes naive comparisons biased!")

    corr_features = ['rfm_score', 'days_since_last_purchase', 'total_past_purchases',
                     'avg_order_value', 'customer_tenure_weeks']

    for feature in corr_features:
        corr = sim_df['received_email'].corr(sim_df[feature])
        print(f"  corr(received_email, {feature}): {corr:+.3f}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput shape: {sim_df.shape}")
    print(f"New columns added:")
    print("  - received_email: Whether customer received email")
    print("  - email_assignment_probability: Prob. of receiving email")
    print("  - true_purchase_probability: Actual purchase probability")
    print("  - purchased_this_week_observed: Binary outcome")
    print("  - true_purchase_prob_if_no_email: Counterfactual")
    print("  - individual_treatment_effect: Individual causal effect")

    return sim_df, ground_truth


def save_simulation_results(sim_df, ground_truth, output_dir):
    """
    Save simulation results and ground truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save simulated dataset
    output_path = output_dir / 'simulated_email_campaigns.csv'
    sim_df.to_csv(output_path, index=False)
    print(f"\n✅ Simulated dataset saved to: {output_path}")

    # Save ground truth
    ground_truth_path = output_dir / 'ground_truth.json'
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"✅ Ground truth saved to: {ground_truth_path}")

    # Save summary statistics
    summary = {
        'total_observations': len(sim_df),
        'unique_customers': sim_df['CustomerID'].nunique(),
        'email_send_rate': float(sim_df['received_email'].mean()),
        'observed_purchase_rate': float(sim_df['purchased_this_week_observed'].mean()),
        'naive_email_effect': float(
            sim_df[sim_df['received_email']]['purchased_this_week_observed'].mean() -
            sim_df[~sim_df['received_email']]['purchased_this_week_observed'].mean()
        ),
        'true_ate': float(sim_df['individual_treatment_effect'].mean()),
        'confounding_detected': True,
        'random_seed': 42
    }

    summary_path = output_dir / 'simulation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary statistics saved to: {summary_path}")

    return output_path, ground_truth_path, summary_path


if __name__ == "__main__":
    # Load panel dataset
    print("\nLoading panel dataset...")
    panel_df = pd.read_csv('data/processed/customer_week_panel.csv')
    print(f"Loaded panel: {panel_df.shape}")

    # Run simulation
    sim_df, ground_truth = simulate_email_campaigns(panel_df, random_seed=42)

    # Save results
    output_path, gt_path, summary_path = save_simulation_results(
        sim_df, ground_truth, 'data/processed'
    )

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE - READY FOR CAUSAL INFERENCE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Open notebooks/02_email_campaign_simulation.ipynb to understand the simulation")
    print("2. Use the simulated data to test causal inference methods")
    print("3. Compare naive estimates vs. causal estimates vs. ground truth")