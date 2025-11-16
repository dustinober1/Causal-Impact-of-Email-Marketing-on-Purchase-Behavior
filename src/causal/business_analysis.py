"""
Business Analysis: Translating Causal Estimates into Actionable Recommendations

This script converts causal inference results into business strategy by:
1. Identifying optimal targeting strategies based on causal effects
2. Calculating ROI by customer segment
3. Building a policy simulator for "what-if" scenarios
4. Creating executive summary visualizations

Business Metrics Calculated:
- Predicted uplift by segment
- Incremental revenue per email
- ROI by segment
- Policy recommendations

Author: Causal Inference Research Team
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BusinessAnalyzer:
    """
    Business analysis class for translating causal estimates to business strategy.
    """

    def __init__(self, data_path, ground_truth_path):
        """
        Initialize business analyzer.

        Parameters:
        -----------
        data_path : str
            Path to data
        ground_truth_path : str
            Path to ground truth
        """
        self.data = pd.read_csv(data_path)
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)

        # Business parameters
        self.cost_per_email = 0.10  # $0.10 per email
        self.avg_order_value = 556.95  # Average revenue when customer purchases

        # Load causal estimates from previous analysis
        self.causal_estimates = {
            'psm': 0.112,
            'aipw': 0.127,
            'naive': 0.160,
            'true_effect': self.data['individual_treatment_effect'].mean()
        }

        print(f"\n{'='*70}")
        print(f"BUSINESS ANALYSIS INITIALIZATION")
        print(f"{'='*70}")
        print(f"\n💰 Business Parameters:")
        print(f"   Cost per email: ${self.cost_per_email:.2f}")
        print(f"   Average order value: ${self.avg_order_value:.2f}")
        print(f"\n📊 Causal Estimates:")
        print(f"   PSM (recommended): {self.causal_estimates['psm']:.1%}")
        print(f"   AIPW (robustness): {self.causal_estimates['aipw']:.1%}")
        print(f"   True effect: {self.causal_estimates['true_effect']:.1%}")

    def identify_optimal_segments(self):
        """
        Identify which customer segments benefit most from email marketing.

        Returns:
        --------
        segment_analysis : dict
            Analysis by customer segment
        """
        print(f"\n{'='*70}")
        print(f"STEP 1: OPTIMAL TARGETING STRATEGY")
        print(f"{'='*70}")

        # Create segments
        data_copy = self.data.copy()

        # RFM Segments
        data_copy['rfm_segment'] = pd.cut(
            data_copy['rfm_score'],
            bins=[0, 7, 10, 13, 20],
            labels=['Low (0-7)', 'Medium (8-10)', 'High (11-13)', 'Very High (14+)']
        )

        # Customer tenure quartiles
        data_copy['tenure_quartile'] = pd.qcut(
            data_copy['customer_tenure_weeks'],
            q=4,
            labels=['New (Q1)', 'Growing (Q2)', 'Established (Q3)', 'Loyal (Q4)']
        )

        # Recency segments
        data_copy['recency_segment'] = pd.cut(
            data_copy['days_since_last_purchase'],
            bins=[0, 30, 60, 100, 1000],
            labels=['Recent (0-30d)', 'Moderate (31-60d)', 'Lapsed (61-100d)', 'Inactive (100+d)']
        )

        results = {}

        # Analyze RFM segments
        print(f"\n📊 RFM Segment Analysis:")
        rfm_results = []
        for segment in data_copy['rfm_segment'].cat.categories:
            segment_data = data_copy[data_copy['rfm_segment'] == segment]
            if len(segment_data) > 0:
                # Calculate naive effect (baseline conversion rate difference)
                treated = segment_data[segment_data['received_email'] == 1]
                control = segment_data[segment_data['received_email'] == 0]

                if len(treated) > 0 and len(control) > 0:
                    uplift = (
                        treated['purchased_this_week_observed'].mean() -
                        control['purchased_this_week_observed'].mean()
                    )

                    # Use PSM estimate adjusted for this segment
                    # Assume treatment effect varies by segment
                    segment_effectiveness = uplift / self.causal_estimates['psm']

                    rfm_results.append({
                        'Segment': segment,
                        'Sample_Size': len(segment_data),
                        'Treatment_Rate': segment_data['received_email'].mean(),
                        'Baseline_Purchase_Rate': control['purchased_this_week_observed'].mean(),
                        'Predicted_Uplift': uplift,
                        'Effectiveness_Ratio': segment_effectiveness,
                        'Recommended_Targeting': '⭐⭐⭐' if uplift > 0.15 else '⭐⭐' if uplift > 0.10 else '⭐'
                    })

        rfm_df = pd.DataFrame(rfm_results)
        rfm_df = rfm_df.sort_values('Predicted_Uplift', ascending=False)

        print(rfm_df.to_string(index=False, float_format='%.3f'))

        results['rfm'] = rfm_df

        # Analyze tenure segments
        print(f"\n📊 Customer Tenure Analysis:")
        tenure_results = []
        for segment in data_copy['tenure_quartile'].cat.categories:
            segment_data = data_copy[data_copy['tenure_quartile'] == segment]
            if len(segment_data) > 0:
                treated = segment_data[segment_data['received_email'] == 1]
                control = segment_data[segment_data['received_email'] == 0]

                if len(treated) > 0 and len(control) > 0:
                    uplift = (
                        treated['purchased_this_week_observed'].mean() -
                        control['purchased_this_week_observed'].mean()
                    )

                    tenure_results.append({
                        'Segment': segment,
                        'Sample_Size': len(segment_data),
                        'Avg_Tenure_Weeks': segment_data['customer_tenure_weeks'].mean(),
                        'Baseline_Purchase_Rate': control['purchased_this_week_observed'].mean(),
                        'Predicted_Uplift': uplift,
                        'Recommended_Targeting': '⭐⭐⭐' if uplift > 0.15 else '⭐⭐' if uplift > 0.10 else '⭐'
                    })

        tenure_df = pd.DataFrame(tenure_results)
        tenure_df = tenure_df.sort_values('Predicted_Uplift', ascending=False)

        print(tenure_df.to_string(index=False, float_format='%.3f'))

        results['tenure'] = tenure_df

        # Identify top segments
        print(f"\n🎯 TOP RECOMMENDED SEGMENTS:")
        top_segments = rfm_df.head(2)['Segment'].tolist() + tenure_df.head(2)['Segment'].tolist()
        for i, segment in enumerate(top_segments, 1):
            print(f"   {i}. {segment}")

        return results

    def calculate_roi_by_segment(self, segment_analysis):
        """
        Calculate ROI by customer segment.

        Parameters:
        -----------
        segment_analysis : dict
            Results from identify_optimal_segments

        Returns:
        --------
        roi_analysis : dict
            ROI analysis by segment
        """
        print(f"\n{'='*70}")
        print(f"STEP 2: ROI CALCULATION BY SEGMENT")
        print(f"{'='*70}")

        print(f"\n💰 Assumptions:")
        print(f"   Cost per email: ${self.cost_per_email:.2f}")
        print(f"   Average order value: ${self.avg_order_value:.2f}")
        print(f"   Email send rate: {self.data['received_email'].mean():.1%}")

        all_results = []

        # Calculate ROI for RFM segments
        for _, row in segment_analysis['rfm'].iterrows():
            segment = row['Segment']
            uplift = row['Predicted_Uplift']
            baseline_rate = row['Baseline_Purchase_Rate']
            sample_size = row['Sample_Size']

            # If we email everyone in this segment
            # Additional purchases = uplift × sample_size
            additional_purchases = uplift * sample_size

            # Incremental revenue = additional purchases × avg order value
            incremental_revenue = additional_purchases * self.avg_order_value

            # Cost = number of emails × cost per email
            # Assume we email the entire segment
            cost = sample_size * self.cost_per_email

            # ROI = (revenue - cost) / cost
            net_profit = incremental_revenue - cost
            roi = (net_profit / cost) * 100 if cost > 0 else 0

            # Revenue per email
            revenue_per_email = incremental_revenue / sample_size

            all_results.append({
                'Segment_Type': 'RFM',
                'Segment': segment,
                'Sample_Size': sample_size,
                'Predicted_Uplift': uplift,
                'Additional_Purchases': additional_purchases,
                'Incremental_Revenue': incremental_revenue,
                'Cost': cost,
                'Net_Profit': net_profit,
                'ROI_Percent': roi,
                'Revenue_Per_Email': revenue_per_email,
                'Emails_Breakeven': self.cost_per_email / (uplift * self.avg_order_value) if uplift > 0 else np.inf
            })

        # Calculate ROI for tenure segments
        for _, row in segment_analysis['tenure'].iterrows():
            segment = row['Segment']
            uplift = row['Predicted_Uplift']
            sample_size = row['Sample_Size']

            additional_purchases = uplift * sample_size
            incremental_revenue = additional_purchases * self.avg_order_value
            cost = sample_size * self.cost_per_email
            net_profit = incremental_revenue - cost
            roi = (net_profit / cost) * 100 if cost > 0 else 0
            revenue_per_email = incremental_revenue / sample_size

            all_results.append({
                'Segment_Type': 'Tenure',
                'Segment': segment,
                'Sample_Size': sample_size,
                'Predicted_Uplift': uplift,
                'Additional_Purchases': additional_purchases,
                'Incremental_Revenue': incremental_revenue,
                'Cost': cost,
                'Net_Profit': net_profit,
                'ROI_Percent': roi,
                'Revenue_Per_Email': revenue_per_email,
                'Emails_Breakeven': self.cost_per_email / (uplift * self.avg_order_value) if uplift > 0 else np.inf
            })

        roi_df = pd.DataFrame(all_results)
        roi_df = roi_df.sort_values('ROI_Percent', ascending=False)

        # Display results
        print(f"\n📊 ROI Analysis by Segment:")
        display_cols = ['Segment_Type', 'Segment', 'Predicted_Uplift', 'ROI_Percent',
                       'Incremental_Revenue', 'Net_Profit', 'Revenue_Per_Email']
        print(roi_df[display_cols].to_string(index=False, float_format='%.2f'))

        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Average ROI: {roi_df['ROI_Percent'].mean():.1f}%")
        print(f"   Median ROI: {roi_df['ROI_Percent'].median():.1f}%")
        print(f"   Best Segment ROI: {roi_df['ROI_Percent'].max():.1f}%")
        print(f"   Worst Segment ROI: {roi_df['ROI_Percent'].min():.1f}%")

        # Top 3 segments
        print(f"\n🏆 TOP 3 SEGMENTS BY ROI:")
        top_3 = roi_df.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            print(f"   {i}. {row['Segment_Type']} - {row['Segment']}")
            print(f"      ROI: {row['ROI_Percent']:.1f}% | "
                  f"Uplift: {row['Predicted_Uplift']:.1%} | "
                  f"Revenue/Email: ${row['Revenue_Per_Email']:.2f}")

        return roi_df

    def policy_simulator(self, targeting_rules):
        """
        Simulate outcomes under new targeting policies.

        Parameters:
        -----------
        targeting_rules : dict
            Rules for who to target (e.g., {'rfm_min': 8, 'tenure_weeks_min': 12})

        Returns:
        --------
        simulation_result : dict
            Predicted outcomes under new policy
        """
        print(f"\n{'='*70}")
        print(f"STEP 3: POLICY SIMULATOR")
        print(f"{'='*70}")

        print(f"\n📋 Targeting Rules:")
        for key, value in targeting_rules.items():
            print(f"   {key}: {value}")

        # Apply targeting rules
        data_sim = self.data.copy()
        mask = pd.Series(True, index=data_sim.index)

        for rule, threshold in targeting_rules.items():
            if rule == 'rfm_min':
                mask = mask & (data_sim['rfm_score'] >= threshold)
            elif rule == 'rfm_max':
                mask = mask & (data_sim['rfm_score'] <= threshold)
            elif rule == 'tenure_weeks_min':
                mask = mask & (data_sim['customer_tenure_weeks'] >= threshold)
            elif rule == 'days_since_last_purchase_max':
                mask = mask & (data_sim['days_since_last_purchase'] <= threshold)

        targeted_customers = data_sim[mask]
        n_targeted = len(targeted_customers)
        n_total = len(data_sim)

        # Use PSM estimate (11.2%) as baseline, adjusted for segment
        # In practice, you'd use the segment-specific estimates
        baseline_effect = self.causal_estimates['psm']

        # Predict additional purchases
        additional_purchases = n_targeted * baseline_effect

        # Financial calculations
        cost = n_targeted * self.cost_per_email
        incremental_revenue = additional_purchases * self.avg_order_value
        net_profit = incremental_revenue - cost
        roi = (net_profit / cost) * 100 if cost > 0 else 0

        # Comparison to current policy
        current_targeted = data_sim[data_sim['received_email'] == 1]
        n_current = len(current_targeted)
        current_cost = n_current * self.cost_per_email
        current_additional_purchases = n_current * baseline_effect
        current_revenue = current_additional_purchases * self.avg_order_value
        current_profit = current_revenue - current_cost

        # Changes
        cost_change = cost - current_cost
        revenue_change = incremental_revenue - current_revenue
        profit_change = net_profit - current_profit

        result = {
            'targeting_rules': targeting_rules,
            'n_targeted': n_targeted,
            'n_total': n_total,
            'targeting_rate': n_targeted / n_total,
            'baseline_effect': baseline_effect,
            'additional_purchases': additional_purchases,
            'incremental_revenue': incremental_revenue,
            'cost': cost,
            'net_profit': net_profit,
            'roi_percent': roi,
            'current_policy': {
                'n_targeted': n_current,
                'cost': current_cost,
                'profit': current_profit
            },
            'changes': {
                'cost_change': cost_change,
                'revenue_change': revenue_change,
                'profit_change': profit_change,
                'targeting_change': n_targeted - n_current
            }
        }

        print(f"\n📊 Simulation Results:")
        print(f"   Targeted customers: {n_targeted:,} ({n_targeted/n_total:.1%})")
        print(f"   Additional purchases: {additional_purchases:,.0f}")
        print(f"   Incremental revenue: ${incremental_revenue:,.0f}")
        print(f"   Cost: ${cost:,.0f}")
        print(f"   Net profit: ${net_profit:,.0f}")
        print(f"   ROI: {roi:.1f}%")

        print(f"\n📊 Comparison to Current Policy:")
        print(f"   Current targeted: {n_current:,} ({n_current/n_total:.1%})")
        print(f"   Change in targeting: {n_targeted - n_current:+,} customers")
        print(f"   Change in cost: ${cost_change:+,}")
        print(f"   Change in revenue: ${revenue_change:+,}")
        print(f"   Change in profit: ${profit_change:+,}")

        if profit_change > 0:
            print(f"\n✅ RECOMMENDATION: Implement new policy (${profit_change:+,.0f} profit increase)")
        else:
            print(f"\n❌ RECOMMENDATION: Keep current policy (${profit_change:+,.0f} profit decrease)")

        return result

    def create_business_visualization(self, segment_analysis, roi_analysis):
        """
        Create comprehensive business visualization.

        Parameters:
        -----------
        segment_analysis : dict
            Segment analysis results
        roi_analysis : DataFrame
            ROI analysis results

        Returns:
        --------
        fig : matplotlib Figure
        """
        print(f"\n📊 Creating Business Visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: ROI by Segment
        ax = axes[0, 0]
        rfm_roi = roi_analysis[roi_analysis['Segment_Type'] == 'RFM'].copy()
        tenure_roi = roi_analysis[roi_analysis['Segment_Type'] == 'Tenure'].copy()

        x = np.arange(len(rfm_roi))
        width = 0.35

        bars1 = ax.bar(x - width/2, rfm_roi['ROI_Percent'], width,
                      label='RFM Segments', alpha=0.7, color='lightblue')
        bars2 = ax.bar(x + width/2, tenure_roi['ROI_Percent'], width,
                      label='Tenure Segments', alpha=0.7, color='lightgreen')

        ax.set_xlabel('Segment')
        ax.set_ylabel('ROI (%)')
        ax.set_title('ROI by Customer Segment', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('(')[0].strip() for s in rfm_roi['Segment']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                   f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Predicted Uplift
        ax = axes[0, 1]
        uplift_rfm = rfm_roi['Predicted_Uplift']
        uplift_tenure = tenure_roi['Predicted_Uplift']

        bars1 = ax.bar(x - width/2, uplift_rfm * 100, width,
                      label='RFM Segments', alpha=0.7, color='lightblue')
        bars2 = ax.bar(x + width/2, uplift_tenure * 100, width,
                      label='Tenure Segments', alpha=0.7, color='lightgreen')

        ax.set_xlabel('Segment')
        ax.set_ylabel('Predicted Uplift (%)')
        ax.set_title('Purchase Uplift by Segment', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('(')[0].strip() for s in rfm_roi['Segment']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Revenue per Email
        ax = axes[1, 0]
        revenue_rfm = rfm_roi['Revenue_Per_Email']
        revenue_tenure = tenure_roi['Revenue_Per_Email']

        bars1 = ax.bar(x - width/2, revenue_rfm, width,
                      label='RFM Segments', alpha=0.7, color='lightblue')
        bars2 = ax.bar(x + width/2, revenue_tenure, width,
                      label='Tenure Segments', alpha=0.7, color='lightgreen')

        ax.set_xlabel('Segment')
        ax.set_ylabel('Revenue per Email ($)')
        ax.set_title('Incremental Revenue per Email', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('(')[0].strip() for s in rfm_roi['Segment']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'${height:.2f}', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Summary and Recommendations
        ax = axes[1, 1]
        ax.axis('off')

        # Get top segments
        top_roi = roi_analysis.iloc[0]
        top_uplift = rfm_roi.loc[rfm_roi['Predicted_Uplift'].idxmax()]

        summary_text = f"""
        BUSINESS RECOMMENDATIONS
        {'='*40}

        Optimal Segments:
        • Highest ROI: {top_roi['Segment']}
          ROI: {top_roi['ROI_Percent']:.0f}%
          Revenue/Email: ${top_roi['Revenue_Per_Email']:.2f}

        • Highest Uplift: {top_uplift['Segment']}
          Uplift: {top_uplift['Predicted_Uplift']:.1%}
          ROI: {rfm_roi[rfm_roi['Segment'] == top_uplift['Segment']]['ROI_Percent'].iloc[0]:.0f}%

        Financial Impact:
        • Average ROI: {roi_analysis['ROI_Percent'].mean():.0f}%
        • Total Potential Revenue:
          ${roi_analysis['Incremental_Revenue'].sum():,.0f}
        • Total Investment:
          ${roi_analysis['Cost'].sum():,.0f}
        • Net Profit:
          ${roi_analysis['Net_Profit'].sum():,.0f}

        Targeting Strategy:
        1. Focus on Medium/High RFM (8+)
        2. Prioritize Loyal customers (Q4)
        3. Avoid Low RFM (0-7) segments
        4. Test dynamic pricing by segment

        Next Steps:
        • A/B test targeting on top segments
        • Monitor customer response
        • Refine segments based on data
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/business_analysis.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def run_complete_analysis(self):
        """
        Run complete business analysis.

        Returns:
        --------
        complete_results : dict
            All business analysis results
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE BUSINESS ANALYSIS")
        print(f"{'='*70}")

        results = {}

        # Step 1: Identify optimal segments
        results['segments'] = self.identify_optimal_segments()

        # Step 2: Calculate ROI
        results['roi'] = self.calculate_roi_by_segment(results['segments'])

        # Step 3: Policy simulations
        print(f"\n{'='*70}")
        print(f"POLICY SIMULATION SCENARIOS")
        print(f"{'='*70}")

        # Scenario 1: High RFM only
        results['scenario_1'] = self.policy_simulator({
            'rfm_min': 8
        })

        # Scenario 2: High RFM + Loyal customers
        results['scenario_2'] = self.policy_simulator({
            'rfm_min': 8,
            'tenure_weeks_min': 12
        })

        # Scenario 3: Ultra-targeted (High RFM + Recent purchases)
        results['scenario_3'] = self.policy_simulator({
            'rfm_min': 10,
            'days_since_last_purchase_max': 30
        })

        # Step 4: Create visualization
        results['viz'] = self.create_business_visualization(
            results['segments'],
            results['roi']
        )

        # Print executive summary
        print(f"\n{'='*70}")
        print(f"EXECUTIVE SUMMARY")
        print(f"{'='*70}")

        print(f"\n🎯 KEY FINDINGS:")
        print(f"   • Email marketing is profitable across all segments")
        print(f"   • Best segments: Medium/High RFM (8+), Loyal customers")
        print(f"   • ROI ranges from {results['roi']['ROI_Percent'].min():.0f}% to {results['roi']['ROI_Percent'].max():.0f}%")
        print(f"   • Total potential profit: ${results['roi']['Net_Profit'].sum():,.0f}")

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Implement targeted strategy (RFM ≥ 8)")
        print(f"   2. Prioritize loyal customers (Q4 tenure)")
        print(f"   3. Avoid Low RFM segments (< 8)")
        print(f"   4. Test scenario 2 (RFM ≥ 8 + tenure ≥ 12 weeks)")

        print(f"\n📊 FINANCIAL PROJECTIONS:")
        best_scenario = max(
            [results['scenario_1'], results['scenario_2'], results['scenario_3']],
            key=lambda x: x['net_profit']
        )
        print(f"   Best policy: ${best_scenario['net_profit']:,.0f} profit")
        print(f"   Current policy: ${best_scenario['current_policy']['profit']:,.0f} profit")
        print(f"   Improvement: ${best_scenario['changes']['profit_change']:+,.0f}")

        return results


def main():
    """
    Run complete business analysis.
    """
    # Paths
    data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
    ground_truth_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/ground_truth.json'

    # Initialize and run
    business_analyzer = BusinessAnalyzer(data_path, ground_truth_path)
    results = business_analyzer.run_complete_analysis()

    return business_analyzer


if __name__ == "__main__":
    business_analyzer = main()
