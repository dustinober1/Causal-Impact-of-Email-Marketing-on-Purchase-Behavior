"""
Streamlit Dashboard: Causal Impact of Email Marketing

A comprehensive interactive dashboard showcasing causal inference analysis
for email marketing effectiveness.

Tabs:
1. Overview: Project description and key findings
2. The Problem: Naive analysis and confounding
3. Causal Methods: Compare PSM, DiD, doubly robust
4. Results: Treatment effects with visualizations

Author: Causal Inference Research Team
Date: 2025-11-16
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure page
st.set_page_config(
    page_title="Email Marketing Causal Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
    .success {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading
@st.cache_data
def load_data():
    """Load all required data."""
    data_path = 'data/processed/data_with_propensity_scores.csv'
    ground_truth_path = 'data/processed/ground_truth.json'

    try:
        data = pd.read_csv(data_path)
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        true_effect = data['individual_treatment_effect'].mean()
        return data, ground_truth, true_effect
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def tab_overview():
    """Tab 1: Overview - Project description and key findings."""
    st.markdown('<h1 class="main-header">📊 Causal Impact of Email Marketing</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
    <strong>Project Overview:</strong> This dashboard showcases comprehensive causal inference analysis
    measuring the effectiveness of email marketing campaigns on customer purchase behavior using
    real-world e-commerce transaction data.
    </div>
    """, unsafe_allow_html=True)

    # Load data
    data, ground_truth, true_effect = load_data()

    if data is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Ground Truth Effect",
                value=f"{true_effect:.1%}",
                help="True causal effect from simulation"
            )

        with col2:
            st.metric(
                label="Sample Size",
                value=f"{len(data):,}",
                help="Total observations analyzed"
            )

        with col3:
            st.metric(
                label="Treatment Rate",
                value=f"{data['received_email'].mean():.1%}",
                help="Percentage receiving emails"
            )

        with col4:
            st.metric(
                label="Methods Compared",
                value="6",
                help="Different causal inference approaches"
            )

    st.markdown('<h2 class="subheader">🎯 Key Findings</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ What We Learned:**

        1. **Naive Analysis Fails**: Simple comparison shows 16.0% effect
           - **Reality**: True effect is only 9.5%
           - **Bias**: 68% overestimate due to confounding

        2. **Causal Methods Work**: PSM recovers 11.2% (closest to truth)
           - **Bias Reduction**: 74% improvement over naive
           - **Best Method**: Propensity Score Matching

        3. **Significant Heterogeneity**: Effects vary by customer segment
           - **Range**: 9.0% (Low RFM) to 18.6% (Loyal customers)
           - **Business Insight**: Target loyal/medium-RFM customers

        4. **Email Marketing is Profitable**:
           - **ROI**: 50,000% - 104,000% across segments
           - **Strategy**: Email 81.7% of customers (optimal)
           - **Expected Impact**: +$1.52M profit (+21.7%)
        """)

    with col2:
        st.markdown("""
        **📊 Methods Implemented:**

        | Method | Estimate | Bias | Valid? |
        |--------|----------|------|--------|
        | Naive | 16.0% | +6.5 pp | ❌ |
        | PSM | 11.2% | +1.7 pp | ✅ |
        | DiD | 0.5% | -9.3 pp | ❌ |
        | IPW | 13.6% | +4.1 pp | ✅ |
        | AIPW | 12.7% | +3.2 pp | ✅ |
        | T-Learner | 12.8% | +3.3 pp | ✅ |

        **🏆 Recommendation**: Use PSM (11.2%) as primary estimate
        with AIPW (12.7%) for robustness check
        """)

    st.markdown('<h2 class="subheader">🔬 Methodology</h2>', unsafe_allow_html=True)

    st.markdown("""
    **Causal Inference Framework:**

    This analysis uses state-of-the-art causal inference methods to establish
    causal relationships between email marketing and purchase behavior:

    1. **Problem Identification**: Recognize confounding in observational data
    2. **Propensity Scores**: Model P(email | customer characteristics)
    3. **Multiple Methods**: PSM, DiD, IPW, AIPW, T-Learner
    4. **Robustness Testing**: E-values, placebo tests, subgroup analysis
    5. **Business Translation**: ROI calculations and targeting strategy

    **Assumptions:**
    - Unconfoundedness: No unmeasured confounders (after conditioning on observables)
    - Positivity: All customers have positive probability of receiving email
    - Common Support: Overlap in propensity scores between groups
    """)

    # Project structure
    st.markdown('<h2 class="subheader">📁 Project Structure</h2>', unsafe_allow_html=True)

    with st.expander("View Files & Implementation", expanded=False):
        st.code("""
        Project Structure:
        ├── data/
        │   ├── raw/online_retail.xlsx
        │   └── processed/
        │       ├── simulated_email_campaigns.csv
        │       └── data_with_propensity_scores.csv
        ├── src/causal/
        │   ├── estimate_propensity_scores.py
        │   ├── propensity_score_matching_v2.py
        │   ├── difference_in_differences.py
        │   ├── inverse_probability_weighting.py
        │   ├── doubly_robust.py
        │   ├── robustness_analysis.py
        │   └── business_analysis.py
        └── src/visualization/ (30+ plots)
        """)

    # Navigation
    st.markdown("---")
    st.markdown("""
    <div class="success">
    <strong>Next Steps:</strong> Explore the tabs below to understand the problem,
    see how different methods compare, and view comprehensive results.
    </div>
    """, unsafe_allow_html=True)


def tab_problem():
    """Tab 2: The Problem - Naive analysis and confounding."""
    st.markdown('<h1 class="main-header">⚠️ The Problem: Why Naive Analysis Fails</h1>',
                unsafe_allow_html=True)

    data, ground_truth, true_effect = load_data()

    if data is not None:
        # Naive vs True comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<h2 class="subheader">📊 The Naive Approach</h2>',
                        unsafe_allow_html=True)

            # Calculate naive effect
            treated = data[data['received_email'] == 1]
            control = data[data['received_email'] == 0]

            naive_effect = (
                treated['purchased_this_week_observed'].mean() -
                control['purchased_this_week_observed'].mean()
            )

            st.metric(
                label="Naive Effect",
                value=f"{naive_effect:.1%}",
                delta="SEVERELY BIASED",
                delta_color="inverse"
            )

            st.markdown(f"""
            **Calculation:**
            - Email group: {treated['purchased_this_week_observed'].mean():.1%} purchase rate
            - No email group: {control['purchased_this_week_observed'].mean():.1%} purchase rate
            - Difference: {naive_effect:.1%}

            **Problem**: This compares different types of customers!
            """)

        with col2:
            st.markdown('<h2 class="subheader">✅ The Truth</h2>',
                        unsafe_allow_html=True)

            st.metric(
                label="True Causal Effect",
                value=f"{true_effect:.1%}",
                help="From simulation ground truth"
            )

            st.markdown(f"""
            **From Counterfactuals**:
            - True individual treatment effects embedded in data
            - Average treatment effect: {true_effect:.1%}
            - Naive bias: {naive_effect - true_effect:.1%}

            **Bias**: {(naive_effect - true_effect) / true_effect * 100:.0f}% overestimate!
            """)

        # Bias visualization
        st.markdown('<h2 class="subheader">📈 Bias Visualization</h2>',
                    unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Naive Estimate', 'True Effect'],
            y=[naive_effect * 100, true_effect * 100],
            text=[f'{naive_effect:.1%}', f'{true_effect:.1%}'],
            textposition='auto',
            marker_color=['#ff7f0e', '#2ca02c'],
            opacity=0.8
        ))

        fig.update_layout(
            title='Naive vs True Effect',
            yaxis_title='Purchase Rate Increase (%)',
            height=400,
            showlegend=False
        )

        fig.add_annotation(
            x=0.5, y=naive_effect * 100 + 0.5,
            text=f'BIAS: +{(naive_effect - true_effect) * 100:.1f}pp',
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            arrowsize=1,
            arrowwidth=2
        )

        st.plotly_chart(fig, use_container_width=True)

        # Confounding analysis
        st.markdown('<h2 class="subheader">🔍 Confounding Evidence</h2>',
                    unsafe_allow_html=True)

        # Calculate standardized differences
        features = ['rfm_score', 'days_since_last_purchase', 'total_past_purchases',
                   'customer_tenure_weeks', 'avg_order_value']

        balance_data = []
        for feature in features:
            treated_mean = treated[feature].mean()
            control_mean = control[feature].mean()
            treated_std = treated[feature].std()
            control_std = control[feature].std()

            # Standardized difference
            std_diff = (treated_mean - control_mean) / np.sqrt(
                (treated_std**2 + control_std**2) / 2
            )

            balance_data.append({
                'Feature': feature,
                'Treated Mean': treated_mean,
                'Control Mean': control_mean,
                'Std Diff': std_diff,
                'Balanced': abs(std_diff) < 0.1
            })

        balance_df = pd.DataFrame(balance_data)

        # Create balance plot
        fig = go.Figure()

        colors = ['#2ca02c' if b else '#d62728' for b in balance_df['Balanced']]

        fig.add_trace(go.Bar(
            x=balance_df['Feature'],
            y=balance_df['Std Diff'],
            marker_color=colors,
            opacity=0.7
        ))

        fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                     annotation_text="Good Balance Threshold")
        fig.add_hline(y=-0.1, line_dash="dash", line_color="orange")

        fig.update_layout(
            title='Covariate Imbalance (Standardized Differences)',
            yaxis_title='Standardized Difference',
            xaxis_title='Feature',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        - Red bars: Severely imbalanced (|Std Diff| > 0.1)
        - Green bars: Well balanced (|Std Diff| < 0.1)
        - All features are severely imbalanced → Confounding!
        """)

        # Feature comparison table
        with st.expander("View Detailed Feature Comparison", expanded=False):
            st.dataframe(
                balance_df.style.format({
                    'Treated Mean': '{:.2f}',
                    'Control Mean': '{:.2f}',
                    'Std Diff': '{:.3f}'
                }),
                use_container_width=True
            )

        # Customer characteristics
        st.markdown('<h2 class="subheader">👥 Who Gets Emails?</h2>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Email Recipients",
                value=f"{len(treated):,}",
                help=f"{len(treated)/len(data):.1%} of total"
            )
            st.markdown(f"""
            **Characteristics:**
            - RFM Score: {treated['rfm_score'].mean():.1f}
            - Days Since Purchase: {treated['days_since_last_purchase'].mean():.0f}
            - Past Purchases: {treated['total_past_purchases'].mean():.1f}
            """)

        with col2:
            st.metric(
                label="No Email",
                value=f"{len(control):,}",
                help=f"{len(control)/len(data):.1%} of total"
            )
            st.markdown(f"""
            **Characteristics:**
            - RFM Score: {control['rfm_score'].mean():.1f}
            - Days Since Purchase: {control['days_since_last_purchase'].mean():.0f}
            - Past Purchases: {control['total_past_purchases'].mean():.1f}
            """)

        with col3:
            st.metric(
                label="Difference",
                value="SEVERE",
                help="Treatment and control groups are different!"
            )
            st.markdown(f"""
            **Key Differences:**
            - RFM: +{treated['rfm_score'].mean() - control['rfm_score'].mean():.1f}
            - Days Since Purchase: {treated['days_since_last_purchase'].mean() - control['days_since_last_purchase'].mean():.0f}
            - Past Purchases: +{treated['total_past_purchases'].mean() - control['total_past_purchases'].mean():.1f}
            """)

        # Why this matters
        st.markdown('<h2 class="subheader">💡 Why This Matters</h2>',
                    unsafe_allow_html=True)

        st.markdown("""
        **The Core Problem:**

        1. **Email recipients are systematically different** from non-recipients
           - Higher RFM scores (better customers)
           - More recent purchases
           - More purchase history

        2. **Naive comparison captures BOTH:**
           - True email effect (what we want)
           - Baseline differences between groups (what we don't want)

        3. **Result: Severely biased estimate**
           - Naive: 16.0% vs True: 9.5%
           - 68% overestimate!

        **Solution: Causal Inference Methods**
        - Control for observed confounding
        - Match similar customers
        - Use counterfactual reasoning
        - Recover true causal effect

        **Next Tab**: See how causal methods fix this problem!
        """)


def tab_methods():
    """Tab 3: Causal Methods - Compare different approaches."""
    st.markdown('<h1 class="main-header">🔬 Causal Inference Methods</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Comparing different causal inference approaches to recover the true effect
    from confounded observational data.
    """)

    # Method comparison table
    methods_data = {
        'Method': ['Naive', 'PSM', 'DiD', 'IPW', 'AIPW', 'T-Learner'],
        'Estimate': [16.0, 11.2, 0.5, 13.6, 12.7, 12.8],
        'Bias (pp)': [6.54, 1.71, -9.33, 4.07, 3.21, 3.31],
        'Valid': ['❌', '✅', '❌', '✅', '✅', '✅'],
        'Notes': [
            'Severely biased by confounding',
            'Best performance, transparent',
            'Wrong method for this data',
            'Weight instability issues',
            'Doubly robust, modern',
            'Individual effects, heterogeneity'
        ]
    }

    methods_df = pd.DataFrame(methods_data)
    methods_df['Estimate'] = methods_df['Estimate'].apply(lambda x: f"{x:.1f}%")
    methods_df['Bias (pp)'] = methods_df['Bias (pp)'].apply(lambda x: f"{x:.2f}")

    st.markdown('<h2 class="subheader">📊 Method Comparison</h2>',
                unsafe_allow_html=True)

    st.dataframe(
        methods_df,
        use_container_width=True,
        hide_index=True
    )

    # Method descriptions
    st.markdown('<h2 class="subheader">📖 Method Descriptions</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. Propensity Score Matching (PSM)** 🥇
        - Match treated and control units with similar propensity scores
        - Creates balanced groups for comparison
        - Estimate: 11.2% (bias: 1.7 pp)
        - Pros: Transparent, interpretable, good diagnostics
        - Cons: Loses some observations

        **2. Difference-in-Differences (DiD)**
        - Use before/after variation with control group
        - Estimate: 0.5% (bias: -9.3 pp)
        - Pros: Controls for time-invariant confounders
        - Cons: Requires exogenous timing (not present here!)

        **3. Inverse Probability Weighting (IPW)**
        - Weight observations by inverse propensity scores
        - Estimate: 13.6% (bias: 4.1 pp)
        - Pros: Uses all data, simple concept
        - Cons: Weight instability with extreme scores
        """)

    with col2:
        st.markdown("""
        **4. AIPW (Doubly Robust)** 🥈
        - Combines IPW and outcome regression
        - Estimate: 12.7% (bias: 3.2 pp)
        - Pros: Robust to model misspecification
        - Cons: More complex, requires both models

        **5. T-Learner** 🥉
        - Separate models for treated and control
        - Estimate: 12.8% mean CATE
        - Pros: Individual-level effects (heterogeneity)
        - Cons: Requires well-specified models

        **6. Naive Comparison** ❌
        - Simple difference in means
        - Estimate: 16.0% (bias: 6.5 pp)
        - Pros: None
        - Cons: Severely biased by confounding
        """)

    # Bias comparison chart
    st.markdown('<h2 class="subheader">📈 Bias Comparison</h2>',
                unsafe_allow_html=True)

    methods_df_viz = pd.DataFrame(methods_data)
    methods_df_viz['abs_bias'] = methods_df_viz['Bias (pp)'].abs()

    fig = px.bar(
        methods_df_viz,
        x='Method',
        y='abs_bias',
        color='Valid',
        color_discrete_map={'✅': '#2ca02c', '❌': '#d62728'},
        text='Bias (pp)',
        title='Absolute Bias by Method (Lower is Better)'
    )

    fig.update_traces(texttemplate='%{text:.2f}pp', textposition='outside')
    fig.update_layout(
        height=400,
        yaxis_title='Absolute Bias (Percentage Points)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown('<h2 class="subheader">💡 Method Recommendations</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    **Primary Method**: Propensity Score Matching (PSM)
    - Lowest bias (1.7 percentage points)
    - Transparent and interpretable
    - Excellent balance diagnostics
    - Industry standard for observational studies

    **Robustness Check**: AIPW (Doubly Robust)
    - Valid even if outcome model is misspecified
    - Provides individual-level effects
    - Modern machine learning approach

    **For Heterogeneity**: T-Learner
    - Individual treatment effects (CATE)
    - Useful for targeting and personalization
    - Shows who benefits most from treatment

    **Avoid**: DiD (wrong study design)
    - Requires exogenous timing variation
    - Our data has selection on observables, not time-based treatment
    """)


def tab_results():
    """Tab 4: Results - Treatment effects and visualizations."""
    st.markdown('<h1 class="main-header">📊 Comprehensive Results</h1>',
                unsafe_allow_html=True)

    data, ground_truth, true_effect = load_data()

    if data is not None:
        # Treatment effects summary
        st.markdown('<h2 class="subheader">🎯 Treatment Effects Summary</h2>',
                    unsafe_allow_html=True)

        results_data = {
            'Method': ['Naive', 'PSM', 'IPW', 'AIPW', 'T-Learner', 'True Effect'],
            'Estimate': [16.0, 11.2, 13.6, 12.7, 12.8, 9.5],
            'Lower CI': [15.7, 10.8, 12.9, 12.0, None, None],
            'Upper CI': [16.4, 11.5, 14.3, 13.3, None, None],
            'Bias': [6.5, 1.7, 4.1, 3.2, 3.3, 0]
        }

        results_df = pd.DataFrame(results_data)

        # Treatment effects plot
        fig = go.Figure()

        # Add estimates with error bars
        for i, row in results_df.iterrows():
            if pd.notna(row['Lower CI']) and pd.notna(row['Upper CI']):
                fig.add_trace(go.Scatter(
                    x=[row['Method']],
                    y=[row['Estimate']],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[row['Upper CI'] - row['Estimate']],
                        arrayminus=[row['Estimate'] - row['Lower CI']]
                    ),
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='green' if row['Method'] == 'True Effect' else 'blue'
                    ),
                    name=row['Method']
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[row['Method']],
                    y=[row['Estimate']],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red' if row['Method'] == 'True Effect' else 'blue'
                    ),
                    name=row['Method']
                ))

        fig.add_hline(
            y=true_effect,
            line_dash="dash",
            line_color="red",
            annotation_text=f"True Effect: {true_effect:.1%}"
        )

        fig.update_layout(
            title='Treatment Effect Estimates with 95% Confidence Intervals',
            yaxis_title='Treatment Effect (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Subgroup analysis
        st.markdown('<h2 class="subheader">👥 Subgroup Analysis</h2>',
                    unsafe_allow_html=True)

        # RFM segments
        data_copy = data.copy()
        data_copy['rfm_segment'] = pd.cut(
            data_copy['rfm_score'],
            bins=[0, 7, 10, 13, 20],
            labels=['Low (0-7)', 'Medium (8-10)', 'High (11-13)', 'Very High (14+)']
        )

        rfm_effects = []
        for segment in data_copy['rfm_segment'].cat.categories:
            segment_data = data_copy[data_copy['rfm_segment'] == segment]
            if len(segment_data) > 0:
                treated = segment_data[segment_data['received_email'] == 1]
                control = segment_data[segment_data['received_email'] == 0]

                if len(treated) > 0 and len(control) > 0:
                    effect = (
                        treated['purchased_this_week_observed'].mean() -
                        control['purchased_this_week_observed'].mean()
                    )
                    rfm_effects.append({'Segment': segment, 'Effect': effect * 100})

        rfm_df = pd.DataFrame(rfm_effects)

        col1, col2 = st.columns(2)

        with col1:
            # RFM effects plot
            fig = px.bar(
                rfm_df,
                x='Segment',
                y='Effect',
                text='Effect',
                title='Treatment Effect by RFM Segment',
                color='Effect',
                color_continuous_scale='Blues'
            )

            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            **Key Findings:**

            - **Medium RFM (8-10)**: Highest response (17.1%)
            - **High RFM (11-13)**: Strong response (16.5%)
            - **Very High RFM (14+)**: Good response (16.1%)
            - **Low RFM (0-7)**: Weakest response (9.0%)

            **Business Insight**: Medium and high RFM customers
            respond best to email marketing. Target these segments
            for maximum ROI!
            """)

        # Robustness summary
        st.markdown('<h2 class="subheader">🔒 Robustness Summary</h2>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>E-Value: 2.58</h4>
                <p>Moderate robustness to unmeasured confounding</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Placebo Test: ❌</h4>
                <p>Failed - suggests persistent confounding</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Methods Agreement</h4>
                <p>Valid methods cluster 11-14%</p>
            </div>
            """, unsafe_allow_html=True)

        # Business impact
        st.markdown('<h2 class="subheader">💼 Business Impact</h2>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Financial Results:**

            - **ROI Range**: 49,922% - 103,677%
            - **Best Segments**: Loyal (Q4), Medium RFM (8-10)
            - **Current Profit**: $7.02M
            - **Optimized Profit**: $8.53M
            - **Expected Gain**: +$1.52M (+21.7%)

            **Strategy**: Email 81.7% of customers (optimal rate)
            with segment-specific personalization
            """)

        with col2:
            st.markdown("""
            **Recommendations:**

            1. **Primary Estimate**: PSM (11.2%)
               - 95% CI: 10.8% - 11.5%
               - Lowest bias

            2. **Target Segments**:
               - Medium/High RFM (8+)
               - Loyal customers (Q4 tenure)

            3. **Personalization**:
               - High-ROI: Premium content
               - Low-ROI: Re-engagement

            4. **Monitor**: Validate with experiments
            """)

        # Summary metrics
        st.markdown('<h2 class="subheader">📈 Summary Metrics</h2>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Best Estimate",
                value="11.2%",
                help="PSM estimate (bias 1.7pp)"
            )

        with col2:
            st.metric(
                label="Bias Reduction",
                value="74%",
                help="PSM vs Naive"
            )

        with col3:
            st.metric(
                label="Valid Methods",
                value="4/6",
                help="PSM, IPW, AIPW, T-Learner"
            )

        with col4:
            st.metric(
                label="Business Impact",
                value="+$1.52M",
                help="Expected profit increase"
            )


def main():
    """Main app function."""
    st.sidebar.title("Navigation")

    tabs = {
        "1️⃣ Overview": tab_overview,
        "2️⃣ The Problem": tab_problem,
        "3️⃣ Methods": tab_methods,
        "4️⃣ Results": tab_results
    }

    selection = st.sidebar.radio("Go to", list(tabs.keys()))

    tab_function = tabs[selection]
    tab_function()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About This Dashboard**

    Built with Streamlit

    Data: UCI Online Retail Dataset

    Methods: PSM, DiD, IPW, AIPW, T-Learner

    Validation: Ground truth simulation
    """)


if __name__ == "__main__":
    main()
