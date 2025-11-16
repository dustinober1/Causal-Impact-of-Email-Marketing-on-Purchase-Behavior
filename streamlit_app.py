"""
Streamlit Dashboard: Causal Impact of Email Marketing

A comprehensive interactive dashboard showcasing causal inference analysis
for email marketing effectiveness.

Enhanced Tabs:
1. Overview: Project description and key findings
2. The Problem: Naive analysis and confounding
3. Causal Methods: Interactive method selector with diagnostics
4. Results: Treatment effects comparison and heterogeneous effects
5. Business Recommendations: ROI calculator and policy simulator

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

        with col2:
            st.metric(
                label="No Email",
                value=f"{len(control):,}",
                help=f"{len(control)/len(data):.1%} of total"
            )

        with col3:
            st.metric(
                label="Difference",
                value="SEVERE",
                help="Treatment and control groups are different!"
            )

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
    """Tab 3: Causal Methods - Interactive method exploration."""
    st.markdown('<h1 class="main-header">🔬 Causal Inference Methods</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Explore different causal inference approaches interactively. Select a method to see
    its assumptions, diagnostics, and treatment effect estimate.
    """)

    data, ground_truth, true_effect = load_data()

    # Method selector
    st.markdown('<h2 class="subheader">🎯 Select Method to Explore</h2>',
                unsafe_allow_html=True)

    method_options = {
        'PSM': 'Propensity Score Matching (Recommended)',
        'AIPW': 'AIPW - Doubly Robust',
        'T-Learner': 'T-Learner - Heterogeneous Effects',
        'IPW': 'Inverse Probability Weighting',
        'DiD': 'Difference-in-Differences',
        'Naive': 'Naive Comparison (Baseline)'
    }

    selected_method = st.selectbox(
        'Choose a causal inference method:',
        list(method_options.keys()),
        format_func=lambda x: method_options[x]
    )

    # Method details based on selection
    method_details = {
        'PSM': {
            'estimate': 11.2,
            'bias': 1.7,
            'ci_lower': 10.8,
            'ci_upper': 11.5,
            'description': """
            **Propensity Score Matching (PSM)** 🥇

            Match treated and control units with similar propensity scores to create
            balanced groups for comparison. Industry standard for observational studies.
            """,
            'pros': ['Transparent and interpretable', 'Excellent diagnostics', 'Industry standard', 'Handles confounding well'],
            'cons': ['Loses some observations', 'Requires good overlap', 'Matching quality matters'],
            'best_for': 'Most observational studies with good baseline covariates'
        },
        'AIPW': {
            'estimate': 12.7,
            'bias': 3.2,
            'ci_lower': 12.0,
            'ci_upper': 13.3,
            'description': """
            **Augmented Inverse Probability Weighting (AIPW)** 🥈

            Combines inverse probability weighting with outcome regression for
            doubly robust estimation. Valid if EITHER model is correctly specified.
            """,
            'pros': ['Doubly robust', 'More efficient', 'Modern approach', 'Uses all data'],
            'cons': ['More complex', 'Requires two models', 'Can be unstable with extreme weights'],
            'best_for': 'Modern causal inference with ML models'
        },
        'T-Learner': {
            'estimate': 12.8,
            'bias': 3.3,
            'ci_lower': None,
            'ci_upper': None,
            'description': """
            **T-Learner** 🥉

            Fit separate outcome models for treated and control groups, then
            estimate individual treatment effects as the difference.
            """,
            'pros': ['Individual effects (CATE)', 'Flexible modeling', 'Heterogeneity insights', 'Good for targeting'],
            'cons': ['Requires two models', 'Sensitive to model misspecification', 'No uncertainty for CATE'],
            'best_for': 'Heterogeneous effects and personalization'
        },
        'IPW': {
            'estimate': 13.6,
            'bias': 4.1,
            'ci_lower': 12.8,
            'ci_upper': 14.3,
            'description': """
            **Inverse Probability Weighting (IPW)**

            Weight observations by the inverse of their propensity score to create
            a pseudo-population where treatment is as-if random.
            """,
            'pros': ['Uses all data', 'Simple concept', 'Unbiased if model correct'],
            'cons': ['Weight instability', 'Sensitive to extreme weights', 'Requires trimming'],
            'best_for': 'When matching is infeasible but have good propensity model'
        },
        'DiD': {
            'estimate': 0.5,
            'bias': -9.3,
            'ci_lower': -1.7,
            'ci_upper': 2.7,
            'description': """
            **Difference-in-Differences (DiD)**

            Compare changes over time between treatment and control groups to
            estimate causal effects while controlling for time-invariant confounders.
            """,
            'pros': ['Controls for time-invariant confounders', 'Good for policy evaluation', 'Robust to some unobservables'],
            'cons': ['Requires exogenous timing', 'Our data violates this', 'Not suitable for selection on observables'],
            'best_for': 'Policy changes with exogenous timing (e.g., minimum wage, law changes)'
        },
        'Naive': {
            'estimate': 16.0,
            'bias': 6.5,
            'ci_lower': 15.7,
            'ci_upper': 16.3,
            'description': """
            **Naive Comparison** ❌

            Simple difference in means between treatment and control groups.
            This is what most people would calculate first, but it's BIASED!
            """,
            'pros': ['Simple', 'Fast', 'No modeling required'],
            'cons': ['Severely biased', 'No causal interpretation', 'Misleads decision making'],
            'best_for': 'Never use for causal inference - only for exploration'
        }
    }

    details = method_details[selected_method]

    # Display method details
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(details['description'])

        st.markdown('<h3 class="subheader">✅ Pros</h3>', unsafe_allow_html=True)
        for pro in details['pros']:
            st.markdown(f"- {pro}")

        st.markdown('<h3 class="subheader">⚠️ Cons</h3>', unsafe_allow_html=True)
        for con in details['cons']:
            st.markdown(f"- {con}")

        st.markdown('<h3 class="subheader">🎯 Best For</h3>', unsafe_allow_html=True)
        st.markdown(details['best_for'])

    with col2:
        st.markdown('<h3 class="subheader">📊 Treatment Effect</h3>',
                    unsafe_allow_html=True)

        if details['ci_lower'] is not None:
            st.metric(
                label="Estimate",
                value=f"{details['estimate']:.1f}%",
                help=f"95% CI: {details['ci_lower']:.1f}% - {details['ci_upper']:.1f}%"
            )

            st.metric(
                label="Bias vs True",
                value=f"+{details['bias']:.1f}pp",
                help=f"True effect: {true_effect:.1f}%"
            )

            # Confidence interval plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[details['ci_lower'], details['ci_upper']],
                y=['Effect', 'Effect'],
                mode='lines+markers',
                line=dict(width=8, color='blue'),
                marker=dict(size=15, color='blue'),
                name='95% CI'
            ))

            fig.add_trace(go.Scatter(
                x=[details['estimate']],
                y=['Effect'],
                mode='markers',
                marker=dict(size=20, color='red', symbol='diamond'),
                name='Point Estimate'
            ))

            fig.add_vline(
                x=true_effect,
                line_dash="dash",
                line_color="green",
                annotation_text=f"True: {true_effect:.1f}%"
            )

            fig.update_layout(
                title='Treatment Effect with 95% CI',
                xaxis_title='Treatment Effect (%)',
                height=200,
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.metric(
                label="Mean CATE",
                value=f"{details['estimate']:.1f}%",
                help="Average Conditional Treatment Effect"
            )

    # Method comparison table
    st.markdown('<h2 class="subheader">📊 Compare All Methods</h2>',
                unsafe_allow_html=True)

    methods_data = {
        'Method': ['Naive', 'PSM', 'DiD', 'IPW', 'AIPW', 'T-Learner', 'True Effect'],
        'Estimate': [16.0, 11.2, 0.5, 13.6, 12.7, 12.8, 9.5],
        'Bias (pp)': [6.54, 1.71, -9.33, 4.07, 3.21, 3.31, 0],
        'Valid': ['❌', '✅', '❌', '✅', '✅', '✅', '✅'],
        'CI Lower': [15.7, 10.8, -1.7, 12.8, 12.0, None, None],
        'CI Upper': [16.4, 11.5, 2.7, 14.3, 13.3, None, None]
    }

    methods_df = pd.DataFrame(methods_data)
    methods_df['Estimate'] = methods_df['Estimate'].apply(lambda x: f"{x:.1f}%")
    methods_df['Bias (pp)'] = methods_df['Bias (pp)'].apply(lambda x: f"{x:.2f}")
    methods_df['95% CI'] = methods_df.apply(
        lambda row: f"[{row['CI Lower']:.1f}%, {row['CI Upper']:.1f}%]"
        if pd.notna(row['CI Lower']) else 'N/A', axis=1
    )

    display_df = methods_df[['Method', 'Estimate', 'Bias (pp)', '95% CI', 'Valid']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Bias comparison chart
    methods_df_viz = pd.DataFrame(methods_data)
    methods_df_viz['abs_bias'] = methods_df_viz['Bias (pp)'].abs()

    fig = px.bar(
        methods_df_viz[methods_df_viz['Method'] != 'True Effect'],
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


def tab_results():
    """Tab 4: Results - Treatment effects and heterogeneous effects."""
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

        # Sensitivity analysis selector
        st.markdown('<h2 class="subheader">🔬 Sensitivity Analysis</h2>',
                    unsafe_allow_html=True)

        sensitivity_option = st.selectbox(
            'Choose sensitivity test:',
            ['Subgroup Analysis', 'E-Value', 'Method Agreement']
        )

        if sensitivity_option == 'Subgroup Analysis':
            st.markdown('<h3 class="subheader">👥 Heterogeneous Effects</h3>',
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
                        rfm_effects.append({
                            'Segment': segment,
                            'Effect': effect * 100,
                            'Sample_Size': len(segment_data)
                        })

            rfm_df = pd.DataFrame(rfm_effects)

            col1, col2 = st.columns(2)

            with col1:
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

                # Show detailed table
                with st.expander("View Detailed Subgroup Results", expanded=False):
                    st.dataframe(rfm_df, use_container_width=True)

        elif sensitivity_option == 'E-Value':
            st.markdown('<h3 class="subheader">🔒 E-Value: Unmeasured Confounding Sensitivity</h3>',
                        unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **What is an E-Value?**

                The E-value is the minimum strength of association that an unmeasured
                confounder would need to have with both the treatment and outcome to
                fully explain away the observed effect.

                **Interpretation:**
                - E-value > 4: Very robust
                - E-value > 3: Fairly robust
                - E-value < 3: Vulnerable

                **For PSM estimate (11.2%):**
                - Baseline rate: 18.6%
                - Risk ratio: 1.60
                """)
            with col2:
                e_value_result = 2.58
                st.metric(
                    label="E-Value",
                    value=f"{e_value_result:.2f}",
                    help="Minimum confounding strength needed"
                )

                st.markdown(f"""
                **Result**: ⚠️ Moderate robustness

                An unmeasured confounder would need to increase both:
                1. Probability of receiving email, AND
                2. Probability of purchasing

                by a factor of **{e_value_result:.1f}** to fully explain away
                the observed effect.
                """)

        else:  # Method Agreement
            st.markdown('<h3 class="subheader">✅ Method Agreement</h3>',
                        unsafe_allow_html=True)

            methods_data = {
                'Method': ['PSM', 'IPW', 'AIPW', 'T-Learner'],
                'Estimate': [11.2, 13.6, 12.7, 12.8],
                'Valid': ['✅', '✅', '✅', '✅']
            }

            methods_df = pd.DataFrame(methods_data)
            mean_est = methods_df['Estimate'].mean()
            std_est = methods_df['Estimate'].std()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Mean Estimate",
                    value=f"{mean_est:.1f}%",
                    help="Average across valid methods"
                )

            with col2:
                st.metric(
                    label="Std Deviation",
                    value=f"{std_est:.1f}pp",
                    help="Variability across methods"
                )

            with col3:
                st.metric(
                    label="Range",
                    value=f"{methods_df['Estimate'].max() - methods_df['Estimate'].min():.1f}pp",
                    help="Min to max estimate"
                )

            st.markdown("""
            **Conclusion**: Valid methods cluster around 11-14%, showing reasonable
            agreement. PSM provides the most conservative (lowest) estimate.
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


def tab_business():
    """Tab 5: Business Recommendations - ROI calculator and policy simulator."""
    st.markdown('<h1 class="main-header">💼 Business Recommendations</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="success">
    <strong>Interactive Business Tools:</strong> Use the calculators below to explore
    different targeting strategies and their financial impact on your business.
    </div>
    """, unsafe_allow_html=True)

    data, ground_truth, true_effect = load_data()

    # Business parameters
    st.markdown('<h2 class="subheader">💰 Business Assumptions</h2>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        cost_per_email = st.number_input(
            'Cost per email ($)',
            min_value=0.01,
            max_value=1.00,
            value=0.10,
            step=0.01,
            help='Cost to send one email'
        )

    with col2:
        avg_order_value = st.number_input(
            'Average order value ($)',
            min_value=100,
            max_value=1000,
            value=556.95,
            step=50,
            help='Average revenue per purchase'
        )

    with col3:
        treatment_effect = st.selectbox(
            'Treatment effect to use:',
            ['PSM (11.2%)', 'AIPW (12.7%)', 'T-Learner (12.8%)', 'Custom'],
            index=0
        )

        custom_effect = None
        if treatment_effect == 'Custom':
            custom_effect = st.number_input(
                'Custom effect (%)',
                min_value=5.0,
                max_value=20.0,
                value=11.2,
                step=0.5
            )

    # ROI Calculator
    st.markdown('<h2 class="subheader">📊 ROI Calculator by Segment</h2>',
                unsafe_allow_html=True)

    if data is not None:
        # Create segments
        data_copy = data.copy()
        data_copy['rfm_segment'] = pd.cut(
            data_copy['rfm_score'],
            bins=[0, 7, 10, 13, 20],
            labels=['Low (0-7)', 'Medium (8-10)', 'High (11-13)', 'Very High (14+)']
        )

        # Calculate effects and ROI
        roi_results = []
        for segment in data_copy['rfm_segment'].cat.categories:
            segment_data = data_copy[data_copy['rfm_segment'] == segment]
            if len(segment_data) > 0:
                treated = segment_data[segment_data['received_email'] == 1]
                control = segment_data[segment_data['received_email'] == 0]

                if len(treated) > 0 and len(control) > 0:
                    uplift = (
                        treated['purchased_this_week_observed'].mean() -
                        control['purchased_this_week_observed'].mean()
                    )

                    # Financial calculations
                    additional_purchases = uplift * len(segment_data)
                    incremental_revenue = additional_purchases * avg_order_value
                    cost = len(segment_data) * cost_per_email
                    net_profit = incremental_revenue - cost
                    roi = (net_profit / cost) * 100 if cost > 0 else 0

                    roi_results.append({
                        'Segment': segment,
                        'Sample_Size': len(segment_data),
                        'Uplift (%)': uplift * 100,
                        'Revenue ($)': incremental_revenue,
                        'Cost ($)': cost,
                        'Net_Profit ($)': net_profit,
                        'ROI (%)': roi
                    })

        roi_df = pd.DataFrame(roi_results)
        roi_df = roi_df.sort_values('ROI (%)', ascending=False)

        # Display ROI table
        st.dataframe(
            roi_df.style.format({
                'Sample_Size': '{:,}',
                'Uplift (%)': '{:.1f}%',
                'Revenue ($)': '${:,.0f}',
                'Cost ($)': '${:,.0f}',
                'Net_Profit ($)': '${:,.0f}',
                'ROI (%)': '{:,.0f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

        # ROI visualization
        fig = px.bar(
            roi_df,
            x='Segment',
            y='ROI (%)',
            text='ROI (%)',
            title='ROI by Customer Segment',
            color='ROI (%)',
            color_continuous_scale='Greens'
        )

        fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    # Policy Simulator
    st.markdown('<h2 class="subheader">🎯 Policy Simulator</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    **Simulate different targeting strategies and see the financial impact.**
    Adjust the parameters below to test "what-if" scenarios.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subheader">Targeting Parameters</h3>',
                    unsafe_allow_html=True)

        min_rfm = st.slider(
            'Minimum RFM score',
            min_value=0,
            max_value=15,
            value=0,
            help='Only target customers with RFM >= this value'
        )

        min_tenure = st.slider(
            'Minimum tenure (weeks)',
            min_value=0,
            max_value=52,
            value=0,
            help='Only target customers with tenure >= this value'
        )

        max_days_since = st.slider(
            'Max days since last purchase',
            min_value=0,
            max_value=365,
            value=365,
            help='Only target customers who purchased within this many days'
        )

    with col2:
        st.markdown('<h3 class="subheader">Results</h3>',
                    unsafe_allow_html=True)

        if data is not None:
            # Apply filters
            sim_data = data.copy()
            mask = (
                (sim_data['rfm_score'] >= min_rfm) &
                (sim_data['customer_tenure_weeks'] >= min_tenure) &
                (sim_data['days_since_last_purchase'] <= max_days_since)
            )

            targeted_customers = sim_data[mask]
            n_targeted = len(targeted_customers)
            n_total = len(sim_data)

            # Use selected treatment effect
            if treatment_effect == 'PSM (11.2%)':
                effect = 0.112
            elif treatment_effect == 'AIPW (12.7%)':
                effect = 0.127
            elif treatment_effect == 'T-Learner (12.8%)':
                effect = 0.128
            else:
                effect = custom_effect / 100.0

            # Financial projections
            additional_purchases = n_targeted * effect
            incremental_revenue = additional_purchases * avg_order_value
            cost = n_targeted * cost_per_email
            net_profit = incremental_revenue - cost
            roi = (net_profit / cost) * 100 if cost > 0 else 0

            st.metric(
                label="Targeted Customers",
                value=f"{n_targeted:,}",
                help=f"{n_targeted/n_total:.1%} of total"
            )

            st.metric(
                label="Additional Purchases",
                value=f"{additional_purchases:,.0f}",
                help=f"At {effect:.1%} uplift rate"
            )

            st.metric(
                label="Incremental Revenue",
                value=f"${incremental_revenue:,.0f}",
                help=f"${avg_order_value:.2f} per purchase"
            )

            st.metric(
                label="Total Cost",
                value=f"${cost:,.0f}",
                help=f"${cost_per_email:.2f} per email"
            )

            st.metric(
                label="Net Profit",
                value=f"${net_profit:,.0f}",
                delta=f"{roi:.0f}% ROI",
                delta_color='normal'
            )

    # Optimization recommendation
    st.markdown('<h2 class="subheader">🏆 Optimization Recommendation</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    **Based on our analysis:**

    1. **Current Strategy**: Email 81.7% of customers
       - Profit: $7.02M
       - ROI: Extremely high across all segments

    2. **Recommended Optimization**:
       - Keep emailing 81.7% of customers (volume strategy)
       - **Prioritize** high-ROI segments with better content
       - **Segment-specific** frequency and messaging
       - **Expected impact**: +$1.52M profit (+21.7%)

    3. **Priority Segments**:
       - **Loyal customers (Q4)**: VIP treatment, 3x/week
       - **Medium RFM (8-10)**: Personalized offers, 2x/week
       - **High RFM (11-13)**: Exclusive access, 2x/week
       - **Low RFM (0-7)**: Re-engagement, 1x/week

    4. **Why This Works**:
       - Email cost extremely low ($0.10)
       - Average order value high ($556.95)
       - Even "low-ROI" segments generate 50,000% ROI
       - Volume matters more than selectivity

    **Next Steps:**
    - Implement segment-specific email templates
    - Set frequency caps by customer value
    - A/B test personalized content
    - Monitor performance by segment
    """)

    # Executive summary
    st.markdown('<h2 class="subheader">📋 Executive Summary</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Key Insights:**

        - Email marketing is **extremely profitable**
          (ROI: 50,000% - 104,000%)

        - **Loyal customers** show highest response
          (18.6% uplift)

        - **Medium RFM** customers are the sweet spot
          (17.1% uplift)

        - **Volume strategy** beats selectivity
          (Email 81.7% customers)
        """)

    with col2:
        st.markdown("""
        **Financial Impact:**

        - **Current profit**: $7.02M
        - **Optimized profit**: $8.53M
        - **Improvement**: +$1.52M (+21.7%)
        - **Investment**: Minimal (labor only)
        - **Payback**: Immediate

        **Bottom Line**: Email marketing is a goldmine!
        Prioritize and personalize, don't exclude.
        """)


def main():
    """Main app function."""
    st.sidebar.title("Navigation")

    tabs = {
        "1️⃣ Overview": tab_overview,
        "2️⃣ The Problem": tab_problem,
        "3️⃣ Causal Methods": tab_methods,
        "4️⃣ Results": tab_results,
        "5️⃣ Business": tab_business
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
