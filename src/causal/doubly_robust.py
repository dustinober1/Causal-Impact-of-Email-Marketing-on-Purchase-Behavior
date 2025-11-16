"""
Doubly Robust Causal Inference: AIPW and T-Learner

This script implements doubly robust methods that combine propensity scores
and outcome modeling for more robust causal inference.

Key Features:
1. AIPW (Augmented Inverse Propensity Weighting)
   - Propensity score model (already available)
   - Outcome regression models (treated and control separately)
   - Doubly robust estimator
   - Bootstrap standard errors

2. T-Learner for Heterogeneous Effects
   - Separate outcome models for treated and control
   - Estimate CATE (Conditional Average Treatment Effect)
   - Predict individual treatment effects
   - Heterogeneous effects by RFM segment

3. Cross-Validation for Model Selection
   - Evaluate outcome models
   - Select best specification

The doubly robust estimator is consistent if EITHER the propensity model
OR the outcome model is correctly specified (not necessarily both!).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DoublyRobustEstimator:
    """
    Doubly Robust Causal Inference with AIPW and T-Learner.
    """

    def __init__(self, random_state=42):
        """
        Initialize the estimator.

        Parameters:
        -----------
        random_state : int
            Random seed
        """
        self.random_state = random_state
        self.propensity_model = None
        self.outcome_model_0 = None  # Control group outcome model
        self.outcome_model_1 = None  # Treated group outcome model
        self.propensity_scores = None

    def fit_propensity_model(self, data, features, treatment_col='received_email'):
        """
        Fit propensity score model (logistic regression).

        Parameters:
        -----------
        data : DataFrame
            Data with features and treatment
        features : list
            List of feature column names
        treatment_col : str
            Treatment column name

        Returns:
        --------
        self : DoublyRobustEstimator
        """
        print("\n" + "=" * 70)
        print("STEP 1: FITTING PROPENSITY SCORE MODEL")
        print("=" * 70)

        X = data[features].values
        T = data[treatment_col].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        self.propensity_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.propensity_model.fit(X_scaled, T)

        # Predict propensity scores
        propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
        self.propensity_scores = propensity_scores

        # Calculate AUC
        auc = roc_auc_score(T, propensity_scores)

        print(f"\n📊 Propensity Model Results:")
        print(f"   AUC: {auc:.3f}")
        print(f"   Features: {len(features)}")
        print(f"   Sample size: {len(data):,}")

        return self

    def fit_outcome_models(self, data, outcome_col='purchased_this_week_observed',
                          features=None, model_type='random_forest'):
        """
        Fit outcome regression models separately for treated and control.

        Parameters:
        -----------
        data : DataFrame
            Data with outcomes and features
        outcome_col : str
            Outcome column name
        features : list
            List of feature column names (default: all numeric columns)
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'linear')

        Returns:
        --------
        self : DoublyRobustEstimator
        """
        print("\n" + "=" * 70)
        print("STEP 2: FITTING OUTCOME REGRESSION MODELS")
        print("=" * 70)

        # Default features
        if features is None:
            features = [col for col in data.columns if col not in [
                outcome_col, 'received_email', 'CustomerID', 'week_number',
                'week_start', 'purchase_this_week', 'revenue_this_week',
                'purchased_this_week_observed', 'email_assignment_probability',
                'individual_treatment_effect', 'true_purchase_probability',
                'true_purchase_prob_if_no_email'
            ]]

        # Split data by treatment
        treated_data = data[data['received_email'] == 1]
        control_data = data[data['received_email'] == 0]

        print(f"\n📊 Data Split:")
        print(f"   Treated group: {len(treated_data):,} observations")
        print(f"   Control group: {len(control_data):,} observations")
        print(f"   Features: {len(features)}")

        # Fit model for control group (Y0)
        print(f"\n🔄 Fitting outcome model for Control group (Y0)...")
        X_control = control_data[features].values
        y_control = control_data[outcome_col].values

        if model_type == 'random_forest':
            self.outcome_model_0 = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.outcome_model_0 = GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            )
        else:  # linear
            scaler_0 = StandardScaler()
            X_control_scaled = scaler_0.fit_transform(X_control)
            self.outcome_model_0 = LinearRegression()
            self.outcome_model_0.fit(X_control_scaled, y_control)
            self.outcome_scaler_0 = scaler_0

        if model_type in ['random_forest', 'gradient_boosting']:
            self.outcome_model_0.fit(X_control, y_control)

        # Evaluate control model
        control_pred = self.outcome_model_0.predict(X_control)
        control_r2 = 1 - np.var(y_control - control_pred) / np.var(y_control)
        control_rmse = np.sqrt(mean_squared_error(y_control, control_pred))

        print(f"   R²: {control_r2:.3f}")
        print(f"   RMSE: {control_rmse:.4f}")

        # Fit model for treated group (Y1)
        print(f"\n🔄 Fitting outcome model for Treated group (Y1)...")
        X_treated = treated_data[features].values
        y_treated = treated_data[outcome_col].values

        if model_type == 'random_forest':
            self.outcome_model_1 = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.outcome_model_1 = GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            )
        else:  # linear
            scaler_1 = StandardScaler()
            X_treated_scaled = scaler_1.fit_transform(X_treated)
            self.outcome_model_1 = LinearRegression()
            self.outcome_model_1.fit(X_treated_scaled, y_treated)
            self.outcome_scaler_1 = scaler_1

        if model_type in ['random_forest', 'gradient_boosting']:
            self.outcome_model_1.fit(X_treated, y_treated)

        # Evaluate treated model
        treated_pred = self.outcome_model_1.predict(X_treated)
        treated_r2 = 1 - np.var(y_treated - treated_pred) / np.var(y_treated)
        treated_rmse = np.sqrt(mean_squared_error(y_treated, treated_pred))

        print(f"   R²: {treated_r2:.3f}")
        print(f"   RMSE: {treated_rmse:.4f}")

        self.features = features
        self.model_type = model_type

        return self

    def estimate_aipw(self, data, outcome_col='purchased_this_week_observed'):
        """
        Estimate Average Treatment Effect using AIPW (Doubly Robust).

        Parameters:
        -----------
        data : DataFrame
            Data with propensity scores and outcomes
        outcome_col : str
            Outcome column name

        Returns:
        --------
        result : dict
            AIPW results
        """
        print("\n" + "=" * 70)
        print("STEP 3: ESTIMATING AIPW (DOUBLY ROBUST)")
        print("=" * 70)

        # Extract variables
        Y = data[outcome_col].values
        T = data['received_email'].values
        e = self.propensity_scores  # Propensity scores

        # Predict outcomes under both treatments
        X = data[self.features].values

        if self.model_type == 'linear':
            X_scaled_0 = self.outcome_scaler_0.transform(X)
            X_scaled_1 = self.outcome_scaler_1.transform(X)
            mu_0 = self.outcome_model_0.predict(X_scaled_0)
            mu_1 = self.outcome_model_1.predict(X_scaled_1)
        else:
            mu_0 = self.outcome_model_0.predict(X)
            mu_1 = self.outcome_model_1.predict(X)

        # AIPW estimator
        # E[Y(1)] = E[ μ1(X) + T*(Y - μ1(X)) / e(X) ]
        # E[Y(0)] = E[ μ0(X) + (1-T)*(Y - μ0(X)) / (1-e(X)) ]

        treated_term = mu_1 + T * (Y - mu_1) / e
        control_term = mu_0 + (1 - T) * (Y - mu_0) / (1 - e)

        ate_1 = np.mean(treated_term)
        ate_0 = np.mean(control_term)
        aipw_ate = ate_1 - ate_0

        # Simple estimators for comparison
        simple_treated = np.mean(Y[T == 1])
        simple_control = np.mean(Y[T == 0])
        simple_ate = simple_treated - simple_control

        print(f"\n📊 AIPW Results:")
        print(f"   AIPW ATE: {aipw_ate:.4f} ({aipw_ate:.1%})")
        print(f"   E[Y(1)]: {ate_1:.4f}")
        print(f"   E[Y(0)]: {ate_0:.4f}")

        print(f"\n📊 Comparison:")
        print(f"   Naive (simple diff): {simple_ate:.4f} ({simple_ate:.1%})")
        print(f"   AIPW (doubly robust): {aipw_ate:.4f} ({aipw_ate:.1%})")

        self.aipw_result = {
            'ate': aipw_ate,
            'e_y1': ate_1,
            'e_y0': ate_0,
            'naive': simple_ate,
            'mu_1': mu_1,
            'mu_0': mu_0
        }

        return self.aipw_result

    def estimate_t_learner(self, data, outcome_col='purchased_this_week_observed'):
        """
        Estimate Conditional Average Treatment Effect (CATE) using T-Learner.

        Parameters:
        -----------
        data : DataFrame
            Data with outcomes and features
        outcome_col : str
            Outcome column name

        Returns:
        --------
        result : dict
            T-learner results including CATE predictions
        """
        print("\n" + "=" * 70)
        print("STEP 4: ESTIMATING T-LEARNER (HETEROGENEOUS EFFECTS)")
        print("=" * 70)

        # Predict CATE for all observations
        X = data[self.features].values

        if self.model_type == 'linear':
            X_scaled_0 = self.outcome_scaler_0.transform(X)
            X_scaled_1 = self.outcome_scaler_1.transform(X)
            mu_0 = self.outcome_model_0.predict(X_scaled_0)
            mu_1 = self.outcome_model_1.predict(X_scaled_1)
        else:
            mu_0 = self.outcome_model_0.predict(X)
            mu_1 = self.outcome_model_1.predict(X)

        # CATE = μ1(X) - μ0(X)
        cate = mu_1 - mu_0

        print(f"\n📊 T-Learner Results:")
        print(f"   Average CATE: {np.mean(cate):.4f} ({np.mean(cate):.1%})")
        print(f"   CATE std: {np.std(cate):.4f}")
        print(f"   CATE range: [{np.min(cate):.4f}, {np.max(cate):.4f}]")

        # Compare to AIPW
        print(f"\n📊 Comparison:")
        print(f"   AIPW ATE (population): {self.aipw_result['ate']:.4f} ({self.aipw_result['ate']:.1%})")
        print(f"   T-Learner ATE (mean CATE): {np.mean(cate):.4f} ({np.mean(cate):.1%})")

        self.t_learner_result = {
            'cate': cate,
            'mu_1': mu_1,
            'mu_0': mu_0
        }

        return self.t_learner_result

    def analyze_heterogeneous_effects(self, data):
        """
        Analyze heterogeneous treatment effects by RFM segment.

        Parameters:
        -----------
        data : DataFrame
            Data with RFM scores

        Returns:
        --------
        heterogeneous_results : dict
            Results by segment
        """
        print("\n" + "=" * 70)
        print("STEP 5: ANALYZING HETEROGENEOUS EFFECTS BY RFM SEGMENT")
        print("=" * 70)

        # Calculate CATE for each customer
        data_copy = data.copy()
        data_copy['cate'] = self.t_learner_result['cate']

        # Create RFM segments
        data_copy['rfm_segment'] = pd.cut(
            data_copy['rfm_score'],
            bins=[0, 7, 10, 13, 20],
            labels=['Low (0-7)', 'Medium (8-10)', 'High (11-13)', 'Very High (14+)']
        )

        # Calculate mean CATE by segment
        cate_by_segment = data_copy.groupby('rfm_segment')['cate'].agg([
            'mean', 'std', 'count'
        ]).round(4)

        print(f"\n📊 CATE by RFM Segment:")
        print(cate_by_segment)

        # Statistical test for differences
        from scipy.stats import f_oneway
        segments = [group['cate'].values for name, group in data_copy.groupby('rfm_segment')]
        f_stat, p_value = f_oneway(*segments)

        print(f"\n📊 ANOVA Test for Segment Differences:")
        print(f"   F-statistic: {f_stat:.2f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Significant differences: {'Yes' if p_value < 0.05 else 'No'}")

        # Create visualization
        self.create_heterogeneous_plot(data_copy)

        self.heterogeneous_results = {
            'cate_by_segment': cate_by_segment,
            'f_stat': f_stat,
            'p_value': p_value
        }

        return self.heterogeneous_results

    def bootstrap_se(self, data, outcome_col='purchased_this_week_observed', n_bootstrap=500):
        """
        Calculate bootstrap standard errors for AIPW.

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
        print("STEP 6: BOOTSTRAP STANDARD ERRORS")
        print("=" * 70)

        print(f"\n🔄 Bootstrapping ({n_bootstrap:,} samples)...")

        n = len(data)
        bootstrap_ates = []

        for i in range(n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{n_bootstrap} ({100*(i+1)/n_bootstrap:.1f}%)")

            # Bootstrap sample
            boot_idx = np.random.choice(n, size=n, replace=True)
            boot_data = data.iloc[boot_idx].copy()

            # Refit models on bootstrap sample
            boot_estimator = DoublyRobustEstimator(random_state=self.random_state)
            boot_estimator.fit_propensity_model(boot_data, self.features)
            boot_estimator.fit_outcome_models(boot_data, outcome_col, self.features, self.model_type)
            boot_result = boot_estimator.estimate_aipw(boot_data, outcome_col)

            bootstrap_ates.append(boot_result['ate'])

        bootstrap_ates = np.array(bootstrap_ates)

        # Calculate statistics
        se_bootstrap = np.std(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        print(f"\n📊 Bootstrap Results:")
        print(f"   AIPW ATE: {self.aipw_result['ate']:.4f}")
        print(f"   Bootstrap SE: {se_bootstrap:.4f}")
        print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

        # Z-test
        z_stat = self.aipw_result['ate'] / se_bootstrap
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

    def create_heterogeneous_plot(self, data):
        """
        Create visualization of heterogeneous effects.

        Parameters:
        -----------
        data : DataFrame
            Data with CATE estimates

        Returns:
        --------
        fig : matplotlib Figure
        """
        print("\n📊 Creating Heterogeneous Effects Visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: CATE by RFM segment
        ax = axes[0, 0]
        cate_by_segment = data.groupby('rfm_segment')['cate'].agg(['mean', 'std']).reset_index()

        bars = ax.bar(range(len(cate_by_segment)), cate_by_segment['mean'],
                     yerr=cate_by_segment['std'], capsize=5, alpha=0.7,
                     color='lightblue', edgecolor='black')

        ax.set_xlabel('RFM Segment')
        ax.set_ylabel('CATE (Percentage Points)')
        ax.set_title('Conditional Average Treatment Effect by RFM', fontweight='bold')
        ax.set_xticks(range(len(cate_by_segment)))
        ax.set_xticklabels(cate_by_segment['rfm_segment'], rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, cate_by_segment['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Distribution of CATE
        ax = axes[0, 1]
        ax.hist(data['cate'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(np.mean(data['cate']), color='red', linestyle='--', linewidth=2,
                  label=f'Mean CATE: {np.mean(data["cate"]):.3f}')
        ax.set_xlabel('CATE (Individual Treatment Effect)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of CATE', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: CATE vs RFM score (scatter)
        ax = axes[1, 0]
        scatter = ax.scatter(data['rfm_score'], data['cate'], alpha=0.3, s=10)
        ax.set_xlabel('RFM Score')
        ax.set_ylabel('CATE')
        ax.set_title('CATE vs RFM Score', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(data['rfm_score'], data['cate'], 1)
        p = np.poly1d(z)
        ax.plot(data['rfm_score'], p(data['rfm_score']), "r--", alpha=0.8, linewidth=2)

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
        DOUBLY ROBUST SUMMARY
        {'='*30}

        AIPW (Average Treatment Effect):
        • ATE: {self.aipw_result['ate']:.4f} ({self.aipw_result['ate']:.1%})
        • E[Y(1)]: {self.aipw_result['e_y1']:.4f}
        • E[Y(0)]: {self.aipw_result['e_y0']:.4f}

        T-Learner (Heterogeneous Effects):
        • Mean CATE: {np.mean(data['cate']):.4f}
        • Std CATE: {np.std(data['cate']):.4f}
        • Min CATE: {np.min(data['cate']):.4f}
        • Max CATE: {np.max(data['cate']):.4f}

        Model Performance:
        • Outcome Model (Control) R²: {self.outcome_model_0.score(
            data[data['received_email']==0][self.features],
            data[data['received_email']==0]['purchased_this_week_observed']
        ):.3f}
        • Outcome Model (Treated) R²: {self.outcome_model_1.score(
            data[data['received_email']==1][self.features],
            data[data['received_email']==1]['purchased_this_week_observed']
        ):.3f}
        • Propensity Model AUC: {roc_auc_score(
            data['received_email'], self.propensity_scores
        ):.3f}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/src/visualization/doubly_robust_results.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        return fig

    def compare_to_true_effect(self):
        """
        Compare AIPW estimate to true causal effect.

        Returns:
        --------
        comparison : dict
            Comparison results
        """
        print("\n" + "=" * 70)
        print("STEP 7: COMPARING TO TRUE CAUSAL EFFECT")
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
        print(f"   AIPW Estimate:    {self.aipw_result['ate']:.4f} ({self.aipw_result['ate']:.1%})")
        print(f"   T-Learner Mean:   {np.mean(self.t_learner_result['cate']):.4f} ({np.mean(self.t_learner_result['cate']):.1%})")
        print(f"   True Effect:      {true_effect:.4f} ({true_effect:.1%})")
        print(f"   Expected (GT):    {expected_effect:.4f} ({expected_effect:.1%})")

        # Calculate bias
        aipw_bias = self.aipw_result['ate'] - true_effect
        tlearner_bias = np.mean(self.t_learner_result['cate']) - true_effect

        print(f"\n📊 Bias Analysis:")
        print(f"   AIPW Bias:     {aipw_bias:.4f} ({aipw_bias:.1%})")
        print(f"   T-Learner Bias: {tlearner_bias:.4f} ({tlearner_bias:.1%})")

        # Compare to naive
        naive_effect = self.aipw_result['naive']
        naive_bias = naive_effect - true_effect

        print(f"\n📊 Method Comparison:")
        print(f"   Naive:        {naive_effect:.4f} ({naive_bias:.4f} bias)")
        print(f"   AIPW:         {self.aipw_result['ate']:.4f} ({aipw_bias:.4f} bias)")
        print(f"   T-Learner:    {np.mean(self.t_learner_result['cate']):.4f} ({tlearner_bias:.4f} bias)")

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
            'aipw_estimate': self.aipw_result['ate'],
            'tlearner_estimate': np.mean(self.t_learner_result['cate']),
            'true_effect': true_effect,
            'expected_effect': expected_effect,
            'aipw_bias': aipw_bias,
            'tlearner_bias': tlearner_bias,
            'naive_bias': naive_bias
        }

        return self.comparison


def main():
    """
    Run complete doubly robust analysis.
    """
    print("\n" + "=" * 70)
    print("DOUBLY ROBUST CAUSAL INFERENCE: AIPW AND T-LEARNER")
    print("=" * 70)

    # Load data
    print("\nLoading data with propensity scores...")
    data_path = '/Users/dustinober/Projects/Causal-Impact-of-Email-Marketing-on-Purchase-Behavior/data/processed/data_with_propensity_scores.csv'
    data = pd.read_csv(data_path)

    print(f"✅ Data loaded: {data.shape}")
    print(f"   Treatment rate: {data['received_email'].mean():.1%}")

    # Define features
    features = [
        'days_since_last_purchase',
        'total_past_purchases',
        'avg_order_value',
        'customer_tenure_weeks',
        'rfm_score'
    ]

    # Initialize estimator
    print(f"\n🔧 Initializing Doubly Robust Estimator...")
    dr = DoublyRobustEstimator(random_state=42)

    # Fit models
    dr.fit_propensity_model(data, features)
    # Use linear regression for speed
    dr.fit_outcome_models(data, features=features, model_type='linear')

    # Estimate effects
    aipw_result = dr.estimate_aipw(data)
    t_learner_result = dr.estimate_t_learner(data)
    heterogeneous_results = dr.analyze_heterogeneous_effects(data)

    # Bootstrap standard errors (reduced to 200 for speed)
    bootstrap_results = dr.bootstrap_se(data, n_bootstrap=200)

    # Compare to true effect
    comparison = dr.compare_to_true_effect()

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n🎯 DOUBLY ROBUST ESTIMATES:")
    print(f"   AIPW ATE:     {aipw_result['ate']:.4f} ({aipw_result['ate']:.1%})")
    print(f"   T-Learner:    {t_learner_result['cate'].mean():.4f} ({t_learner_result['cate'].mean():.1%})")
    print(f"   95% CI:       [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
    print(f"   P-value:      {bootstrap_results['p_value']:.4f}")

    print(f"\n✅ VALIDATION:")
    print(f"   True Effect:  {comparison['true_effect']:.4f} ({comparison['true_effect']:.1%})")
    print(f"   AIPW Bias:    {comparison['aipw_bias']:.4f}")
    print(f"   CI includes true: {'Yes' if (bootstrap_results['ci_lower'] <= comparison['true_effect'] <= bootstrap_results['ci_upper']) else 'No'}")

    print(f"\n📊 HETEROGENEOUS EFFECTS:")
    print(f"   CATE std:     {t_learner_result['cate'].std():.4f}")
    print(f"   CATE range:   [{t_learner_result['cate'].min():.4f}, {t_learner_result['cate'].max():.4f}]")

    print(f"\n🎉 Doubly Robust Analysis Complete!")

    return dr


if __name__ == "__main__":
    dr_estimator = main()
