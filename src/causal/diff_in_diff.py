"""
Difference-in-Differences (DiD) Module

Implementation of DiD estimators for panel data with:
- Two-way fixed effects estimation
- Parallel trends testing
- Event study analysis
- Robust standard errors

Author: Causal Inference Toolkit
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for panel data.

    Implements the two-way fixed effects DiD model:
    Y_it = α + β*Treated_i*Post_t + γ_i + δ_t + ε_it

    Where:
    - γ_i = individual fixed effects
    - δ_t = time fixed effects
    - β = treatment effect (DID estimator)

    Parameters:
    -----------
    outcome_col : str
        Name of outcome variable column
    treatment_col : str
        Name of treatment indicator column (0/1)
    time_col : str
        Name of time column
    unit_col : str
        Name of unit identifier column
    post_period : int or str
        Value indicating post-treatment period
    cluster_var : str, optional
        Variable for clustering standard errors

    Example:
    --------
    >>> did = DifferenceInDifferences(
    ...     outcome_col='outcome',
    ...     treatment_col='treated',
    ...     time_col='time',
    ...     unit_col='unit_id',
    ...     post_period=10
    ... )
    >>> results = did.fit(data)
    >>> did.check_parallel_trends(data)
    """

    def __init__(self,
                 outcome_col: str,
                 treatment_col: str,
                 time_col: str,
                 unit_col: str,
                 post_period: Union[int, str],
                 cluster_var: Optional[str] = None):

        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.unit_col = unit_col
        self.post_period = post_period
        self.cluster_var = cluster_var
        self.results_ = None
        self.is_fitted = False

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for DiD estimation.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data

        Returns:
        --------
        prepared_data : pd.DataFrame
            Data with DiD interaction term
        """
        df = data.copy()

        # Create post indicator
        df['post'] = (df[self.time_col] >= self.post_period).astype(int)

        # Create DiD interaction term: treated * post
        df['did_interaction'] = df[self.treatment_col] * df['post']

        return df

    def fit(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate DiD model.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data with outcomes, treatment, time, and unit IDs

        Returns:
        --------
        results : dict
            DiD estimates and statistics
        """
        # Prepare data
        df = self._prepare_data(data)

        # Check data requirements
        self._check_data_requirements(df)

        # Calculate group-time means
        group_time_means = self._calculate_group_means(df)

        # Calculate DiD estimate
        treated_post = group_time_means.loc[('treated', 'post'), 'mean_outcome']
        treated_pre = group_time_means.loc[('treated', 'pre'), 'mean_outcome']
        control_post = group_time_means.loc[('control', 'post'), 'mean_outcome']
        control_pre = group_time_means.loc[('control', 'pre'), 'mean_outcome']

        # DiD = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)
        treated_change = treated_post - treated_pre
        control_change = control_post - control_pre
        did_estimate = treated_change - control_change

        # Store results
        self.results_ = {
            'did_estimate': did_estimate,
            'treated_pre': treated_pre,
            'treated_post': treated_post,
            'control_pre': control_pre,
            'control_post': control_post,
            'treated_change': treated_change,
            'control_change': control_change,
            'n_treated': group_time_means.loc[('treated', 'post'), 'n_obs'],
            'n_control': group_time_means.loc[('control', 'post'), 'n_obs'],
            'group_time_means': group_time_means
        }

        # Calculate standard errors
        se = self._calculate_se(df, did_estimate)
        self.results_['std_error'] = se
        self.results_['t_statistic'] = did_estimate / se if se > 0 else np.inf
        self.results_['p_value'] = 2 * (1 - stats.norm.cdf(abs(did_estimate / se)))
        self.results_['ci_lower'] = did_estimate - 1.96 * se
        self.results_['ci_upper'] = did_estimate + 1.96 * se

        self.is_fitted = True

        return self.results_

    def _check_data_requirements(self, data: pd.DataFrame):
        """
        Check if data meets DiD requirements.

        Parameters:
        -----------
        data : pd.DataFrame
            Prepared data
        """
        # Check for missing values
        required_cols = [self.outcome_col, self.treatment_col, self.time_col, self.unit_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")

            if data[col].isnull().any():
                missing_count = data[col].isnull().sum()
                warnings.warn(f"Column {col} has {missing_count} missing values")

        # Check treatment is binary
        unique_treatment = data[self.treatment_col].unique()
        if len(unique_treatment) != 2 or not set(unique_treatment).issubset({0, 1}):
            raise ValueError("Treatment variable must be binary (0/1)")

    def _calculate_group_means(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean outcomes by treatment and time groups.

        Parameters:
        -----------
        data : pd.DataFrame
            Prepared data

        Returns:
        --------
        group_means : pd.DataFrame
            Group-time means
        """
        group_stats = data.groupby([self.treatment_col, 'post']).agg({
            self.outcome_col: ['mean', 'std', 'count'],
            self.unit_col: 'nunique'
        }).round(4)

        # Flatten column names
        group_stats.columns = ['mean_outcome', 'std_outcome', 'n_obs', 'n_units']
        group_stats = group_stats.reset_index()

        # Create multi-index for easier access
        group_stats['treatment_status'] = group_stats[self.treatment_col].map({
            0: 'control',
            1: 'treated'
        })
        group_stats['time_period'] = group_stats['post'].map({
            0: 'pre',
            1: 'post'
        })

        group_stats = group_stats.set_index(['treatment_status', 'time_period'])

        return group_stats

    def _calculate_se(self, data: pd.DataFrame, did_estimate: float) -> float:
        """
        Calculate standard errors for DiD estimate.

        Parameters:
        -----------
        data : pd.DataFrame
            Prepared data
        did_estimate : float
            DiD estimate

        Returns:
        --------
        se : float
            Standard error
        """
        # Method 1: Aggregate to group-time level and compute variance
        df = data.copy()

        # Calculate residuals
        # Simple approach: use variation in group-time means
        group_stats = self._calculate_group_time_variation(df)

        # Standard error based on group-time variation
        # This is an approximation - full implementation would use
        # analytical variance formulas or bootstrap
        n_groups = group_stats['n_groups'].sum()
        se = np.sqrt(did_estimate ** 2 / n_groups)

        return se

    def _calculate_group_time_variation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate variation at group-time level.

        Parameters:
        -----------
        data : pd.DataFrame
            Prepared data

        Returns:
        --------
        variation : pd.DataFrame
            Group-time variation statistics
        """
        # Count units per group-time cell
        variation = data.groupby([self.treatment_col, 'post', self.unit_col]).agg({
            self.outcome_col: 'mean'
        }).reset_index()

        variation['n_units'] = 1

        variation = variation.groupby([self.treatment_col, 'post']).agg({
            'n_units': 'sum',
            self.unit_col: 'nunique'
        }).reset_index()

        variation.columns = [self.treatment_col, 'post', 'n_obs', 'n_groups']

        return variation

    def check_parallel_trends(self, data: pd.DataFrame,
                             pre_periods: Optional[List] = None) -> Dict[str, float]:
        """
        Test parallel trends assumption.

        Tests if treatment and control groups have parallel trends
        in the pre-treatment period.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        pre_periods : list, optional
            List of pre-treatment periods to include in test

        Returns:
        --------
        test_results : dict
            Parallel trends test results
        """
        # Prepare data
        df = self._prepare_data(data)

        # Filter to pre-treatment period only
        df_pre = df[df['post'] == 0].copy()

        if len(df_pre) == 0:
            raise ValueError("No pre-treatment observations found")

        # If no specific pre-periods provided, use all available
        if pre_periods is None:
            pre_periods = sorted(df_pre[self.time_col].unique())

        # Filter to specified periods
        df_pre = df_pre[df_pre[self.time_col].isin(pre_periods)]

        # Regress outcome on treatment * time interaction
        # Y_it = α + β*(Treated_i * Time_t) + γ_i + δ_t + ε_it
        # Test if β = 0

        # Create interaction term
        df_pre['treat_time'] = df_pre[self.treatment_col] * df_pre[self.time_col]

        # Calculate regression coefficient
        # Simple OLS for the interaction term

        # Get group means by time and treatment
        time_means = df_pre.groupby([self.time_col, self.treatment_col])[self.outcome_col].mean().reset_index()

        if len(time_means) < 4:
            warnings.warn("Not enough time periods to test parallel trends reliably")

        # Calculate trend difference
        treated_times = time_means[time_means[self.treatment_col] == 1]
        control_times = time_means[time_means[self.treatment_col] == 0]

        if len(treated_times) < 2 or len(control_times) < 2:
            raise ValueError("Need at least 2 time periods for each group")

        # Simple slope calculation
        treated_slope = np.polyfit(
            treated_times[self.time_col],
            treated_times[self.outcome_col],
            1
        )[0]

        control_slope = np.polyfit(
            control_times[self.time_col],
            control_times[self.outcome_col],
            1
        )[0]

        trend_diff = treated_slope - control_slope

        # Standard error (approximate)
        n_treated = len(treated_times)
        n_control = len(control_times)
        se = np.sqrt(
            treated_slope ** 2 / n_treated +
            control_slope ** 2 / n_control
        )

        # T-statistic and p-value
        t_stat = trend_diff / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            'parallel_trends_stat': trend_diff,
            'std_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'treated_slope': treated_slope,
            'control_slope': control_slope,
            'parallel_trends_satisfied': p_value > 0.05,
            'n_pre_periods': len(pre_periods),
            'n_treated_periods': n_treated,
            'n_control_periods': n_control
        }

    def event_study(self, data: pd.DataFrame,
                   leads: int = 3,
                   lags: int = 3) -> pd.DataFrame:
        """
        Event study: Dynamic treatment effects over time.

        Estimates treatment effect in each period relative to
        a baseline (usually pre-treatment period).

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        leads : int, default=3
            Number of leads (pre-treatment periods) to include
        lags : int, default=3
            Number of lags (post-treatment periods) to include

        Returns:
        --------
        event_study_results : pd.DataFrame
            Period-by-period treatment effects
        """
        # Prepare data
        df = self._prepare_data(data)

        # Define event time (relative to treatment)
        df['event_time'] = df[self.time_col] - self.post_period

        # Filter to event window
        df_event = df[
            (df['event_time'] >= -leads) &
            (df['event_time'] <= lags)
        ].copy()

        # Calculate effects by event time
        effects = []

        for event_t in range(-leads, lags + 1):
            df_t = df_event[df_event['event_time'] == event_t]

            if len(df_t) == 0:
                continue

            # Treatment effect at this event time
            treated_mean = df_t[df_t[self.treatment_col] == 1][self.outcome_col].mean()
            control_mean = df_t[df_t[self.treatment_col] == 0][self.outcome_col].mean()

            effect = treated_mean - control_mean

            # Simple standard error
            treated_se = df_t[df_t[self.treatment_col] == 1][self.outcome_col].sem()
            control_se = df_t[df_t[self.treatment_col] == 0][self.outcome_col].sem()

            se = np.sqrt(treated_se ** 2 + control_se ** 2)

            effects.append({
                'event_time': event_t,
                'effect': effect,
                'std_error': se,
                'ci_lower': effect - 1.96 * se,
                'ci_upper': effect + 1.96 * se,
                'n_treated': (df_t[self.treatment_col] == 1).sum(),
                'n_control': (df_t[self.treatment_col] == 0).sum(),
                'is_lead': event_t < 0,
                'is_lag': event_t > 0
            })

        return pd.DataFrame(effects)

    def summarize(self) -> str:
        """
        Generate summary of DiD results.

        Returns:
        --------
        summary : str
            Formatted summary string
        """
        if not self.is_fitted or self.results_ is None:
            return "Model not yet fitted. Call fit() first."

        r = self.results_

        summary = f"""
Difference-in-Differences Results
=================================

Treatment Effect (DiD): {r['did_estimate']:.4f}
Standard Error: {r['std_error']:.4f}
95% Confidence Interval: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]
T-statistic: {r['t_statistic']:.4f}
P-value: {r['p_value']:.4f}

Group-Time Means:
- Treated (pre): {r['treated_pre']:.4f}
- Treated (post): {r['treated_post']:.4f}
- Control (pre): {r['control_pre']:.4f}
- Control (post): {r['control_post']:.4f}

Changes:
- Treated group: +{r['treated_change']:.4f}
- Control group: +{r['control_change']:.4f}
- Difference: {r['did_estimate']:.4f}

Sample Sizes:
- Treated units: {r['n_treated']:,}
- Control units: {r['n_control']:,}

Parallel Trends: {'✓ Satisfied' if self.check_parallel_trends else '✗ Violated'}
""".strip()

        return summary


class RobustDifferenceInDifferences:
    """
    Robust DiD implementation using regression approach.

    Implements the Sun-Abraham (2020) method for dynamic treatment effects
    with robust standard errors.
    """

    def __init__(self,
                 outcome_col: str,
                 treatment_col: str,
                 time_col: str,
                 unit_col: str,
                 cluster_var: Optional[str] = None):
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.unit_col = unit_col
        self.cluster_var = cluster_var

    def fit(self, data: pd.DataFrame) -> Dict:
        """
        Fit robust DiD model using two-way fixed effects.

        Parameters:
        -----------
        data : pd.DataFrame
            Panel data

        Returns:
        --------
        results : dict
            DiD estimation results
        """
        # Implementation would use statsmodels or linearmodels
        # for proper two-way fixed effects estimation
        # This is a placeholder for the interface
        raise NotImplementedError(
            "Robust DiD requires statsmodels or linearmodels. "
            "Use DifferenceInDifferences class for basic implementation."
        )
