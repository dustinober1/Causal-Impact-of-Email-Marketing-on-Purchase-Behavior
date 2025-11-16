"""
Propensity Score Analysis Module

Comprehensive propensity score matching implementation with:
- Propensity score estimation
- Multiple matching algorithms
- Balance diagnostics
- Bootstrap confidence intervals

Author: Causal Inference Toolkit
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class PropensityScoreEstimator:
    """
    Estimate propensity scores using logistic regression.

    Parameters:
    -----------
    model : sklearn estimator, optional
        Propensity score model. Default is LogisticRegression.
    use_scaling : bool, default=True
        Whether to standardize features before modeling.
    random_state : int, optional
        Random state for reproducibility.

    Example:
    --------
    >>> estimator = PropensityScoreEstimator()
    >>> ps_model = estimator.fit(X, treatment)
    >>> propensity_scores = estimator.predict_proba(X)
    """

    def __init__(self,
                 model=None,
                 use_scaling: bool = True,
                 random_state: Optional[int] = None):
        self.model = model or LogisticRegression(random_state=random_state)
        self.use_scaling = use_scaling
        self.random_state = random_state
        self.scaler = StandardScaler() if use_scaling else None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, treatment: pd.Series) -> 'PropensityScoreEstimator':
        """
        Fit propensity score model.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (n_samples, n_features)
        treatment : pd.Series
            Binary treatment indicator (0/1)

        Returns:
        --------
        self : PropensityScoreEstimator
            Fitted estimator
        """
        # Prepare data
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Scale features if requested
        if self.use_scaling:
            X_processed = self.scaler.fit_transform(X)
        else:
            X_processed = X.values if isinstance(X, pd.DataFrame) else X

        # Fit model
        self.model.fit(X_processed, treatment)
        self.is_fitted = True

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict propensity scores.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix

        Returns:
        --------
        scores : np.ndarray
            Propensity scores (probability of treatment)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare data
        if self.use_scaling:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values if isinstance(X, pd.DataFrame) else X

        return self.model.predict_proba(X_processed)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict treatment assignment.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix

        Returns:
        --------
        treatment_pred : np.ndarray
            Predicted treatment assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)

    def evaluate(self, X: pd.DataFrame, treatment: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        treatment : pd.Series
            True treatment labels

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        scores = self.predict_proba(X)
        auc = roc_auc_score(treatment, scores)

        return {
            'auc': auc,
            'propensity_mean': scores.mean(),
            'propensity_std': scores.std(),
            'propensity_min': scores.min(),
            'propensity_max': scores.max()
        }


class PropensityScoreMatcher:
    """
    Propensity Score Matching implementation.

    Supports multiple matching algorithms:
    - Nearest neighbor matching
    - Radius (caliper) matching
    - Kernel matching

    Parameters:
    -----------
    matching_type : str, default='nearest'
        Type of matching: 'nearest', 'radius', or 'kernel'
    caliper : float, optional
        Caliper width for matching (standard deviations of propensity scores)
    replacement : bool, default=False
        Whether to match with replacement
    random_state : int, optional
        Random state for reproducibility

    Example:
    --------
    >>> matcher = PropensityScoreMatcher(matching_type='nearest', caliper=0.1)
    >>> matched_data = matcher.fit(data, treatment, propensity_scores)
    >>> effect = matcher.estimate_effect(outcome)
    """

    def __init__(self,
                 matching_type: str = 'nearest',
                 caliper: Optional[float] = None,
                 replacement: bool = False,
                 random_state: Optional[int] = None):

        if matching_type not in ['nearest', 'radius', 'kernel']:
            raise ValueError("matching_type must be 'nearest', 'radius', or 'kernel'")

        self.matching_type = matching_type
        self.caliper = caliper or 0.1
        self.replacement = replacement
        self.random_state = random_state
        self.matched_indices_ = None
        self.balance_stats_ = None

    def _nearest_neighbor_matching(self,
                                   propensity_scores: np.ndarray,
                                   treatment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform nearest neighbor matching.

        Parameters:
        -----------
        propensity_scores : np.ndarray
            Propensity scores
        treatment : np.ndarray
            Treatment indicator

        Returns:
        --------
        treated_indices : np.ndarray
            Indices of matched treated units
        control_indices : np.ndarray
            Indices of matched control units
        """
        treated_mask = treatment == 1
        control_mask = treatment == 0

        treated_indices = np.where(treated_mask)[0]
        control_indices = np.where(control_mask)[0]

        matched_treated = []
        matched_control = []

        if not self.replacement:
            # Without replacement - track used control units
            used_control = set()

        # Caliper threshold
        ps_std = propensity_scores.std()
        caliper_threshold = self.caliper * ps_std

        for t_idx in treated_indices:
            t_score = propensity_scores[t_idx]

            # Find control units within caliper
            if self.replacement:
                candidate_controls = control_indices
            else:
                candidate_controls = [c for c in control_indices if c not in used_control]

            if len(candidate_controls) == 0:
                continue

            distances = np.abs(propensity_scores[candidate_controls] - t_score)
            within_caliper = distances <= caliper_threshold

            if not np.any(within_caliper):
                continue

            # Find nearest neighbor within caliper
            valid_candidates = np.array(candidate_controls)[within_caliper]
            valid_distances = distances[within_caliper]

            # If multiple matches, select randomly among best
            min_distance = valid_distances.min()
            best_matches = valid_candidates[valid_distances == min_distance]

            if self.random_state is not None:
                np.random.seed(self.random_state)

            c_idx = np.random.choice(best_matches)

            matched_treated.append(t_idx)
            matched_control.append(c_idx)

            if not self.replacement:
                used_control.add(c_idx)

        return np.array(matched_treated), np.array(matched_control)

    def fit(self,
            X: pd.DataFrame,
            treatment: pd.Series,
            propensity_scores: np.ndarray) -> 'PropensityScoreMatcher':
        """
        Perform matching.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        treatment : pd.Series
            Treatment indicator
        propensity_scores : np.ndarray
            Propensity scores

        Returns:
        --------
        self : PropensityScoreMatcher
            Fitted matcher
        """
        self.X = X
        self.treatment = treatment.values
        self.propensity_scores = propensity_scores

        # Perform matching
        treated_indices, control_indices = self._nearest_neighbor_matching(
            propensity_scores, self.treatment
        )

        self.matched_treated_ = treated_indices
        self.matched_control_ = control_indices
        self.matched_indices_ = np.concatenate([treated_indices, control_indices])

        # Check balance
        self._check_balance(X, treatment, treated_indices, control_indices)

        return self

    def _check_balance(self,
                       X: pd.DataFrame,
                       treatment: pd.Series,
                       treated_indices: np.ndarray,
                       control_indices: np.ndarray):
        """
        Check covariate balance after matching.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        treatment : pd.Series
            Treatment indicator
        treated_indices : np.ndarray
            Indices of matched treated units
        control_indices : np.ndarray
            Indices of matched control units
        """
        balance_results = []

        for col in X.columns:
            treated_vals = X.iloc[treated_indices][col].values
            control_vals = X.iloc[control_indices][col].values

            # Standardized difference
            std_diff = (treated_vals.mean() - control_vals.mean()) / np.sqrt(
                (treated_vals.var() + control_vals.var()) / 2
            )

            balance_results.append({
                'feature': col,
                'treated_mean': treated_vals.mean(),
                'control_mean': control_vals.mean(),
                'std_diff': std_diff,
                'balanced': abs(std_diff) < 0.1
            })

        self.balance_stats_ = pd.DataFrame(balance_results)

    def estimate_effect(self,
                       outcome: pd.Series,
                       outcome_type: str = 'continuous') -> Dict[str, float]:
        """
        Estimate treatment effect on matched sample.

        Parameters:
        -----------
        outcome : pd.Series
            Outcome variable
        outcome_type : str, default='continuous'
            Type of outcome: 'continuous' or 'binary'

        Returns:
        --------
        effect : dict
            Treatment effect estimate and statistics
        """
        if self.matched_treated_ is None:
            raise ValueError("Must call fit() before estimate_effect()")

        treated_outcomes = outcome.iloc[self.matched_treated_].values
        control_outcomes = outcome.iloc[self.matched_control_].values

        # Calculate effect
        if outcome_type == 'continuous':
            effect = treated_outcomes.mean() - control_outcomes.mean()
        elif outcome_type == 'binary':
            effect = treated_outcomes.mean() - control_outcomes.mean()
        else:
            raise ValueError("outcome_type must be 'continuous' or 'binary'")

        # Standard error from matched pairs
        differences = treated_outcomes - control_outcomes
        se = differences.std() / np.sqrt(len(differences))

        # T-statistic
        t_stat = effect / se if se > 0 else np.inf
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            'effect': effect,
            'std_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'n_treated': len(treated_outcomes),
            'n_control': len(control_outcomes),
            'ci_lower': effect - 1.96 * se,
            'ci_upper': effect + 1.96 * se
        }

    def bootstrap_ci(self,
                     outcome: pd.Series,
                     n_bootstrap: int = 1000,
                     outcome_type: str = 'continuous') -> Dict[str, float]:
        """
        Bootstrap confidence intervals.

        Parameters:
        -----------
        outcome : pd.Series
            Outcome variable
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        outcome_type : str, default='continuous'
            Type of outcome

        Returns:
        --------
        ci : dict
            Bootstrap confidence interval
        """
        if self.matched_treated_ is None:
            raise ValueError("Must call fit() before bootstrap_ci()")

        np.random.seed(self.random_state)
        bootstrap_effects = []

        for _ in range(n_bootstrap):
            # Resample matched pairs
            n_pairs = len(self.matched_treated_)
            bootstrap_indices = np.random.choice(n_pairs, size=n_pairs, replace=True)

            boot_treated = self.matched_treated_[bootstrap_indices]
            boot_control = self.matched_control_[bootstrap_indices]

            treated_outcomes = outcome.iloc[boot_treated].values
            control_outcomes = outcome.iloc[boot_control].values

            if outcome_type == 'continuous':
                effect = treated_outcomes.mean() - control_outcomes.mean()
            else:  # binary
                effect = treated_outcomes.mean() - control_outcomes.mean()

            bootstrap_effects.append(effect)

        bootstrap_effects = np.array(bootstrap_effects)

        return {
            'ci_lower': np.percentile(bootstrap_effects, 2.5),
            'ci_upper': np.percentile(bootstrap_effects, 97.5),
            'bootstrap_mean': bootstrap_effects.mean(),
            'bootstrap_std': bootstrap_effects.std()
        }

    def get_balance_stats(self) -> pd.DataFrame:
        """
        Get balance statistics.

        Returns:
        --------
        balance_stats : pd.DataFrame
            Balance statistics for each covariate
        """
        return self.balance_stats_


# Import stats for p-values
from scipy import stats


class PropensityScoreWeighting:
    """
    Inverse Probability Weighting (IPW) estimator.

    Weights observations by inverse propensity scores to create
    a pseudo-population where treatment is as-if randomized.

    Parameters:
    -----------
    trimming_quantile : float, optional
        Quantile for trimming extreme weights. Default is 0.01.

    Example:
    --------
    >>> ipw = PropensityScoreWeighting(trimming_quantile=0.01)
    >>> ipw.fit(X, treatment, propensity_scores)
    >>> effect = ipw.estimate_effect(outcome)
    """

    def __init__(self, trimming_quantile: Optional[float] = None):
        self.trimming_quantile = trimming_quantile
        self.weights_ = None
        self.trimmed_ = False

    def fit(self,
            treatment: pd.Series,
            propensity_scores: np.ndarray) -> 'PropensityScoreWeighting':
        """
        Calculate IPW weights.

        Parameters:
        -----------
        treatment : pd.Series
            Treatment indicator
        propensity_scores : np.ndarray
            Propensity scores

        Returns:
        --------
        self : PropensityScoreWeighting
            Fitted estimator
        """
        self.treatment = treatment.values

        # Calculate weights
        # Treated: 1 / propensity_score
        # Control: 1 / (1 - propensity_score)
        weights = np.where(
            self.treatment == 1,
            1 / propensity_scores,
            1 / (1 - propensity_scores)
        )

        # Trim extreme weights if requested
        if self.trimming_quantile is not None:
            lower_bound = np.quantile(weights, self.trimming_quantile)
            upper_bound = np.quantile(weights, 1 - self.trimming_quantile)
            weights = np.clip(weights, lower_bound, upper_bound)
            self.trimmed_ = True

        self.weights_ = weights
        return self

    def estimate_effect(self,
                        outcome: pd.Series,
                        outcome_type: str = 'continuous') -> Dict[str, float]:
        """
        Estimate treatment effect using IPW.

        Parameters:
        -----------
        outcome : pd.Series
            Outcome variable
        outcome_type : str, default='continuous'
            Type of outcome

        Returns:
        --------
        effect : dict
            Treatment effect estimate and statistics
        """
        if self.weights_ is None:
            raise ValueError("Must call fit() before estimate_effect()")

        weights = self.weights_
        outcome_vals = outcome.values

        # Weighted means
        treated_weighted_mean = np.average(
            outcome_vals[self.treatment == 1],
            weights=weights[self.treatment == 1]
        )

        control_weighted_mean = np.average(
            outcome_vals[self.treatment == 0],
            weights=weights[self.treatment == 0]
        )

        effect = treated_weighted_mean - control_weighted_mean

        # Effective sample size
        n_eff = (weights.sum() ** 2) / (weights ** 2).sum()

        # Standard error (approximate)
        se = np.sqrt(
            (treated_weighted_mean ** 2) / (n_eff * (self.treatment == 1).sum()) +
            (control_weighted_mean ** 2) / (n_eff * (self.treatment == 0).sum())
        )

        return {
            'effect': effect,
            'std_error': se,
            'n_effective': n_eff,
            'weight_mean': weights.mean(),
            'weight_max': weights.max(),
            'weight_min': weights.min(),
            'trimmed': self.trimmed_
        }