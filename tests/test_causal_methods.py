"""
Unit Tests for Causal Inference Methods

Tests for propensity score estimation, matching, weighting,
and difference-in-differences implementations.

Run with: pytest tests/test_causal_methods.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


class TestPropensityScoreEstimator:
    """Test PropensityScoreEstimator class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n)
        })
        # Create treatment with some correlation with X
        prob_treatment = 0.3 + 0.2 * X['x1'] + 0.1 * X['x2']
        prob_treatment = np.clip(prob_treatment, 0, 1)
        treatment = np.random.binomial(1, prob_treatment)
        return X, treatment

    def test_initialization(self):
        """Test model initialization."""
        estimator = PropensityScoreEstimator()
        assert estimator.use_scaling is True
        assert estimator.random_state is None

        estimator = PropensityScoreEstimator(
            use_scaling=False,
            random_state=42
        )
        assert estimator.use_scaling is False
        assert estimator.random_state == 42

    def test_fit_predict(self, sample_data):
        """Test fitting and prediction."""
        X, treatment = sample_data
        estimator = PropensityScoreEstimator(random_state=42)
        estimator.fit(X, treatment)

        # Check predictions
        scores = estimator.predict_proba(X)
        assert scores.shape == (len(X),)
        assert np.all((scores >= 0) & (scores <= 1))

        # Check AUC is reasonable
        eval_results = estimator.evaluate(X, treatment)
        assert 'auc' in eval_results
        assert 0.4 <= eval_results['auc'] <= 0.9  # Reasonable range

    def test_evaluation_metrics(self, sample_data):
        """Test evaluation metrics."""
        X, treatment = sample_data
        estimator = PropensityScoreEstimator(random_state=42)
        estimator.fit(X, treatment)

        eval_results = estimator.evaluate(X, treatment)

        # Check all metrics present
        required_keys = ['auc', 'propensity_mean', 'propensity_std',
                        'propensity_min', 'propensity_max']
        for key in required_keys:
            assert key in eval_results

        # Check values are reasonable
        assert 0 <= eval_results['auc'] <= 1
        assert 0 <= eval_results['propensity_mean'] <= 1
        assert eval_results['propensity_std'] >= 0
        assert 0 <= eval_results['propensity_min'] <= 1
        assert 0 <= eval_results['propensity_max'] <= 1

    def test_not_fitted_error(self, sample_data):
        """Test error when calling predict before fit."""
        X, treatment = sample_data
        estimator = PropensityScoreEstimator()

        with pytest.raises(ValueError, match="Model must be fitted"):
            estimator.predict_proba(X)

        with pytest.raises(ValueError, match="Model must be fitted"):
            estimator.evaluate(X, treatment)


class TestPropensityScoreMatcher:
    """Test PropensityScoreMatcher class."""

    @pytest.fixture
    def sample_matching_data(self):
        """Generate sample data for matching."""
        np.random.seed(42)
        n = 500

        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n)
        })

        # Create propensity scores
        ps_model = LogisticRegression()
        ps_model.fit(X, np.random.binomial(1, 0.5, n))
        propensity_scores = ps_model.predict_proba(X)[:, 1]

        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.normal(0, 1, n)

        return X, treatment, propensity_scores, outcome

    def test_initialization(self):
        """Test matcher initialization."""
        matcher = PropensityScoreMatcher()
        assert matcher.matching_type == 'nearest'
        assert matcher.caliper == 0.1
        assert matcher.replacement is False

        matcher = PropensityScoreMatcher(
            matching_type='radius',
            caliper=0.05,
            replacement=True
        )
        assert matcher.matching_type == 'radius'
        assert matcher.caliper == 0.05
        assert matcher.replacement is True

    def test_invalid_matching_type(self):
        """Test error for invalid matching type."""
        with pytest.raises(ValueError, match="matching_type must be"):
            PropensityScoreMatcher(matching_type='invalid')

    def test_fit_matching(self, sample_matching_data):
        """Test matching procedure."""
        X, treatment, propensity_scores, outcome = sample_matching_data

        matcher = PropensityScoreMatcher(caliper=0.2, random_state=42)
        matcher.fit(X, treatment, propensity_scores)

        # Check matched indices
        assert matcher.matched_treated_ is not None
        assert matcher.matched_control_ is not None
        assert len(matcher.matched_treated_) > 0
        assert len(matcher.matched_control_) > 0
        assert len(matcher.matched_treated_) == len(matcher.matched_control_)

        # Check balance statistics
        assert matcher.balance_stats_ is not None
        assert 'feature' in matcher.balance_stats_.columns
        assert 'std_diff' in matcher.balance_stats_.columns
        assert 'balanced' in matcher.balance_stats_.columns
        assert len(matcher.balance_stats_) == len(X.columns)

    def test_estimate_effect_continuous(self, sample_matching_data):
        """Test effect estimation for continuous outcome."""
        X, treatment, propensity_scores, outcome = sample_matching_data

        matcher = PropensityScoreMatcher(caliper=0.2, random_state=42)
        matcher.fit(X, treatment, propensity_scores)

        results = matcher.estimate_effect(pd.Series(outcome), outcome_type='continuous')

        # Check results structure
        required_keys = ['effect', 'std_error', 't_statistic', 'p_value',
                        'n_treated', 'n_control', 'ci_lower', 'ci_upper']
        for key in required_keys:
            assert key in results

        # Check values
        assert isinstance(results['effect'], (int, float))
        assert results['std_error'] >= 0
        assert results['n_treated'] > 0
        assert results['n_control'] > 0
        assert results['ci_lower'] <= results['ci_upper']

    def test_estimate_effect_binary(self, sample_matching_data):
        """Test effect estimation for binary outcome."""
        X, treatment, propensity_scores, _ = sample_matching_data

        # Create binary outcome
        outcome = np.random.binomial(1, 0.3, len(X))

        matcher = PropensityScoreMatcher(caliper=0.2, random_state=42)
        matcher.fit(X, treatment, propensity_scores)

        results = matcher.estimate_effect(pd.Series(outcome), outcome_type='binary')

        assert 'effect' in results
        assert 0 <= results['effect'] <= 1

    def test_not_fitted_error(self, sample_matching_data):
        """Test error when calling estimate before fit."""
        X, treatment, propensity_scores, outcome = sample_matching_data

        matcher = PropensityScoreMatcher()
        outcome_series = pd.Series(outcome)

        with pytest.raises(ValueError, match="Must call fit"):
            matcher.estimate_effect(outcome_series)

        with pytest.raises(ValueError, match="Must call fit"):
            matcher.bootstrap_ci(outcome_series)


class TestPropensityScoreWeighting:
    """Test PropensityScoreWeighting class."""

    @pytest.fixture
    def sample_weighting_data(self):
        """Generate sample data for IPW."""
        np.random.seed(42)
        n = 500

        propensity_scores = np.random.beta(2, 2, n)  # Between 0 and 1
        treatment = np.random.binomial(1, propensity_scores)
        outcome = np.random.normal(0, 1, n)

        return treatment, propensity_scores, outcome

    def test_initialization(self):
        """Test IPW initialization."""
        ipw = PropensityScoreWeighting()
        assert ipw.trimming_quantile is None
        assert ipw.trimmed_ is False

        ipw = PropensityScoreWeighting(trimming_quantile=0.01)
        assert ipw.trimming_quantile == 0.01

    def test_fit_weights(self, sample_weighting_data):
        """Test weight calculation."""
        treatment, propensity_scores, outcome = sample_weighting_data

        ipw = PropensityScoreWeighting()
        ipw.fit(pd.Series(treatment), propensity_scores)

        # Check weights
        assert ipw.weights_ is not None
        assert len(ipw.weights_) == len(treatment)
        assert np.all(ipw.weights_ > 0)  # Inverse weights should be positive

    def test_trimming(self, sample_weighting_data):
        """Test weight trimming."""
        treatment, propensity_scores, outcome = sample_weighting_data

        # Create extreme weights
        propensity_scores[0] = 0.001  # Very low
        propensity_scores[1] = 0.999  # Very high

        ipw = PropensityScoreWeighting(trimming_quantile=0.01)
        ipw.fit(pd.Series(treatment), propensity_scores)

        assert ipw.trimmed_ is True
        # Trimmed weights should have smaller max
        assert ipw.weights_.max() < 1 / 0.001

    def test_estimate_effect(self, sample_weighting_data):
        """Test IPW effect estimation."""
        treatment, propensity_scores, outcome = sample_weighting_data

        ipw = PropensityScoreWeighting()
        ipw.fit(pd.Series(treatment), pd.Series(propensity_scores))

        results = ipw.estimate_effect(pd.Series(outcome))

        # Check results structure
        required_keys = ['effect', 'std_error', 'n_effective',
                        'weight_mean', 'weight_max', 'weight_min', 'trimmed']
        for key in required_keys:
            assert key in results

        # Check values
        assert isinstance(results['effect'], (int, float))
        assert results['std_error'] >= 0
        assert results['n_effective'] > 0
        assert results['weight_mean'] > 0
        assert results['weight_max'] > results['weight_min']

    def test_not_fitted_error(self, sample_weighting_data):
        """Test error when calling estimate before fit."""
        _, propensity_scores, outcome = sample_weighting_data

        ipw = PropensityScoreWeighting()

        with pytest.raises(ValueError, match="Must call fit"):
            ipw.estimate_effect(pd.Series(outcome))


class TestDifferenceInDifferences:
    """Test DifferenceInDifferences class."""

    @pytest.fixture
    def sample_did_data(self):
        """Generate sample panel data for DiD."""
        np.random.seed(42)
        n_units = 100
        n_periods = 10

        data = []
        for unit in range(n_units):
            for period in range(n_periods):
                treated = 1 if unit > 50 else 0  # First 50 units treated
                post = 1 if period >= 5 else 0   # Treatment starts at period 5

                # Outcome with treatment effect
                outcome = (
                    10 +  # base outcome
                    2 * treated * post +  # treatment effect
                    np.random.normal(0, 1)  # noise
                )

                data.append({
                    'unit_id': unit,
                    'time': period,
                    'treated': treated,
                    'post': post,
                    'outcome': outcome
                })

        return pd.DataFrame(data)

    def test_initialization(self):
        """Test DiD initialization."""
        did = DifferenceInDifferences(
            outcome_col='y',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )
        assert did.outcome_col == 'y'
        assert did.treatment_col == 'treated'
        assert did.time_col == 'time'
        assert did.unit_col == 'unit_id'
        assert did.post_period == 5

    def test_prepare_data(self, sample_did_data):
        """Test data preparation."""
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )

        prepared = did._prepare_data(sample_did_data)

        # Check new columns
        assert 'post' in prepared.columns
        assert 'did_interaction' in prepared.columns
        assert prepared['post'].sum() > 0  # Some post periods

    def test_check_data_requirements(self, sample_did_data):
        """Test data requirement checks."""
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )

        # Valid data should not raise
        prepared = did._prepare_data(sample_did_data)
        did._check_data_requirements(prepared)  # Should not raise

        # Missing column should raise
        invalid_data = sample_did_data.drop('outcome', axis=1)
        with pytest.raises(ValueError, match="Column outcome not found"):
            did._check_data_requirements(invalid_data)

    def test_fit_basic(self, sample_did_data):
        """Test basic DiD estimation."""
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )

        results = did.fit(sample_did_data)

        # Check results structure
        required_keys = ['did_estimate', 'std_error', 't_statistic', 'p_value',
                        'ci_lower', 'ci_upper', 'n_treated', 'n_control',
                        'treated_pre', 'treated_post', 'control_pre', 'control_post']
        for key in required_keys:
            assert key in results

        # Check values
        assert isinstance(results['did_estimate'], (int, float))
        assert results['did_estimate'] > 0  # Should be positive given our data
        assert results['std_error'] >= 0
        assert results['p_value'] >= 0
        assert results['ci_lower'] <= results['ci_upper']

    def test_parallel_trends(self, sample_did_data):
        """Test parallel trends test."""
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )

        results = did.check_parallel_trends(sample_did_data)

        # Check results structure
        required_keys = ['parallel_trends_stat', 'std_error', 't_statistic',
                        'p_value', 'parallel_trends_satisfied', 'n_pre_periods']
        for key in required_keys:
            assert key in results

        # Check values
        assert isinstance(results['parallel_trends_stat'], (int, float))
        assert results['p_value'] >= 0
        assert isinstance(results['parallel_trends_satisfied'], bool)
        assert results['n_pre_periods'] > 0

    def test_event_study(self, sample_did_data):
        """Test event study analysis."""
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )

        results = did.event_study(sample_did_data, leads=2, lags=2)

        # Check results structure
        required_columns = ['event_time', 'effect', 'std_error',
                          'ci_lower', 'ci_upper', 'is_lead', 'is_lag']
        for col in required_columns:
            assert col in results.columns

        # Check values
        assert len(results) > 0
        assert results['event_time'].min() == -2
        assert results['event_time'].max() == 2

    def test_summarize(self, sample_did_data):
        """Test summary generation."""
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=5
        )

        # Before fit
        summary = did.summarize()
        assert "Model not yet fitted" in summary

        # After fit
        did.fit(sample_did_data)
        summary = did.summarize()

        # Check summary contains key results
        assert "Treatment Effect" in summary
        assert "Standard Error" in summary
        assert "Confidence Interval" in summary


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_psm_workflow(self):
        """Test complete propensity score matching workflow."""
        # Generate data
        np.random.seed(42)
        n = 300
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n)
        })

        # Create treatment
        ps_model = LogisticRegression()
        ps_model.fit(X, np.random.binomial(1, 0.5, n))
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        treatment = np.random.binomial(1, 0.5, n)

        # Create outcome
        outcome = (
            0 +  # base
            2 * treatment +  # treatment effect
            0.5 * X['x1'] + 0.3 * X['x2'] +  # confounding
            np.random.normal(0, 0.5, n)  # noise
        )

        # Step 1: Estimate propensity scores
        ps_estimator = PropensityScoreEstimator(random_state=42)
        ps_estimator.fit(X, pd.Series(treatment))
        eval_results = ps_estimator.evaluate(X, pd.Series(treatment))
        assert eval_results['auc'] > 0.5

        # Step 2: Perform matching
        matcher = PropensityScoreMatcher(caliper=0.2, random_state=42)
        matcher.fit(X, pd.Series(treatment), propensity_scores)

        # Check matching worked
        assert len(matcher.matched_treated_) > 0
        assert len(matcher.matched_control_) > 0

        # Step 3: Estimate effect
        results = matcher.estimate_effect(pd.Series(outcome))

        # Effect should be close to true effect (2.0)
        assert abs(results['effect'] - 2.0) < 1.0
        assert results['std_error'] > 0
        assert results['p_value'] >= 0

    def test_complete_ipw_workflow(self):
        """Test complete IPW workflow."""
        # Generate data
        np.random.seed(42)
        n = 300
        propensity_scores = np.random.beta(2, 2, n)
        treatment = np.random.binomial(1, propensity_scores)
        outcome = (
            0 +
            2 * treatment +
            np.random.normal(0, 0.5, n)
        )

        # Step 1: Calculate weights
        ipw = PropensityScoreWeighting()
        ipw.fit(pd.Series(treatment), propensity_scores)

        # Check weights
        assert np.all(ipw.weights_ > 0)

        # Step 2: Estimate effect
        results = ipw.estimate_effect(pd.Series(outcome))

        # Effect should be reasonable
        assert abs(results['effect'] - 2.0) < 1.5
        assert results['n_effective'] > 0

    def test_complete_did_workflow(self):
        """Test complete DiD workflow."""
        # Generate panel data
        np.random.seed(42)
        n_units = 50
        n_periods = 8

        data = []
        for unit in range(n_units):
            treated = 1 if unit > 25 else 0
            for period in range(n_periods):
                post = 1 if period >= 4 else 0
                outcome = (
                    10 +
                    2 * treated * post +
                    np.random.normal(0, 1)
                )
                data.append({
                    'unit_id': unit,
                    'time': period,
                    'treated': treated,
                    'outcome': outcome
                })

        df = pd.DataFrame(data)

        # Step 1: Fit DiD
        did = DifferenceInDifferences(
            outcome_col='outcome',
            treatment_col='treated',
            time_col='time',
            unit_col='unit_id',
            post_period=4
        )

        results = did.fit(df)

        # Check estimate is reasonable
        assert abs(results['did_estimate'] - 2.0) < 1.0

        # Step 2: Check parallel trends
        pt_results = did.check_parallel_trends(df)
        assert 'parallel_trends_satisfied' in pt_results

        # Step 3: Event study
        event_study_results = did.event_study(df, leads=2, lags=2)
        assert len(event_study_results) > 0


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_all_tests()
