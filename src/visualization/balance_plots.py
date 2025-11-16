"""
Balance Plots and Visualization Module

Reusable plotting functions for causal inference diagnostics:
- Love plots (covariate balance)
- Propensity score distributions
- Standardized mean differences
- QQ plots for balance assessment

Author: Causal Inference Toolkit
Date: 2025-11-16
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings


# Set style
plt.style.use('default')
sns.set_palette("husl")


class BalanceVisualizer:
    """
    Create comprehensive balance diagnostics visualizations.

    Example:
    --------
    >>> viz = BalanceVisualizer()
    >>> fig = viz.love_plot(balance_stats, save_path='love_plot.png')
    >>> fig = viz.propensity_score_distributions(pscores, treatment, save_path='ps_dist.png')
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'treated': '#1f77b4',
            'control': '#ff7f0e',
            'balanced': '#2ca02c',
            'unbalanced': '#d62728',
            'before': '#ff7f0e',
            'after': '#2ca02c'
        }

    def love_plot(self,
                  balance_stats: pd.DataFrame,
                  before_stats: Optional[pd.DataFrame] = None,
                  save_path: Optional[str] = None,
                  title: str = "Covariate Balance: Love Plot") -> plt.Figure:
        """
        Create Love plot showing standardized mean differences.

        Parameters:
        -----------
        balance_stats : pd.DataFrame
            Balance statistics with columns: 'feature', 'std_diff', 'balanced'
        before_stats : pd.DataFrame, optional
            Pre-matching balance statistics
        save_path : str, optional
            Path to save the figure
        title : str, default="Covariate Balance: Love Plot"
            Plot title

        Returns:
        --------
        fig : plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Ensure we have the right columns
        if 'std_diff' not in balance_stats.columns:
            raise ValueError("balance_stats must contain 'std_diff' column")

        # Sort by absolute standardized difference
        plot_data = balance_stats.copy()
        plot_data['abs_std_diff'] = plot_data['std_diff'].abs()
        plot_data = plot_data.sort_values('abs_std_diff', ascending=False)

        # Determine colors
        if 'balanced' in plot_data.columns:
            colors = ['#2ca02c' if b else '#d62728' for b in plot_data['balanced']]
        else:
            colors = [self.colors['unbalanced']] * len(plot_data)

        # Create horizontal bar plot
        y_pos = np.arange(len(plot_data))
        bars = ax.barh(y_pos, plot_data['std_diff'], color=colors, alpha=0.7)

        # Add threshold lines
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Good Balance (±0.1)')
        ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.05, color='orange', linestyle=':', alpha=0.5, label='Excellent Balance (±0.05)')
        ax.axvline(x=-0.05, color='orange', linestyle=':', alpha=0.5)

        # Add before matching if provided
        if before_stats is not None:
            before_data = before_stats.copy()
            before_data['abs_std_diff'] = before_data['std_diff'].abs()
            before_data = before_data.sort_values('abs_std_diff', ascending=False)

            ax.scatter(before_data['std_diff'], y_pos,
                      color='red', s=100, alpha=0.6, marker='X',
                      label='Before Matching')

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data['feature'], fontsize=10)
        ax.set_xlabel('Standardized Mean Difference', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Largest values at top

        # Add text annotations
        for i, (bar, val) in enumerate(zip(bars, plot_data['std_diff'])):
            ax.text(val + 0.01 if val >= 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                   fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Love plot saved to: {save_path}")

        return fig

    def propensity_score_distributions(self,
                                      propensity_scores: np.ndarray,
                                      treatment: np.Series,
                                      save_path: Optional[str] = None,
                                      title: str = "Propensity Score Distributions") -> plt.Figure:
        """
        Plot propensity score distributions by treatment group.

        Parameters:
        -----------
        propensity_scores : np.ndarray
            Propensity scores
        treatment : pd.Series
            Treatment indicator (0/1)
        save_path : str, optional
            Path to save the figure
        title : str, default="Propensity Score Distributions"
            Plot title

        Returns:
        --------
        fig : plt.Figure
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'propensity_score': propensity_scores,
            'treatment': treatment.values
        })

        # Histogram
        ax1.hist(plot_df[plot_df['treatment'] == 0]['propensity_score'],
                bins=50, alpha=0.7, label='Control', color=self.colors['control'], density=True)
        ax1.hist(plot_df[plot_df['treatment'] == 1]['propensity_score'],
                bins=50, alpha=0.7, label='Treated', color=self.colors['treated'], density=True)
        ax1.set_xlabel('Propensity Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution by Treatment Group', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Box plot
        box_data = [plot_df[plot_df['treatment'] == 0]['propensity_score'],
                   plot_df[plot_df['treatment'] == 1]['propensity_score']]
        box = ax2.boxplot(box_data, labels=['Control', 'Treated'], patch_artist=True)
        box['boxes'][0].set_facecolor(self.colors['control'])
        box['boxes'][1].set_facecolor(self.colors['treated'])
        ax2.set_ylabel('Propensity Score', fontsize=12)
        ax2.set_title('Box Plot Comparison', fontsize=14)
        ax2.grid(alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Propensity score distributions saved to: {save_path}")

        return fig

    def balance_summary_plot(self,
                            before_stats: pd.DataFrame,
                            after_stats: pd.DataFrame,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create before/after balance comparison plot.

        Parameters:
        -----------
        before_stats : pd.DataFrame
            Pre-matching balance statistics
        after_stats : pd.DataFrame
            Post-matching balance statistics
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        fig : plt.Figure
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # Ensure same features in both
        common_features = set(before_stats['feature']).intersection(set(after_stats['feature']))
        before_data = before_stats[before_stats['feature'].isin(common_features)].copy()
        after_data = after_stats[after_stats['feature'].isin(common_features)].copy()

        # Sort by before balance
        before_data = before_data.sort_values('std_diff', key=abs, ascending=False)

        # Bar plot of before
        bars1 = ax1.bar(range(len(before_data)), before_data['std_diff'],
                       color=self.colors['before'], alpha=0.7)
        ax1.set_xticks(range(len(before_data)))
        ax1.set_xticklabels(before_data['feature'], rotation=45, ha='right')
        ax1.set_ylabel('Standardized Mean Difference', fontsize=12)
        ax1.set_title('Before Matching', fontsize=14, fontweight='bold')
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)

        # Bar plot of after
        # Reorder after_data to match before_data
        after_data = after_data.set_index('feature').reindex(before_data['feature']).reset_index()

        bars2 = ax2.bar(range(len(after_data)), after_data['std_diff'],
                       color=self.colors['after'], alpha=0.7)
        ax2.set_xticks(range(len(after_data)))
        ax2.set_xticklabels(after_data['feature'], rotation=45, ha='right')
        ax2.set_ylabel('Standardized Mean Difference', fontsize=12)
        ax2.set_title('After Matching', fontsize=14, fontweight='bold')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)

        # Add improvement annotation
        improvement = (before_data['std_diff'].abs().mean() -
                      after_data['std_diff'].abs().mean())
        improvement_pct = (improvement / before_data['std_diff'].abs().mean()) * 100

        fig.suptitle(f'Balance Improvement: {improvement_pct:.1f}% reduction in imbalance',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Balance summary plot saved to: {save_path}")

        return fig

    def qq_plot_balance(self,
                       X: pd.DataFrame,
                       treatment: pd.Series,
                       propensity_scores: Optional[np.ndarray] = None,
                       n_features: int = 6,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create QQ plots to assess covariate balance.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        treatment : pd.Series
            Treatment indicator
        propensity_scores : np.ndarray, optional
            Propensity scores (if available)
        n_features : int, default=6
            Number of features to plot
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        fig : plt.Figure
            Matplotlib figure object
        """
        # Select features to plot
        features = X.columns[:n_features]
        n_rows = int(np.ceil(len(features) / 3))

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

        treated_mask = treatment == 1
        control_mask = treatment == 0

        for i, feature in enumerate(features):
            if i >= len(axes):
                break

            ax = axes[i]

            treated_vals = X[treated_mask][feature].dropna()
            control_vals = X[control_mask][feature].dropna()

            # Quantiles
            quantiles = np.linspace(0.01, 0.99, 99)
            treated_quantiles = np.quantile(treated_vals, quantiles)
            control_quantiles = np.quantile(control_vals, quantiles)

            # Plot
            ax.scatter(control_quantiles, treated_quantiles, alpha=0.6, s=20)
            min_val = min(control_quantiles.min(), treated_quantiles.min())
            max_val = max(control_quantiles.max(), treated_quantiles.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

            ax.set_xlabel(f'{feature} (Control)', fontsize=10)
            ax.set_ylabel(f'{feature} (Treated)', fontsize=10)
            ax.set_title(f'QQ Plot: {feature}', fontsize=11)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('QQ Plots: Covariate Balance Assessment', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"QQ plot saved to: {save_path}")

        return fig

    def common_support_plot(self,
                           propensity_scores: np.ndarray,
                           treatment: pd.Series,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
    Plot common support region for propensity scores.

    Parameters:
    -----------
    propensity_scores : np.ndarray
        Propensity scores
    treatment : pd.Series
        Treatment indicator
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure object
    """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate overlap
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]

        # Plot densities
        ax.hist(control_ps, bins=50, alpha=0.7, label='Control',
               color=self.colors['control'], density=True)
        ax.hist(treated_ps, bins=50, alpha=0.7, label='Treated',
               color=self.colors['treated'], density=True)

        # Common support region
        common_min = max(treated_ps.min(), control_ps.min())
        common_max = min(treated_ps.max(), control_ps.max())

        ax.axvspan(common_min, common_max, alpha=0.2, color='green',
                  label=f'Common Support: [{common_min:.3f}, {common_max:.3f}]')

        # Calculate overlap statistics
        overlap = (common_max - common_min) / (max(treated_ps.max(), control_ps.max()) -
                                              min(treated_ps.min(), control_ps.min()))

        ax.set_xlabel('Propensity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Common Support Analysis\nOverlap: {overlap:.1%}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Add statistics
        stats_text = f"""Treated:
  Mean: {treated_ps.mean():.3f}
  Std: {treated_ps.std():.3f}
  Min: {treated_ps.min():.3f}
  Max: {treated_ps.max():.3f}

Control:
  Mean: {control_ps.mean():.3f}
  Std: {control_ps.std():.3f}
  Min: {control_ps.min():.3f}
  Max: {control_ps.max():.3f}"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Common support plot saved to: {save_path}")

        return fig


def plot_treatment_effects_comparison(effects_dict: Dict[str, Dict],
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of treatment effects across methods.

    Parameters:
    -----------
    effects_dict : dict
        Dictionary with method names as keys and effect dicts as values
        Each effect dict should have: 'estimate', 'ci_lower', 'ci_upper', 'valid'
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    methods = list(effects_dict.keys())
    estimates = [effects_dict[m]['estimate'] for m in methods]
    ci_lower = [effects_dict[m].get('ci_lower', estimate) for m in methods]
    ci_upper = [effects_dict[m].get('ci_upper', estimate) for m in methods]
    valid = [effects_dict[m].get('valid', True) for m in methods]

    # Colors based on validity
    colors = ['#2ca02c' if v else '#d62728' for v in valid]

    # Plot points
    y_pos = np.arange(len(methods))
    ax.scatter(estimates, y_pos, c=colors, s=100, alpha=0.8, zorder=3)

    # Add error bars
    for i, (est, lower, upper) in enumerate(zip(estimates, ci_lower, ci_upper)):
        ax.plot([lower, upper], [i, i], color='black', linewidth=2, alpha=0.7)

    # Add vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Treatment Effect', fontsize=12)
    ax.set_title('Treatment Effect Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ca02c', label='Valid'),
                      Patch(facecolor='#d62728', label='Invalid')]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add effect values as text
    for i, (method, est, valid_flag) in enumerate(zip(methods, estimates, valid)):
        color = 'green' if valid_flag else 'red'
        ax.text(est + 0.001, i, f'{est:.3f}', va='center', ha='left',
               fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Treatment effects comparison saved to: {save_path}")

    return fig


def plot_heterogeneous_effects(effects_df: pd.DataFrame,
                              group_col: str,
                              effect_col: str,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot heterogeneous treatment effects by group.

    Parameters:
    -----------
    effects_df : pd.DataFrame
        DataFrame with group and effect columns
    group_col : str
        Name of group column
    effect_col : str
        Name of effect column
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by effect size
    plot_df = effects_df.sort_values(effect_col, ascending=False)

    # Color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(plot_df)))

    # Bar plot
    bars = ax.bar(range(len(plot_df)), plot_df[effect_col], color=colors)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_df[effect_col])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Customize
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df[group_col], rotation=45, ha='right')
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.set_title('Heterogeneous Treatment Effects by Group', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heterogeneous effects plot saved to: {save_path}")

    return fig


def plot_roi_analysis(roi_df: pd.DataFrame,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROI analysis by segment.

    Parameters:
    -----------
    roi_df : pd.DataFrame
        DataFrame with ROI metrics
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # ROI by segment
    bars1 = ax1.bar(roi_df['Segment'], roi_df['ROI (%)'], color='green', alpha=0.7)
    ax1.set_title('ROI by Segment', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ROI (%)')
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    # Sample size
    ax2.bar(roi_df['Segment'], roi_df['Sample_Size'], color='skyblue', alpha=0.7)
    ax2.set_title('Sample Size by Segment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Size')
    ax2.tick_params(axis='x', rotation=45)

    # Net profit
    bars3 = ax3.bar(roi_df['Segment'], roi_df['Net_Profit ($)'], color='gold', alpha=0.7)
    ax3.set_title('Net Profit by Segment', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Net Profit ($)')
    ax3.tick_params(axis='x', rotation=45)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'${height:.0f}', ha='center', va='bottom', fontsize=9)

    # Uplift
    ax4.bar(roi_df['Segment'], roi_df['Uplift (%)'], color='orange', alpha=0.7)
    ax4.set_title('Treatment Effect by Segment', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Uplift (%)')
    ax4.tick_params(axis='x', rotation=45)

    fig.suptitle('ROI Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROI analysis plot saved to: {save_path}")

    return fig
