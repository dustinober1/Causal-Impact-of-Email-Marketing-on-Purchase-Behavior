"""
IPython configuration to automatically save matplotlib plots.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

# Set matplotlib to use SVG for better quality in notebooks
set_matplotlib_formats('png', 'retina')

# Create a custom hook to save plots
import sys
import os
from pathlib import Path

# Add visualization path
viz_dir = Path(__file__).parent / 'src' / 'visualization'
viz_dir.mkdir(parents=True, exist_ok=True)

# Counter for plot numbering
plot_counter = [0]


def custom_show():
    """Custom show function that saves plots."""
    fig = plt.gcf()
    if fig:
        plot_counter[0] += 1
        filename = viz_dir / f"notebook_plot_{plot_counter[0]:03d}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Saved plot to {filename}")
        plt.close(fig)


# Monkey patch plt.show
original_show = plt.show
plt.show = custom_show

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.ioff()  # Turn off interactive mode
