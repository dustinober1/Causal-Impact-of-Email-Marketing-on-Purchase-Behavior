"""
Visualization utilities for saving plots to files.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Create output directory
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# Store all created figures
_created_figs = []


def auto_save(func):
    """Decorator to automatically save plots instead of showing them."""
    def wrapper(*args, **kwargs):
        # Call the original function
        result = func(*args, **kwargs)

        # Save the figure if one was created
        fig = plt.gcf()
        if fig:
            # Generate filename from function name
            filename = OUTPUT_DIR / f"{func.__name__}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
            _created_figs.append(filename)

            # Close to free memory
            plt.close(fig)

        return result
    return wrapper


def save_figure(fig, filename):
    """Save a matplotlib figure to the visualization directory."""
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    return filepath


def get_output_dir():
    """Get the path to the output directory."""
    return OUTPUT_DIR


def list_saved_plots():
    """List all saved plot files."""
    plots = list(OUTPUT_DIR.glob("*.png"))
    return sorted(plots)


# Monkey patch plt.show to save plots instead
_original_show = plt.show()


def patched_show():
    """Patched version of plt.show() that saves plots."""
    fig = plt.gcf()
    if fig:
        # Generate a generic filename
        import inspect
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name if 'func_name' in dir(frame) else 'plot'
        filename = f"{func_name}.png"
        save_figure(fig, filename)
        plt.close(fig)
    else:
        print("No active figure to save")


# Replace plt.show
plt.show = patched_show
