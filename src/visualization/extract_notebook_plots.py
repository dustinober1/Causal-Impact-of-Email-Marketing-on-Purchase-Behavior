"""
Extract all plots from executed Jupyter notebooks.
"""

import json
import base64
from pathlib import Path
import re


def extract_plots_from_notebook(notebook_path, output_dir):
    """
    Extract all matplotlib plots from a Jupyter notebook.

    Parameters:
    -----------
    notebook_path : Path
        Path to the executed notebook
    output_dir : Path
        Directory to save extracted plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Counter for plots
    plot_num = 0

    # Track which notebook this is
    notebook_name = notebook_path.stem

    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            # Look for matplotlib outputs
            for output in cell.get('outputs', []):
                # Handle display_data with image data
                if output.get('output_type') == 'display_data':
                    # Look for PNG data
                    if 'image/png' in output.get('data', {}):
                        plot_num += 1
                        image_data = output['data']['image/png']

                        # Decode and save
                        image_bytes = base64.b64decode(image_data)
                        output_file = output_dir / f"{notebook_name}_plot_{plot_num:03d}.png"

                        with open(output_file, 'wb') as f:
                            f.write(image_bytes)

                        print(f"✅ Extracted: {output_file}")

                    # Also check for SVG
                    elif 'image/svg+xml' in output.get('data', {}):
                        plot_num += 1
                        svg_data = output['data']['image/svg+xml']
                        output_file = output_dir / f"{notebook_name}_plot_{plot_num:03d}.svg"

                        with open(output_file, 'w') as f:
                            f.write(svg_data)

                        print(f"✅ Extracted: {output_file}")

    return plot_num


def extract_all_plots():
    """Extract plots from all executed notebooks."""
    project_root = Path(__file__).parent.parent.parent
    notebooks_dir = project_root / 'notebooks'
    output_dir = Path(__file__).parent

    # Find all notebooks
    notebooks = list(notebooks_dir.glob("*.ipynb"))

    print("=" * 70)
    print("EXTRACTING PLOTS FROM NOTEBOOKS")
    print("=" * 70)

    total_plots = 0

    for notebook in sorted(notebooks):
        print(f"\nProcessing: {notebook.name}")
        print("-" * 70)

        try:
            num_plots = extract_plots_from_notebook(notebook, output_dir)
            total_plots += num_plots
            print(f"Found {num_plots} plots in {notebook.name}")
        except Exception as e:
            print(f"❌ Error processing {notebook.name}: {e}")

    print("\n" + "=" * 70)
    print(f"TOTAL PLOTS EXTRACTED: {total_plots}")
    print("=" * 70)

    # List all saved plots
    plots = list(output_dir.glob("*.png")) + list(output_dir.glob("*.svg"))
    plots = sorted(plots)

    print(f"\nAll saved plots in {output_dir}:")
    print("-" * 70)
    for plot in plots:
        print(f"  - {plot.name}")

    return total_plots


if __name__ == "__main__":
    extract_all_plots()
