"""
Example Script: Generate Data Analysis Visualizations
======================================================

This script demonstrates how to generate all 6 data analysis visualization figures
for power system load data.

Usage:
    python examples/generate_data_analysis.py <path_to_data> [output_dir]

Example:
    python examples/generate_data_analysis.py data/sample.xlsx outputs/analysis
    python examples/generate_data_analysis.py data/ outputs/analysis
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.load_and_clean import load_and_clean
from visualization.data_analysis_plots import (
    generate_all_analysis_plots,
    plot_load_duration_curve,
    plot_temporal_heatmap,
    plot_distribution_analysis,
    plot_seasonal_decomposition,
    plot_power_factor_analysis,
    plot_yearly_seasonality
)


def main():
    """Main function to generate data analysis plots."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate data analysis visualization figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all 6 analysis plots
  python examples/generate_data_analysis.py data/sample.xlsx

  # Specify custom output directory
  python examples/generate_data_analysis.py data/sample.xlsx outputs/my_analysis

  # Generate individual plots
  python examples/generate_data_analysis.py data/sample.xlsx --plot ldc
  python examples/generate_data_analysis.py data/sample.xlsx --plot heatmap
  python examples/generate_data_analysis.py data/sample.xlsx --plot distribution
  python examples/generate_data_analysis.py data/sample.xlsx --plot decomposition
  python examples/generate_data_analysis.py data/sample.xlsx --plot powerfactor
  python examples/generate_data_analysis.py data/sample.xlsx --plot seasonality
        """
    )

    parser.add_argument('data_path', type=str,
                       help='Path to Excel file or directory containing data')

    parser.add_argument('output_dir', type=str, nargs='?',
                       default='outputs/data_analysis',
                       help='Output directory for plots (default: outputs/data_analysis)')

    parser.add_argument('--plot', type=str, choices=['all', 'ldc', 'heatmap',
                                                     'distribution', 'decomposition',
                                                     'powerfactor', 'seasonality'],
                       default='all',
                       help='Which plot to generate (default: all)')

    args = parser.parse_args()

    # Load data
    print("\n" + "="*70)
    print("Loading Data...")
    print("="*70)
    print(f"Data path: {args.data_path}")

    try:
        df = load_and_clean(args.data_path)
        print(f"[OK] Data loaded successfully: {len(df)} samples")
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {str(e)}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots based on selection
    try:
        if args.plot == 'all':
            # Generate all 6 plots
            generate_all_analysis_plots(df, output_dir)

        elif args.plot == 'ldc':
            print("\nGenerating Load Duration Curve...")
            plot_load_duration_curve(df, output_dir)
            print(f"[OK] Plot saved to {output_dir}")

        elif args.plot == 'heatmap':
            print("\nGenerating Temporal Heatmap...")
            plot_temporal_heatmap(df, output_dir)
            print(f"[OK] Plot saved to {output_dir}")

        elif args.plot == 'distribution':
            print("\nGenerating Distribution Analysis...")
            plot_distribution_analysis(df, output_dir)
            print(f"[OK] Plot saved to {output_dir}")

        elif args.plot == 'decomposition':
            print("\nGenerating Seasonal Decomposition...")
            plot_seasonal_decomposition(df, output_dir)
            print(f"[OK] Plot saved to {output_dir}")

        elif args.plot == 'powerfactor':
            print("\nGenerating Power Factor Analysis...")
            plot_power_factor_analysis(df, output_dir)
            print(f"[OK] Plot saved to {output_dir}")

        elif args.plot == 'seasonality':
            print("\nGenerating Yearly Seasonality...")
            plot_yearly_seasonality(df, output_dir)
            print(f"[OK] Plot saved to {output_dir}")

    except Exception as e:
        print(f"\n[ERROR] Error generating plots: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
