"""
Main Execution Script for Grid Switch MPC

This script demonstrates the complete workflow for transformer switching optimization.

Usage:
    python run_grid_switch_mpc.py --data_path <path_to_excel_or_directory> --epochs 20 --outdir outputs
"""

import argparse
import sys
import time
from pathlib import Path

# Add grid_switch_mpc to path if not installed
sys.path.insert(0, str(Path(__file__).parent))

from data.load_and_clean import load_and_clean
from data.preprocessing import preprocess_data
from models.training import train_model_from_dataframe, create_datasets_and_loaders
from evaluation.backtesting import run_backtest
from config import CONFIG
import torch
from models.networks import TemporalFusionTransformer
from visualization.training_plots import plot_training_curves, plot_predictions, plot_multistep_predictions


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Grid Transformer Switching Optimization with Stochastic MPC'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to Excel file or directory containing Excel files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='./outputs/output',
        help='Output directory for results (default: ./outputs/output)'
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training and use saved model'
    )
    parser.add_argument(
        '--backtest_start',
        type=str,
        default='2022-01-01',
        help='Start date for backtesting period (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).'
    )
    parser.add_argument(
        '--backtest_end',
        type=str,
        default='2022-12-31',
        help='End date for backtesting period (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).'
    )
    parser.add_argument(
        '--evaluate_multistep',
        action='store_true',
        help='Run comprehensive multi-step evaluation after training/backtesting'
    )
    parser.add_argument(
        '--track_multistep_training',
        action='store_true',
        help='Track multi-step metrics during training (slower but informative)'
    )
    parser.add_argument(
        '--train_baselines',
        action='store_true',
        help='Train baseline models (Persistence + LSTM) for comparison'
    )
    parser.add_argument(
        '--compare_models',
        action='store_true',
        help='Compare TFT against baselines (requires trained models or --train_baselines)'
    )
    parser.add_argument(
        '--advanced_calibration',
        action='store_true',
        help='Run advanced calibration analysis with per-horizon PIT, conditional calibration, and interval scores'
    )
    parser.add_argument(
        '--analyze_mpc',
        action='store_true',
        help='Run MPC performance analysis after backtesting (loss decomposition, switching stats, regime analysis, timing)'
    )
    parser.add_argument(
        '--analyze_safety',
        action='store_true',
        help='Run safety layer analysis after backtesting (rule activations, temporal patterns, cost of safety, constraint validation)'
    )
    parser.add_argument(
        '--paper_figures',
        action='store_true',
        help='Generate publication-quality figures for the paper (weekly operation, transition zoom, heatmap, LDC, cumulative losses)'
    )

    args = parser.parse_args()

    # Set config flag for multistep tracking if requested
    if args.track_multistep_training:
        CONFIG['compute_multistep_metrics'] = True
        print("[INFO] Multi-step metric tracking enabled during training (slower)")

    # Start timer
    start_time = time.time()

    print("="*70)
    print("GRID TRANSFORMER SWITCHING OPTIMIZATION")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print(f"Training epochs: {args.epochs}")
    print(f"Output directory: {args.outdir}")
    print(f"Configuration: M={CONFIG['M']}, T={CONFIG['T']}, kappa={CONFIG['kappa']}")
    if args.evaluate_multistep:
        print(f"Multi-step evaluation: ENABLED")
    print("="*70)

    # Step 1: Load data
    print("\n[1/4] Loading data...")
    try:
        df = load_and_clean(args.data_path)
        print(f"[OK] Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        sys.exit(1)

    # Step 2: Preprocess data
    print("\n[2/4] Preprocessing data (feature engineering + normalization)...")
    try:
        df_processed, metadata = preprocess_data(df)
        n_features = len([col for col in df_processed.columns if col not in ['S_TOTAL', 'split']])
        print(f"[OK] Created {n_features} features")
        print(f"[OK] Train: {sum(df_processed['split']=='train')} | "
              f"Val: {sum(df_processed['split']=='val')} | "
              f"Test: {sum(df_processed['split']=='test')}")
    except Exception as e:
        print(f"[ERROR] Error preprocessing data: {e}")
        sys.exit(1)

    # Step 3: Train model
    model = None
    val_loader = None
    device = None
    if not args.skip_training:
        print(f"\n[3/4] Training Temporal Fusion Transformer ({args.epochs} epochs)...")
        try:
            model, val_loader, device = train_model_from_dataframe(
                df_processed,
                epochs=args.epochs,
                outdir=args.outdir,
                metadata=metadata
            )
            print("[OK] Model trained successfully")

            # Generate visualization plots
            if model is not None and val_loader is not None:
                print("\n[3.1/4] Generating training visualizations...")

                # Training curves (loss, metrics over epochs)
                try:
                    if hasattr(model, 'training_history'):
                        plot_training_curves(model, args.outdir)
                        print("[OK] Training curves generated")
                except Exception as e:
                    print(f"[WARNING] Could not generate training curves: {e}")

                # Prediction analysis plots
                try:
                    plot_predictions(model, val_loader, device, args.outdir, metadata=metadata)
                    print("[OK] Prediction analysis plots generated")
                except Exception as e:
                    print(f"[WARNING] Could not generate prediction plots: {e}")

                try:
                    plot_multistep_predictions(model, val_loader, device, args.outdir, metadata=metadata, num_examples=5)
                    print("[OK] Multistep prediction plots generated")
                except Exception as e:
                    print(f"[WARNING] Could not generate multistep plots: {e}")
        except Exception as e:
            print(f"[ERROR] Error training model: {e}")
            print("[WARNING] Continuing with persistence forecast...")
            model = None
            val_loader = None
            device = None
    else:
        print("\n[3/4] Skipping training. Checking for saved model...")
        model_path = Path(args.outdir) / 'model.pt'
        if model_path.exists():
            print(f"Found saved model: {model_path}")
            try:
                # Check for model config file (preferred method)
                config_path = Path(args.outdir) / 'model_config.json'
                if config_path.exists():
                    print(f"Loading model configuration from: {config_path}")
                    import json
                    with open(config_path, 'r') as f:
                        model_config = json.load(f)

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = TemporalFusionTransformer(
                        input_dim=model_config['input_dim'],
                        d_model=model_config['d_model'],
                        num_heads=model_config['num_heads'],
                        num_layers=model_config['num_layers'],
                        horizon=model_config['horizon'],
                        num_quantiles=model_config['num_quantiles'],
                    )
                    model.initialize_from_data(metadata['feature_columns'])
                    print(f"[OK] Model created with {model_config['input_dim']} input features")
                else:
                    # Fallback: try to use current preprocessing features
                    print("[WARNING] No model_config.json found!")
                    print("[WARNING] Using current preprocessing features (may cause shape mismatch)")
                    n_features = len(metadata['feature_columns'])
                    print(f"[INFO] Current preprocessing has {n_features} features")

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = TemporalFusionTransformer(
                        input_dim=n_features,
                        d_model=128,
                        num_heads=4,
                        num_layers=2,
                        horizon=CONFIG['H'],
                        num_quantiles=3
                    )
                    model.initialize_from_data(metadata['feature_columns'])

                # Load state dict
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                model.eval()
                print("[OK] Loaded saved model successfully.")

                # Load training history if available
                history_path = Path(args.outdir) / 'training_history.json'
                if history_path.exists():
                    try:
                        import json
                        with open(history_path, 'r') as f:
                            model.training_history = json.load(f)
                        print("[OK] Loaded training history")
                    except Exception as e:
                        print(f"[WARNING] Could not load training history: {e}")

                # Create validation data loader for plotting
                print("\n[3.1/4] Creating validation data loader for visualizations...")
                try:
                    _, val_loader, _ = create_datasets_and_loaders(df_processed)
                    print("[OK] Validation data loader created")

                    # Generate visualization plots
                    print("\n[3.2/4] Generating visualizations...")

                    # Training curves if history is available
                    try:
                        if hasattr(model, 'training_history'):
                            plot_training_curves(model, args.outdir)
                            print("[OK] Training curves generated")
                    except Exception as e:
                        print(f"[WARNING] Could not generate training curves: {e}")

                    # Prediction analysis plots
                    try:
                        plot_predictions(model, val_loader, device, args.outdir, metadata=metadata)
                        print("[OK] Prediction analysis plots generated")
                    except Exception as e:
                        print(f"[WARNING] Could not generate prediction plots: {e}")

                    try:
                        plot_multistep_predictions(model, val_loader, device, args.outdir, metadata=metadata, num_examples=5)
                        print("[OK] Multistep prediction plots generated")
                    except Exception as e:
                        print(f"[WARNING] Could not generate multistep plots: {e}")

                except Exception as e:
                    print(f"[WARNING] Could not create validation loader: {e}")
                    val_loader = None

            except Exception as e:
                print(f"[ERROR] Failed to load saved model: {e}")
                print("[WARNING] Continuing with persistence forecast...")
                model = None
                val_loader = None
                device = None
        else:
            print("[WARNING] No saved model found. Continuing with persistence forecast...")
            model = None

    # Step 4: Run backtesting simulation
    print(f"\n[4/4] Running backtesting simulation...")
    if args.backtest_start or args.backtest_end:
        period_str = f"from {args.backtest_start or 'beginning'} to {args.backtest_end or 'end'}"
        print(f"Backtesting period: {period_str}")
    try:
        results, results_df, switching_events = run_backtest(
            df_processed,
            model=model,
            epochs=args.epochs,
            outdir=args.outdir,
            metadata=metadata,
            backtest_start=args.backtest_start,
            backtest_end=args.backtest_end
        )
        print("[OK] Backtesting completed")
    except Exception as e:
        print(f"[ERROR] Error running backtest: {e}")
        sys.exit(1)

    # Step 4.5: MPC Performance Analysis (if requested)
    if args.analyze_mpc:
        print("\n" + "="*70)
        print("RUNNING MPC PERFORMANCE ANALYSIS")
        print("="*70)

        try:
            from evaluation.mpc_analysis import run_mpc_analysis

            logs_path = Path(args.outdir) / 'logs.csv'
            mpc_results = run_mpc_analysis(logs_path, switching_events, args.outdir)

            print(f"\n[OK] MPC analysis complete. Results saved to {args.outdir}/")
            print(f"     - mpc_analysis_results.json")
            print(f"     - mpc_loss_decomposition.png")
            print(f"     - mpc_switching_statistics.png")
            print(f"     - mpc_operational_regimes.png")
            if mpc_results.get('timing') is not None:
                print(f"     - mpc_computational_timing.png")
        except Exception as e:
            print(f"[WARNING] MPC analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 4.6: Safety Layer Analysis (if requested)
    if args.analyze_safety:
        print("\n" + "="*70)
        print("RUNNING SAFETY LAYER ANALYSIS")
        print("="*70)

        try:
            from evaluation.safety_analysis import run_safety_analysis

            logs_path = Path(args.outdir) / 'logs.csv'
            safety_results = run_safety_analysis(logs_path, switching_events, args.outdir)

            if safety_results is not None:
                print(f"\n[OK] Safety analysis complete. Results saved to {args.outdir}/")
                print(f"     - safety_analysis_results.json")
                print(f"     - safety_rule_activations.png")
                print(f"     - safety_temporal_patterns.png")
            else:
                print("[WARNING] Safety analysis returned no results (missing columns in logs)")
        except Exception as e:
            print(f"[WARNING] Safety analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 4.7: Publication Figures (if requested)
    if args.paper_figures:
        print("\n" + "="*70)
        print("GENERATING PUBLICATION FIGURES")
        print("="*70)

        try:
            from visualization.paper_figures import generate_paper_figures

            logs_path = Path(args.outdir) / 'logs.csv'
            generate_paper_figures(logs_path, args.outdir)

            print(f"\n[OK] Paper figures complete. Saved to {args.outdir}/")
            print(f"     - fig_weekly_operation.png/pdf")
            print(f"     - fig_transition_zoom.png/pdf")
            print(f"     - fig_switching_heatmap.png/pdf")
            print(f"     - fig_ldc_operating.png/pdf")
            print(f"     - fig_cumulative_losses.png/pdf")
        except Exception as e:
            print(f"[WARNING] Paper figure generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 5: Multi-step evaluation (if requested)
    if args.evaluate_multistep and model is not None:
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE MULTI-STEP EVALUATION")
        print("="*70)

        try:
            from evaluation.evaluation_runner import run_comprehensive_evaluation

            # Create test loader if not already exists
            if 'test_loader' not in locals() or test_loader is None:
                print("Creating test dataset loader...")
                _, _, test_loader = create_datasets_and_loaders(
                    df_processed, target_col='S_TOTAL', split='test'
                )

            eval_results = run_comprehensive_evaluation(
                model=model,
                data_loader=test_loader if 'test_loader' in locals() else val_loader,
                device=device,
                metadata=metadata,
                outdir=args.outdir,
                horizons='all',
                advanced_calibration=args.advanced_calibration
            )

            print(f"\n[OK] Multi-step evaluation complete. Results saved to {args.outdir}/")
            print(f"     - multistep_metrics_per_horizon.csv")
            print(f"     - calibration_data.json")
            print(f"     - stratified_metrics.json")
            print(f"     - multistep_point_metrics.png")
            print(f"     - multistep_prob_metrics.png")
            print(f"     - reliability_diagram.png")
            print(f"     - pit_histogram.png")

            if args.advanced_calibration:
                print(f"\n     Advanced Calibration Analysis:")
                print(f"     - advanced_calibration_results.json")
                print(f"     - per_horizon_calibration.png")
                print(f"     - pit_by_horizon.png")
                print(f"     - calibration_heatmap.png")
                print(f"     - conditional_calibration.png")
                print(f"     - interval_scores.png")

        except Exception as e:
            print(f"[WARNING] Multi-step evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 6: Train baseline models (if requested)
    persistence_model = None
    lstm_model = None

    if args.train_baselines:
        print("\n" + "="*70)
        print("TRAINING BASELINE MODELS")
        print("="*70)

        try:
            from models.baselines import PersistenceModel, SimpleLSTM, train_baseline_lstm

            # Persistence model (no training needed)
            persistence_model = PersistenceModel(horizon=CONFIG['H'])
            print("\n[1/2] Persistence Model")
            print("  Type: Naive persistence (repeats last value)")
            print("  Training: Not required")
            print("  [OK] Persistence model ready")

            # LSTM baseline
            print("\n[2/2] SimpleLSTM Baseline")

            # Create train/val loaders if not already available
            if 'train_loader' not in locals() or train_loader is None:
                print("  Creating data loaders...")
                train_loader, val_loader, _ = create_datasets_and_loaders(
                    df_processed, target_col='S_TOTAL'
                )

            lstm_model = train_baseline_lstm(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=len(metadata['feature_columns']),
                epochs=20,
                device=device if device else ('cuda' if torch.cuda.is_available() else 'cpu'),
                metadata=metadata,
                outdir=args.outdir
            )
            print("  [OK] LSTM baseline trained and saved")

        except Exception as e:
            print(f"[ERROR] Baseline training failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 7: Model comparison (if requested)
    if args.compare_models:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        try:
            from evaluation.model_comparison import compare_models, print_improvement_summary

            # Build models dict
            models_to_compare = {}

            # Add TFT if available
            if model is not None:
                models_to_compare['TFT'] = model
                print("[Model] TFT (Temporal Fusion Transformer)")
            else:
                print("[WARNING] TFT model not available, skipping from comparison")

            # Add Persistence
            if persistence_model is None and not args.train_baselines:
                # Load/create persistence model
                from models.baselines import PersistenceModel
                persistence_model = PersistenceModel(horizon=CONFIG['H'])

            if persistence_model is not None:
                models_to_compare['Persistence'] = persistence_model
                print("[Model] Persistence (Naive baseline)")

            # Add LSTM
            if lstm_model is None and not args.train_baselines:
                # Try to load saved LSTM
                lstm_path = Path(args.outdir) / 'baseline_lstm.pt'
                lstm_config_path = Path(args.outdir) / 'baseline_lstm_config.json'

                if lstm_path.exists() and lstm_config_path.exists():
                    print("[Loading] LSTM baseline from saved model...")
                    from models.baselines import load_baseline_lstm
                    lstm_model = load_baseline_lstm(
                        lstm_path,
                        lstm_config_path,
                        device=device if device else 'cpu'
                    )
                    print("  [OK] Loaded saved LSTM model")
                else:
                    print("[WARNING] LSTM model not found. Use --train_baselines to train it.")

            if lstm_model is not None:
                models_to_compare['LSTM'] = lstm_model
                print("[Model] LSTM (Simple 2-layer)")

            # Check we have models to compare
            if len(models_to_compare) < 2:
                print("[ERROR] Need at least 2 models for comparison")
                print("  Available models:", list(models_to_compare.keys()))
                print("  Use --train_baselines to train baseline models")
            else:
                # Create test loader if needed
                if 'test_loader' not in locals() or test_loader is None:
                    print("\nCreating test dataset loader...")
                    _, _, test_loader = create_datasets_and_loaders(
                        df_processed, target_col='S_TOTAL', split='test'
                    )

                # Run comparison
                comparison_results = compare_models(
                    models_dict=models_to_compare,
                    data_loader=test_loader,
                    device=device if device else 'cpu',
                    metadata=metadata,
                    outdir=args.outdir
                )

                # Print improvement summary
                if 'Persistence' in comparison_results:
                    print_improvement_summary(comparison_results, baseline_name='Persistence')

                print(f"\n[OK] Model comparison complete")
                print(f"     - model_comparison_aggregate.csv")
                print(f"     - model_comparison_horizons.csv")
                print(f"     - model_comparison_aggregate.png")
                print(f"     - model_comparison_horizons.png")

        except Exception as e:
            print(f"[ERROR] Model comparison failed: {e}")
            import traceback
            traceback.print_exc()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Execution time: {elapsed_time:.1f} seconds")
    print(f"Simulation period: {results['simulation_hours']:.1f} hours")

    annualized_losses = results['total_loss_kwh'] * (8760 / results['simulation_hours'])

    print(f"Annualized losses: {annualized_losses:.0f} kWh/year")
    print(f"Energy savings: {results['savings_kwh']:.0f} kWh ({results['savings_percentage']:.1f}% vs Always-3 baseline)")
    print(f"Switching activity: {results['num_switches']} switches ({results['switches_per_month']:.1f}/month)")
    print(f"System reliability: {results['overload_count']} overload events")
    print(f"Results saved to: {args.outdir}")
    print("\nGenerated files:")
    print("  - model.pt (trained model weights)")
    print("  - logs.csv (simulation timeline)")
    print("  - training_curves.png (training metrics)")
    print("  - probabilistic_metrics.png (forecast quality)")
    print("  - prediction_analysis.png (forecast validation)")
    print("  - predictions_multistep.png (multistep forecast examples)")
    print("  - trajectory_samples_example_*.png (uncertainty quantification, 5 examples)")
    print("  - backtest_analysis.png (switching timeline + savings)")
    print("  - kpi_summary.png (performance dashboard)")

    if args.evaluate_multistep:
        print("\n  Multi-step Evaluation Results:")
        print("  - multistep_metrics_per_horizon.csv (per-horizon metrics)")
        print("  - calibration_data.json (reliability & PIT data)")
        print("  - stratified_metrics.json (conditional performance)")
        print("  - evaluation_summary.json (aggregate summary)")
        print("  - multistep_point_metrics.png (MAE/RMSE/MAPE/R² vs horizon)")
        print("  - multistep_prob_metrics.png (CRPS/PICP/Sharpness vs horizon)")
        print("  - reliability_diagram.png (calibration curves)")
        print("  - pit_histogram.png (distributional calibration)")

        if args.advanced_calibration:
            print("\n  Advanced Calibration Analysis:")
            print("  - advanced_calibration_results.json (detailed per-horizon calibration)")
            print("  - per_horizon_calibration.png (reliability diagrams by horizon)")
            print("  - pit_by_horizon.png (PIT histograms with uniformity tests)")
            print("  - calibration_heatmap.png (calibration error heatmap)")
            print("  - conditional_calibration.png (calibration by load/time/day)")
            print("  - interval_scores.png (Winkler scores and interval metrics)")

    if args.analyze_mpc:
        print("\n  MPC Performance Analysis:")
        print("  - mpc_analysis_results.json (comprehensive metrics)")
        print("  - mpc_loss_decomposition.png (iron/copper loss comparison)")
        print("  - mpc_switching_statistics.png (switching frequency & dwell times)")
        print("  - mpc_operational_regimes.png (actual vs optimal N distribution)")
        print("  - mpc_computational_timing.png (solver performance)")

    if args.train_baselines or args.compare_models:
        print("\n  Baseline Comparison Results:")
        if args.train_baselines:
            print("  - baseline_lstm.pt (LSTM baseline model)")
            print("  - baseline_lstm_config.json (LSTM architecture)")
            print("  - baseline_lstm_history.json (LSTM training history)")
        if args.compare_models:
            print("  - model_comparison_aggregate.csv (model comparison summary)")
            print("  - model_comparison_horizons.csv (comparison by horizon)")
            print("  - model_comparison_aggregate.png (bar charts)")
            print("  - model_comparison_horizons.png (horizon curves)")

    print("="*70)


if __name__ == '__main__':
    main()
