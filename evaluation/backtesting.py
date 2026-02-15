"""
Backtesting Module

Runs complete backtesting simulation with trained model:
- Generates forecasts using trained model or persistence
- Executes stochastic MPC with safety layer
- Tracks switching events, overloads, and losses
- Computes KPIs and generates visualizations
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import CONFIG
from control.mpc import stochastic_mpc
from control.safety import safety_layer
from control.power_model import transformer_loss_model, compute_breakpoints_and_thresholds
from control.forecasting import sample_trajectories_from_quantiles
from visualization.results_plots import plot_trajectory_samples


def run_backtest(df, model=None, epochs=20, outdir='./outputs', metadata=None, backtest_start=None, backtest_end=None):
    """
    Run complete backtesting simulation

    Args:
        df: Preprocessed dataframe with features and target
        model: Trained forecasting model (optional)
        epochs: Number of training epochs (for reference)
        outdir: Output directory for results
        metadata: Preprocessing metadata
        backtest_start: Start date/time for backtesting (str, format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
        backtest_end: End date/time for backtesting (str, format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
    """

    print("\n" + "="*60)
    print("RUNNING BACKTESTING SIMULATION")
    print("="*60)

    # Ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract normalization stats for denormalization
    target_mean = 0.0
    target_std = 1.0
    if metadata is not None and 'normalization' in metadata:
        target_mean = metadata['normalization'].get('target_mean', 0.0)
        target_std = metadata['normalization'].get('target_std', 1.0)
        print(f"Denormalization parameters: mean={target_mean:.2f}, std={target_std:.2f}")

    # Use all available data for backtesting (or filter by date range if specified)
    test_data = df.copy()

    # Filter by date range if specified
    if backtest_start is not None or backtest_end is not None:
        original_length = len(test_data)

        if backtest_start is not None:
            start_time = pd.to_datetime(backtest_start)
            test_data = test_data[test_data.index >= start_time]
            print(f"Filtering data from: {start_time}")

        if backtest_end is not None:
            end_time = pd.to_datetime(backtest_end)
            test_data = test_data[test_data.index <= end_time]
            print(f"Filtering data until: {end_time}")

        filtered_length = len(test_data)
        print(f"Filtered {original_length} samples down to {filtered_length} samples")

        if filtered_length == 0:
            raise ValueError(f"No data found in specified date range: {backtest_start} to {backtest_end}")

    print(f"Backtesting period: {test_data.index[0]} to {test_data.index[-1]}")
    print(f"Backtesting samples: {len(test_data)}")

    # Prepare model for inference if available
    device = 'cpu'
    if model is not None:
        if TORCH_AVAILABLE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()
            print(f"Using trained model for forecasting (device: {device})")
        else:
            print("WARNING: PyTorch not available, model will be ignored")
            model = None
    else:
        print("WARNING: No trained model provided, using simple persistence forecast")

    # Get feature columns for model input
    feature_cols = [col for col in df.columns if col not in ['S_TOTAL', 'split']]

    # Initialize simulation
    thresholds = compute_breakpoints_and_thresholds()
    simulation_state = {'N': 1, 'dwell_time': 10, 'last_action': 'on'}
    results = []
    switching_events = []

    # Get the position of test data in full dataframe
    test_start_idx_result = df.index.get_loc(test_data.index[0])
    # Handle case where get_loc returns a slice (duplicate indices)
    if isinstance(test_start_idx_result, slice):
        test_start_idx = test_start_idx_result.start
    else:
        test_start_idx = test_start_idx_result

    # Import generate_summary_plots here to avoid circular import
    from visualization.results_plots import generate_summary_plots

    # Counter for trajectory sample plots (only plot first few examples)
    trajectory_plot_counter = 0
    max_trajectory_plots = 5  # Plot first 5 examples

    # Main simulation loop with progress bar
    simulation_pbar = tqdm(test_data.iterrows(), total=len(test_data), desc="Backtesting Simulation", unit="step")

    for i, (timestamp, row) in enumerate(simulation_pbar):

        # CRITICAL: Denormalize current load from normalized dataframe values
        current_load_normalized = row['S_TOTAL']
        current_load = current_load_normalized * target_std + target_mean  # Convert to MVA

        # Calculate actual position in full dataframe
        actual_idx = test_start_idx + i

        # Generate forecast samples using model or simple persistence
        if model is not None and TORCH_AVAILABLE and actual_idx >= CONFIG['sequence_length']:
            # Use model for forecasting
            # Get historical sequence from full dataframe
            start_idx = actual_idx - CONFIG['sequence_length']
            hist_data = df.iloc[start_idx:actual_idx]

            if len(hist_data) >= CONFIG['sequence_length']:

                # Prepare features and target
                features = torch.FloatTensor(hist_data[feature_cols].values[-CONFIG['sequence_length']:]).unsqueeze(0).to(device)
                hist_target = torch.FloatTensor(hist_data['S_TOTAL'].values[-CONFIG['sequence_length']:]).unsqueeze(0).to(device)

                # Get model predictions (quantiles) - these are NORMALIZED
                with torch.no_grad():
                    quantile_predictions = model(features, hist_target)  # [1, horizon, 3]
                    quantile_predictions = quantile_predictions.cpu().numpy()[0]  # [horizon, 3]

                # CRITICAL: Denormalize predictions before passing to MPC
                # Model outputs are normalized (mean=0, std=1), need to convert back to MVA
                quantile_predictions = quantile_predictions * target_std + target_mean  # [horizon, 3]

                # Apply bias correction to compensate for model over-prediction
                # Observed bias: model systematically predicts 1.32 MVA higher than actual
                quantile_predictions = quantile_predictions - CONFIG['forecast_bias_correction']

                # Extract first-step forecast quantiles for plotting
                forecast_q10_1step = quantile_predictions[0, 0]  # First timestep, q10
                forecast_q50_1step = quantile_predictions[0, 1]  # First timestep, q50
                forecast_q90_1step = quantile_predictions[0, 2]  # First timestep, q90

                # Sample trajectories from quantile predictions (now in MVA)
                forecast_samples = sample_trajectories_from_quantiles(
                    quantile_predictions,
                    CONFIG['M']
                )

                # Plot trajectory samples for first few examples
                if trajectory_plot_counter < max_trajectory_plots:
                    try:
                        plot_trajectory_samples(
                            quantile_predictions,
                            forecast_samples,
                            outdir,
                            example_idx=trajectory_plot_counter,
                            num_samples=30
                        )
                        trajectory_plot_counter += 1
                    except Exception as e:
                        print(f"Warning: Could not plot trajectory samples: {e}")

            else:
                # Not enough history, use simple persistence
                forecast_samples = np.tile(current_load, (CONFIG['M'], CONFIG['H']))
                forecast_q10_1step = None
                forecast_q50_1step = None
                forecast_q90_1step = None
        else:
            # Use simple persistence forecast (repeat current load)
            forecast_samples = np.tile(current_load, (CONFIG['M'], CONFIG['H']))
            forecast_q10_1step = None
            forecast_q50_1step = None
            forecast_q90_1step = None

        # Run MPC
        mpc_start = time.perf_counter()
        mpc_decision = stochastic_mpc(forecast_samples, simulation_state)
        mpc_elapsed_ms = (time.perf_counter() - mpc_start) * 1000.0

        # Apply safety layer
        safe_decision, rules_triggered = safety_layer(mpc_decision, forecast_samples, thresholds, simulation_state)

        # Calculate loss
        if current_load <= safe_decision * CONFIG['Sr']:
            load_per_transformer = current_load / safe_decision
            loss_per_transformer = transformer_loss_model(load_per_transformer, CONFIG['P0'], CONFIG['Ploss_rated'], CONFIG['Sr'])
            realized_loss_kw = safe_decision * loss_per_transformer
            overload = False
        else:
            realized_loss_kw = 1000.0
            overload = True

        realized_loss_kwh = realized_loss_kw / 6.0  # 10-min to kWh

        # Capture prev_N before updating state
        prev_N = simulation_state['N']

        # Record switching
        if safe_decision != prev_N:
            switching_events.append({
                'timestamp': timestamp,
                'from_N': prev_N,
                'to_N': safe_decision,
                'mpc_decision': mpc_decision,
                'safety_override': mpc_decision != safe_decision,
                'rules_triggered': list(rules_triggered),
            })
            simulation_state['dwell_time'] = 0
            simulation_state['last_action'] = 'on' if safe_decision > simulation_state['N'] else 'off'
        else:
            simulation_state['dwell_time'] += 1

        simulation_state['N'] = safe_decision

        # Store results
        result_dict = {
            'timestamp': timestamp,
            'S_TOTAL': current_load,
            'N_current': safe_decision,
            'N_mpc': mpc_decision,
            'prev_N': prev_N,
            'loss_kw': realized_loss_kw,
            'loss_kwh': realized_loss_kwh,
            'overload': overload,
            'safety_rules': ','.join(str(r) for r in rules_triggered) if rules_triggered else '',
        }

        # Add MPC solve time
        result_dict['mpc_solve_time_ms'] = mpc_elapsed_ms

        # Add forecast quantiles if available (for plotting)
        if forecast_q10_1step is not None:
            result_dict['forecast_q10'] = forecast_q10_1step
            result_dict['forecast_q50'] = forecast_q50_1step
            result_dict['forecast_q90'] = forecast_q90_1step

        results.append(result_dict)

        # Update progress bar with current info
        simulation_pbar.set_postfix({
            'Load': f'{current_load:.1f}MVA',
            'Transformers': f'{safe_decision}',
            'Switches': f'{len(switching_events)}',
            'Loss': f'{realized_loss_kw:.0f}kW'
        })

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(outdir / 'logs.csv', index=False)

    # Compute KPIs
    total_hours = len(results_df) / 6.0
    total_loss_kwh = results_df['loss_kwh'].sum()

    # Baseline (always 3 transformers - actual operator practice)
    baseline_loss_kwh = 0
    for _, row in results_df.iterrows():
        load = row['S_TOTAL']
        if load <= 3 * CONFIG['Sr']:

            # 3 transformers can handle the load
            loss_per_transformer = transformer_loss_model(load / 3, CONFIG['P0'], CONFIG['Ploss_rated'], CONFIG['Sr'])
            baseline_loss_kwh += 3 * loss_per_transformer / 6.0
        else:
            # Overload situation - high penalty
            baseline_loss_kwh += 1000.0 / 6.0

    savings_kwh = baseline_loss_kwh - total_loss_kwh
    savings_pct = (savings_kwh / baseline_loss_kwh) * 100 if baseline_loss_kwh > 0 else 0

    # Debug information
    print(f"\nDEBUG - Energy Analysis:")
    print(f"  Total actual losses: {total_loss_kwh:.1f} kWh")
    print(f"  Baseline (Always-3) losses: {baseline_loss_kwh:.1f} kWh")
    print(f"  Energy savings: {savings_kwh:.1f} kWh")
    print(f"  Average load: {results_df['S_TOTAL'].mean():.1f} MVA")
    print(f"  Load range: {results_df['S_TOTAL'].min():.1f} - {results_df['S_TOTAL'].max():.1f} MVA")

    # Transformer utilization
    n1_time = (results_df['N_current'] == 1).sum()
    n2_time = (results_df['N_current'] == 2).sum()
    n3_time = (results_df['N_current'] == 3).sum()
    total_time = len(results_df)

    print(f"  Transformer usage: 1T={n1_time/total_time*100:.1f}%, 2T={n2_time/total_time*100:.1f}%, 3T={n3_time/total_time*100:.1f}%")

    num_switches = len(switching_events)
    switches_per_month = num_switches * (30.44 * 24 / total_hours)
    overload_count = results_df['overload'].sum()

    kpis = {
        'simulation_hours': total_hours,
        'total_loss_kwh': total_loss_kwh,
        'baseline_loss_kwh': baseline_loss_kwh,
        'savings_kwh': savings_kwh,
        'savings_percentage': savings_pct,
        'num_switches': num_switches,
        'switches_per_month': switches_per_month,
        'overload_count': overload_count,
    }

    # Generate plots
    generate_summary_plots(results_df, switching_events, kpis, outdir)

    return kpis, results_df, switching_events
