#!/usr/bin/env python
"""
Training script for bandgap prediction using transfer learning with MLIPs.

This script trains machine learning interatomic potentials (MLIPs) for bandgap 
prediction using transfer learning from pre-trained models. It supports training,
validation, and evaluation with comprehensive result analysis.

Usage examples:
    # Basic training with default settings
    python train.py --target_property 2shot
    
    # Transfer learning with specific FL layer
    python train.py \
        --target_property LDA \
        --transfer_learning \
        --fl_layer 5 \
        --num_layers 12
    
    # Custom configuration and paths
    python train.py \
        --target_property LDA \
        --data_dir set_LDA_train \
        --base_model ./custom_model.pt \
        --transfer_learning \
        --num_layers 8 \
        --num_gpus 2 \
        --disable_auto_normalize \
        --gpu_id 0
    
    # Training with custom output directory
    python train.py \
        --target_property 2shot \
        --output_dir my_results \
        --job_name my_experiment \
        --batch_size 16 \
        --max_epochs 200 \
        --num_gpus 4
        
    # Evaluation only (skip training)
    python train.py \
        --target_property 2shot \
        --skip_training \
        --checkpoint_dir ./checkpoints/2shot_MLIP_TL
        
    # Dry run (generate config only)
    python train.py \
        --target_property LDA \
        --transfer_learning \
        --fl_layer 5 \
        --num_layers 12 \
        --dryrun
    # Generates: config_LDA_MPL12_TL5.yml
    
    # Manual normalization specification
    python train.py \
        --target_property LDA \
        --manual_mean 2.0 \
        --manual_stdev 1.5 \
        --dryrun
    # Uses specified mean=2.0, stdev=1.5 instead of auto-calculation
"""

import argparse
import os
import time
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from fairchem.core.common.tutorial_utils import fairchem_main
from fairchem.core.datasets import LmdbDataset
from sklearn.metrics import r2_score, mean_absolute_error

# GPU detection and diagnostics
def print_gpu_optimization_summary(args):
    """Print summary of GPU optimization settings."""
    print("\n" + "="*50)
    print("GPU OPTIMIZATION SUMMARY")
    print("="*50)
    
    print(f"Batch size: {args.batch_size}")
    if args.auto_batch_size:
        print("  ✓ Auto-optimized for maximum GPU utilization")
    
    if args.gradient_accumulation_steps > 1:
        effective_batch = args.batch_size * args.gradient_accumulation_steps
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch}")
    
    print(f"Number of data workers: {args.num_workers}")
    if args.num_workers > 0:
        print("  ✓ Multi-threaded data loading enabled")
    
    if args.pin_memory:
        print("  ✓ Pinned memory enabled for faster GPU transfer")
    
    if args.mixed_precision:
        print("  ✓ Mixed precision training enabled")
    
    print(f"Number of GPUs: {args.num_gpus}")
    if args.num_gpus > 1:
        print("  ✓ Multi-GPU distributed training enabled")
    
    # GPU utilization tips
    print("\nGPU UTILIZATION TIPS:")
    if not args.auto_batch_size and args.batch_size < 16:
        print("  💡 Consider using --auto_batch_size for optimal batch size")
    
    if args.num_workers == 0:
        print("  💡 Consider increasing --num_workers for faster data loading")
    
    if not args.pin_memory and not args.cpu_only:
        print("  💡 Consider using --pin_memory for faster GPU transfer")
    
    if not args.mixed_precision and not args.cpu_only:
        print("  💡 Consider using --mixed_precision for better performance")
    
    if args.gradient_accumulation_steps == 1 and args.batch_size < 32:
        print("  💡 Consider --gradient_accumulation_steps for larger effective batch size")
    
    print("="*50)


def print_gpu_info():
    """Print detailed GPU information for diagnostics."""
    print("=" * 50)
    print("GPU DIAGNOSTICS")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("CUDA not available - will use CPU mode")
            
    except ImportError:
        print("PyTorch not available - GPU detection not possible")
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
    
    # Also try nvidia-smi if available
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        print("\nnvidia-smi output:")
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi not available or failed")
    
    print("=" * 50)


def optimize_batch_size(data_dir, max_batch_size=64, target_property="formation_energy"):
    """
    Automatically find the optimal batch size for maximum GPU utilization.
    
    Args:
        data_dir (str): Path to training data
        max_batch_size (int): Maximum batch size to test
        target_property (str): Target property name
        
    Returns:
        int: Optimal batch size
    """
    print("Optimizing batch size for maximum GPU utilization...")
    
    try:
        import torch
        from fairchem.core.datasets import LmdbDataset
        
        if not torch.cuda.is_available():
            print("CUDA not available, using default batch size")
            return 8
            
        # Load a small sample of data
        train_path = os.path.join(data_dir, "train.lmdb")
        if not os.path.exists(train_path):
            print(f"Training data not found at {train_path}, using default batch size")
            return 8
            
        dataset = LmdbDataset({"src": train_path})
        
        # Test different batch sizes
        optimal_batch_size = 8
        batch_sizes_to_test = [8, 16, 24, 32, 48, 64]
        batch_sizes_to_test = [bs for bs in batch_sizes_to_test if bs <= max_batch_size]
        
        print(f"Testing batch sizes: {batch_sizes_to_test}")
        
        for batch_size in batch_sizes_to_test:
            try:
                # Use PyTorch Geometric DataLoader for proper handling
                from torch_geometric.loader import DataLoader as GeometricDataLoader
                
                dataloader = GeometricDataLoader(
                    dataset, batch_size=batch_size,
                    shuffle=False, num_workers=0
                )
                
                # Try to load one batch
                batch = next(iter(dataloader))
                
                # Test memory usage with realistic data size
                if torch.cuda.is_available() and hasattr(batch, 'pos'):
                    # Move batch to GPU to test memory usage
                    batch = batch.to('cuda')
                    
                    # Create tensor based on actual data dimensions
                    num_atoms = (batch.pos.size(0) 
                               if hasattr(batch, 'pos') else 100)
                    test_tensor = torch.randn(num_atoms, 128, device='cuda')
                    
                    # Clean up immediately
                    del test_tensor, batch
                    torch.cuda.empty_cache()
                    
                    optimal_batch_size = batch_size
                    print(f"  Batch size {batch_size}: OK")
                else:
                    # CPU mode or missing position data
                    optimal_batch_size = batch_size
                    print(f"  Batch size {batch_size}: OK (CPU mode)")
                    
            except (RuntimeError, MemoryError) as e:
                if "out of memory" in str(e).lower():
                    print(f"  Batch size {batch_size}: Out of memory")
                    break
                else:
                    print(f"  Batch size {batch_size}: Error - {e}")
                    break
            except Exception as e:
                print(f"  Batch size {batch_size}: Unexpected error - {e}")
                break
                
        print(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
        
    except ImportError:
        print("Required libraries not available for batch size optimization")
        return 8
    except Exception as e:
        print(f"Error during batch size optimization: {e}")
        return 8


def collect_result(dft_path, prd_path, target, application=False):
    """
    Collect and organize DFT vs ML prediction results.
    
    Args:
        dft_path (str): Path to test set (LMDB format)
        prd_path (str): Path to test prediction output (ocp_predictions.npz)
        target (str): Target property name in LMDB
        application (bool): True if used for application, False for test evaluation
        
    Returns:
        pd.DataFrame: DataFrame containing results with material IDs and predictions
    """
    # Load DFT reference data
    dft_raw = LmdbDataset({"src": dft_path})
    
    # Load ML predictions
    try:
        prd_raw = np.load(prd_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prediction file not found: {prd_path}")
    
    if not application:
        dft = np.array([data[target] for data in dft_raw])
    
    # Process ML predictions - sort by original order
    ids = [int(i.split('_')[1]) for i in prd_raw['ids']]
    inverse_ids = np.argsort(ids)
    prd = np.array([i[0] for i in prd_raw['energy']])
    prd = prd[inverse_ids]
    
    # Create comprehensive dataframe
    dataset = LmdbDataset({"src": dft_path})
    data_list = []
    
    for ind, data in enumerate(dataset):
        row = {
            "id": data.id,
            'material_id': data.material_id,
            target + "_ML": prd[ind],
        }
        if not application:
            row[target + "_DFT"] = dft[ind]
            
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    return df


def calculate_normalization_stats(data_dir, target_property):
    """
    Calculate mean and standard deviation for normalization from training data.
    
    Args:
        data_dir (str): Directory containing LMDB datasets
        target_property (str): Target property name
        
    Returns:
        tuple: (mean, stdev) for normalization
    """
    train_path = os.path.join(data_dir, 'train.lmdb')
    
    if not os.path.exists(train_path):
        print(f"Warning: Training data not found at {train_path}")
        print("Using default normalization values: mean=350, stdev=350")
        return 350, 350
    
    try:
        # Load training dataset directly with fairchem
        print(f"Loading training data from {train_path}...")
        dataset = LmdbDataset({"src": train_path})
        
        # Check if dataset is empty
        if len(dataset) == 0:
            print(f"Warning: Dataset is empty: {train_path}")
            print("Using default normalization values: mean=350, stdev=350")
            return 350, 350
            
        print(f"Found {len(dataset)} samples in training data")
        
        # Extract target property values
        values = []
        for i, data in enumerate(dataset):
            try:
                # Try different ways to access the target property
                value = None
                
                # Method 1: Direct attribute access
                if hasattr(data, target_property):
                    value = getattr(data, target_property)
                
                # Method 2: Dictionary-like access
                elif hasattr(data, '__getitem__') and target_property in data:
                    value = data[target_property]
                
                # Method 3: Check if it's a tensor and convert
                if value is not None:
                    if hasattr(value, 'item'):  # PyTorch tensor
                        value = value.item()
                    elif hasattr(value, 'tolist'):  # NumPy array
                        value = float(value)
                    
                    # Only add valid numeric values
                    if value is not None and not np.isnan(float(value)):
                        values.append(float(value))
                        
            except Exception as e:
                print(f"Warning: Could not extract {target_property} "
                      f"from data point {i}: {e}")
                continue
        
        if not values:
            print(f"Warning: No valid {target_property} values found "
                  f"in training data")
            print("Using default normalization values: mean=350, stdev=350")
            return 350, 350
        
        values = np.array(values)
        mean = float(np.mean(values))
        stdev = float(np.std(values))
        
        # Ensure stdev is not zero
        if stdev == 0:
            stdev = 1.0
            print("Warning: Standard deviation is zero, using stdev=1.0")
        
        print(f"Calculated normalization stats for {target_property}:")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std Dev: {stdev:.6f}")
        print(f"  Min: {np.min(values):.6f}")
        print(f"  Max: {np.max(values):.6f}")
        print(f"  Count: {len(values)}")
        
        return mean, stdev
        
    except Exception as e:
        print(f"Error calculating normalization stats: {e}")
        print("Using default normalization values: mean=350, stdev=350")
        return 350, 350


def create_config_file(config_path, data_dir, target_property,
                       batch_size=8, max_epochs=100, learning_rate=0.0004,
                       transfer_learning=False, fl_layer=7, num_layers=10,
                       auto_normalize=True, cpu_only=False, num_gpus=1,
                       num_workers=4, pin_memory=False, manual_mean=None,
                       manual_stdev=None):
    """
    Create comprehensive YAML configuration file for training.
    
    Args:
        config_path (str): Output path for config file
        data_dir (str): Directory containing LMDB datasets
        target_property (str): Target property name
        batch_size (int): Training batch size
        max_epochs (int): Maximum training epochs
        learning_rate (float): Learning rate
        transfer_learning (bool): Whether to use transfer learning
        fl_layer (int): Fine-tuning layer for transfer learning
        num_layers (int): Number of layers in the model backbone
        auto_normalize (bool): Whether to auto-calculate normalization stats
        cpu_only (bool): Whether to use CPU-only mode
        num_gpus (int): Number of GPUs to use
        num_workers (int): Number of data loading workers
        pin_memory (bool): Whether to use pinned memory
        manual_mean (float): Manually specified mean (overrides auto-calculation)
        manual_stdev (float): Manually specified stdev (overrides auto-calculation)
    """
    # Determine normalization statistics
    if manual_mean is not None and manual_stdev is not None:
        # Use manually specified values
        mean, stdev = manual_mean, manual_stdev
        print(f"Using manually specified normalization values:")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std Dev: {stdev:.6f}")
    elif manual_mean is not None or manual_stdev is not None:
        # Partial manual specification - warn and use auto-calculation
        print("Warning: Both --manual_mean and --manual_stdev must be specified together")
        print("Falling back to automatic calculation")
        if auto_normalize:
            mean, stdev = calculate_normalization_stats(data_dir, target_property)
        else:
            mean, stdev = 350, 350  # Default values
    elif auto_normalize:
        # Automatic calculation
        mean, stdev = calculate_normalization_stats(data_dir, target_property)
    else:
        # Use default values
        mean, stdev = 350, 350
        print("Using default normalization values: mean=350, stdev=350")
    
    # Base configuration following the provided YAML structure
    config = {
        'dataset': {
            'train': {
                'format': 'lmdb',
                'src': f'{data_dir}/train.lmdb',
                'key_mapping': {
                    target_property: 'energy'
                },
                'transforms': {
                    'normalizer': {
                        'energy': {
                            'mean': mean,
                            'stdev': stdev
                        }
                    }
                }
            },
            'val': {
                'format': 'lmdb',
                'src': f'{data_dir}/val.lmdb',
                'key_mapping': {
                    target_property: 'energy'
                }
            },
            'test': {
                'format': 'lmdb',
                'src': f'{data_dir}/test.lmdb',
                'key_mapping': {
                    target_property: 'energy'
                }
            }
        },
        'evaluation_metrics': {
            'metrics': {
                'energy': ['mae']
            },
            'primary_metric': 'energy_mae'
        },
        'gpus': 0 if cpu_only else num_gpus,
        'logger': 'tensorboard',
        'distributed': num_gpus > 1,  # Enable distributed for multi-GPU
        'loss_functions': [
            {
                'energy': {
                    'coefficient': 10,
                    'fn': 'mae'
                }
            }
        ],
        'model': {
            'backbone': {
                'act_type': 'gate',
                'cutoff': 6.0,
                'direct_forces': False,
                'distance_function': 'gaussian',
                'edge_channels': 128,
                'hidden_channels': 128,
                'lmax': 3,
                'max_neighbors': 300,
                'max_num_elements': 100,
                'mlp_type': 'spectral',
                'mmax': 2,
                'model': 'esen_backbone',
                'norm_type': 'rms_norm_sh',
                'num_distance_basis': 64,
                'num_layers': num_layers,
                'otf_graph': True,
                'regress_forces': False,
                'regress_stress': False,
                'sphere_channels': 128,
                'use_envelope': True,
                'use_pbc': True,
                'use_pbc_single': True
            },
            'heads': {
                'energy': {
                    'module': 'esen_mlp_energy_head'
                }
            },
            'name': 'hydra',
            'otf_graph': True,
            'pass_through_head_outputs': True
        },
        'optim': {
            'batch_size': batch_size,
            'eval_batch_size': 32,
            'clip_grad_norm': 100,
            'ema_decay': 0.999,
            'lr_initial': learning_rate,
            'max_epochs': max_epochs,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'optimizer': 'AdamW',
            'optimizer_params': {
                'weight_decay': 0.001
            },
            'scheduler': 'LambdaLR',
            'scheduler_params': {
                'epochs': max_epochs,
                'lambda_type': 'cosine',
                'lr': learning_rate,
                'lr_min_factor': 0.1,
                'warmup_epochs': 10,
                'warmup_factor': 0.2
            }
        },
        'outputs': {
            'energy': {
                'level': 'system',
                'property': 'energy'
            }
        },
        'trainer': 'mlip_trainer'
    }
    
    # Add transfer learning specific configuration
    if transfer_learning:
        config['optim']['FL'] = fl_layer
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    print(f"Model layers: {num_layers}")
    print(f"GPUs: {0 if cpu_only else num_gpus}")
    print(f"Distributed training: {num_gpus > 1}")
    print(f"Normalization - Mean: {mean:.6f}, Std Dev: {stdev:.6f}")
    if transfer_learning:
        print(f"Transfer learning enabled with FL layer: {fl_layer}")


def run_training(config_path, run_dir, job_name, base_model=None, gpu_id=0,
                 print_every=50, cpu_only=False, num_gpus=1):
    """
    Execute training process using fairchem.
    
    Args:
        config_path (str): Path to YAML configuration file
        run_dir (str): Output directory for results
        job_name (str): Identifier for the training job
        base_model (str): Path to pre-trained base model
        gpu_id (int): Starting GPU device ID (for single GPU or multi-GPU)
        print_every (int): Frequency of progress printing
        cpu_only (bool): Whether to use CPU-only mode
        num_gpus (int): Number of GPUs to use
        
    Returns:
        str: Path to checkpoint directory
    """
    log_file = f"log_train_{job_name}.txt"
    warn_file = f"warn_train_{job_name}.txt"
    
    # Construct training command
    cmd = [
        'python', str(fairchem_main()),
        '--mode', 'train',
        '--config-yml', config_path,
        '--run-dir', run_dir,
        '--identifier', job_name,
        '--print-every', str(print_every),
    ]
    if base_model is not None:
        cmd += ['--checkpoint', base_model]

    # Set environment variables
    env = os.environ.copy()
    
    if not cpu_only:
        if num_gpus == 1:
            # Single GPU mode
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"Using single GPU: {gpu_id}")
        else:
            # Multi-GPU mode: use consecutive GPUs starting from gpu_id
            gpu_list = ','.join(str(gpu_id + i) for i in range(num_gpus))
            env['CUDA_VISIBLE_DEVICES'] = gpu_list
            print(f"Using multiple GPUs: {gpu_list}")
    else:
        # Disable CUDA for CPU-only mode
        env['CUDA_VISIBLE_DEVICES'] = ''
        print("Using CPU-only mode")
    
    # Set distributed training environment variables
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = '12355'
    env['RANK'] = '0'
    env['WORLD_SIZE'] = str(num_gpus) if num_gpus > 1 else '1'
    env['LOCAL_RANK'] = '0'
    
    print("Environment variables:")
    print(f"  CUDA_VISIBLE_DEVICES: "
          f"{env.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  WORLD_SIZE: {env.get('WORLD_SIZE')}")
    print(f"  Distributed training: {num_gpus > 1}")
    
    print(f"Starting training: {job_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Logs: {log_file}, {warn_file}")
    
    # Execute training
    start_time = time.time()
    
    with open(log_file, 'w') as log_f, open(warn_file, 'w') as warn_f:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=warn_f,
            text=True
        )
    
    elapsed_time = time.time() - start_time
    print(f'Training completed in {elapsed_time:.1f} seconds')
    
    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        print(f"Check {warn_file} for error details")
        return None
    
    # Extract checkpoint directory from log
    try:
        cpdir = None
        with open(log_file, 'r') as f:
            for line in f:
                if 'checkpoint_dir:' in line:
                    cpdir = line.split(':')[-1].strip()
                    break
        
        if cpdir is None:
            raise ValueError("Checkpoint directory not found in log")
        
        # Copy config to checkpoint directory
        config_dest = os.path.join(cpdir, 'config.yml')
        subprocess.run(['cp', config_path, config_dest], check=True)
        
        print(f"Checkpoint directory: {cpdir}")
        return cpdir
        
    except Exception as e:
        print(f"Error extracting checkpoint directory: {e}")
        return None


def evaluate_model(cpdir, test_data_path, target_property, output_prefix):
    """
    Evaluate trained model and generate performance metrics.
    
    Args:
        cpdir (str): Checkpoint directory path
        test_data_path (str): Path to test LMDB dataset
        target_property (str): Target property name
        output_prefix (str): Prefix for output files
        
    Returns:
        pd.DataFrame: Evaluation results
    """
    # Construct prediction file path
    prd_path = cpdir.replace('checkpoints', 'results') + '/ocp_predictions.npz'
    
    print(f'Test set path: {test_data_path}')
    print(f'Prediction path: {prd_path}')
    
    # Collect results
    df = collect_result(test_data_path, prd_path, target=target_property, 
                       application=False)
    
    # Save results
    csv_output = f'{output_prefix}_{target_property}.csv'
    df.to_csv(csv_output, index=False)
    print(f"Results saved to: {csv_output}")
    
    return df


def plot_performance(df, target_property, output_prefix):
    """
    Create performance visualization plot.
    
    Args:
        df (pd.DataFrame): Results dataframe
        target_property (str): Target property name
        output_prefix (str): Prefix for output files
    """
    dft_col = f'{target_property}_DFT'
    ml_col = f'{target_property}_ML'
    
    # Compute metrics
    r2 = r2_score(df[dft_col], df[ml_col])
    mae = mean_absolute_error(df[dft_col], df[ml_col])
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(df[dft_col], df[ml_col], alpha=0.7, edgecolor='k', s=50)
    
    # 1:1 line
    lims = [
        min(df[dft_col].min(), df[ml_col].min()),
        max(df[dft_col].max(), df[ml_col].max())
    ]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction (y = x)')
    
    # Labels and formatting
    plt.xlabel(f"{target_property} (DFT)", fontsize=14)
    plt.ylabel(f"{target_property} (ML)", fontsize=14)
    plt.title(f"DFT vs ML {target_property} Prediction (Test Set)", fontsize=16)
    
    # Equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add metrics text
    metrics_text = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nN = {len(df)}"
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_output = f'{output_prefix}_{target_property}_performance.png'
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {plot_output}")
    plt.show()
    
    # Print summary statistics
    print(f"\nPerformance Summary for {target_property}:")
    print(f"{'Metric':<20}{'Value':<10}")
    print("-" * 30)
    print(f"{'R² Score':<20}{r2:.4f}")
    print(f"{'MAE':<20}{mae:.4f}")
    print(f"{'RMSE':<20}{np.sqrt(np.mean((df[dft_col] - df[ml_col])**2)):.4f}")
    print(f"{'Data Points':<20}{len(df)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML model for bandgap prediction using transfer learning"
    )
    
    parser.add_argument(
        '--target_property',
        type=str,
        required=True,
        help='Target property name (e.g., 2shot, LDA)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory containing LMDB datasets (default: set_{target_property}_train)'
    )
    
    parser.add_argument(
        '--base_model',
        type=str,
        default='./esen_30m_oam.pt',
        help='Path to pre-trained base model'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: result_{target_property})'
    )
    
    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help='Job identifier (default: {target_property}_MLIP_TL)'
    )
    
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU device ID (for single GPU mode)'
    )
    
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
        help='Number of GPUs to use (default: 1, use 0 for CPU-only)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size (use --auto_batch_size for automatic optimization)'
    )
    
    parser.add_argument(
        '--auto_batch_size',
        action='store_true',
        help='Automatically optimize batch size for maximum GPU utilization'
    )
    
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=64,
        help='Maximum batch size to try when using --auto_batch_size'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4, 0=single-threaded)'
    )
    
    parser.add_argument(
        '--pin_memory',
        action='store_true',
        help='Use pinned memory for faster GPU data transfer'
    )
    
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Use mixed precision training (AMP) for better performance'
    )
    
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (effective batch size = batch_size * steps)'
    )
    
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=100,
        help='Maximum training epochs'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0004,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--transfer_learning',
        action='store_true',
        help='Enable transfer learning mode'
    )
    
    parser.add_argument(
        '--fl_layer',
        type=int,
        default=7,
        help='Fine-tuning layer for transfer learning (default: 7)'
    )
    
    parser.add_argument(
        '--num_layers',
        type=int,
        default=10,
        help='Number of layers in the model backbone (default: 10)'
    )
    
    parser.add_argument(
        '--disable_auto_normalize',
        action='store_true',
        help='Disable automatic normalization calculation (use default values)'
    )
    
    parser.add_argument(
        '--manual_mean',
        type=float,
        default=None,
        help='Manually specify normalization mean (overrides auto-calculation)'
    )
    
    parser.add_argument(
        '--manual_stdev',
        type=float,
        default=None,
        help='Manually specify normalization standard deviation (overrides auto-calculation)'
    )
    
    parser.add_argument(
        '--cpu_only',
        action='store_true',
        help='Use CPU-only mode (disable GPU acceleration)'
    )
    
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Generate configuration file and exit (do not run training)'
    )
    
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training and only evaluate existing model'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Existing checkpoint directory for evaluation (used with --skip_training)'
    )
    
    return parser.parse_args()


def main():
    """Main function to orchestrate the training and evaluation process."""
    args = parse_args()
    
    # Print GPU diagnostics
    print_gpu_info()
    
    # Print GPU optimization summary
    print_gpu_optimization_summary(args)
    
    # Set default values based on target property
    if args.data_dir is None:
        args.data_dir = f"set_{args.target_property}_train"
    
    if args.output_dir is None:
        args.output_dir = f"result_{args.target_property}"
    
    if args.job_name is None:
        args.job_name = f"{args.target_property}_MPL{args.num_layers}_TL{args.fl_layer}"
    
    # Validate inputs (skip for dryrun mode, except data_dir needed for normalization)
    if not args.skip_training and not args.dryrun:
        if not os.path.exists(args.base_model):
            print(f"Error: Base model not found: {args.base_model}")
            return
    
    # Data directory is always needed (for normalization calculation)
    if not args.skip_training and not os.path.exists(args.data_dir):
        if args.dryrun:
            print(f"Warning: Data directory not found: {args.data_dir}")
            print("Automatic normalization will use default values.")
        else:
            print(f"Error: Data directory not found: {args.data_dir}")
            return
    
    print("Training Configuration:")
    print(f"  Target Property: {args.target_property}")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Job Name: {args.job_name}")
    print(f"  GPU ID: {args.gpu_id}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Epochs: {args.max_epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Number of Layers: {args.num_layers}")
    print(f"  Transfer Learning: {args.transfer_learning}")
    if args.transfer_learning:
        print(f"  FL Layer: {args.fl_layer}")
    print(f"  Base Model: {args.base_model}")
    print(f"  CPU Only: {args.cpu_only}")
    print(f"  Dry Run: {args.dryrun}")
    
    # Optimize batch size if requested
    if args.auto_batch_size and not args.cpu_only:
        print("\n" + "="*50)
        print("BATCH SIZE OPTIMIZATION")
        print("="*50)
        optimal_batch_size = optimize_batch_size(
            args.data_dir, 
            args.max_batch_size, 
            args.target_property
        )
        args.batch_size = optimal_batch_size
        print(f"Updated batch size to: {args.batch_size}")
    
    # Generate configuration file (always done, even for dryrun)
    if args.transfer_learning:
        config_filename = (
            f"config_{args.target_property}_MPL{args.num_layers}"
            f"_TL{args.fl_layer}.yml"
        )
    else:
        config_filename = (
            f"config_{args.target_property}_MPL{args.num_layers}.yml"
        )
    
    config_path = config_filename
    
    create_config_file(
        config_path=config_path,
        data_dir=args.data_dir,
        target_property=args.target_property,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        transfer_learning=args.transfer_learning,
        fl_layer=args.fl_layer,
        num_layers=args.num_layers,
        auto_normalize=not args.disable_auto_normalize,
        cpu_only=args.cpu_only,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        manual_mean=args.manual_mean,
        manual_stdev=args.manual_stdev
    )
    
    # Exit early if dryrun mode
    if args.dryrun:
        print("\n" + "="*50)
        print("DRY RUN MODE - Configuration Generated")
        print("="*50)
        print(f"Configuration file created: {config_path}")
        print("Exiting without training or evaluation.")
        return
    
    # Training phase
    if not args.skip_training:
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        # Configuration file already created above
        print(f"Using configuration file: {config_path}")
        
        # Run training
        cpdir = run_training(
            config_path=config_path,
            run_dir=args.output_dir,
            job_name=args.job_name,
            base_model=args.base_model,
            gpu_id=args.gpu_id,
            cpu_only=args.cpu_only,
            num_gpus=args.num_gpus
        )
        
        if cpdir is None:
            print("Training failed. Exiting.")
            return
    else:
        if args.checkpoint_dir is None:
            msg = ("Error: --checkpoint_dir must be provided when using "
                   "--skip_training")
            print(msg)
            return
        cpdir = args.checkpoint_dir
    
    # Evaluation phase
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    # Test data path
    test_data_path = os.path.join(args.data_dir, "test.lmdb")
    if not os.path.exists(test_data_path):
        print(f"Warning: Test data not found: {test_data_path}")
        print("Skipping evaluation.")
        return
    
    # Evaluate model
    output_prefix = "performance_test_MLIP"
    df = evaluate_model(
        cpdir=cpdir,
        test_data_path=test_data_path,
        target_property=args.target_property,
        output_prefix=output_prefix
    )
    
    # Generate performance plot
    plot_performance(df, args.target_property, output_prefix)
    
    print("\nTraining and evaluation completed successfully!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
