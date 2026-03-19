#!/usr/bin/env python
"""
Standalone evaluation script for finished or partially-finished training jobs.

Takes a test LMDB and an ocp_predictions.npz file, then produces a CSV of
results and a scatter-plot performance figure.

Usage:
    python evaluate_result.py \
        --lmdb_path  set_Tc_(K)(KKR-FULL)_train/test.lmdb \
        --pred_path  result_Tc_K_KKR-FULL/checkpoints/<timestamp>/results/ocp_predictions.npz \
        --target_property "Tc (K)(KKR-FULL)" \
        --material_id UUID

    # Write outputs to a specific directory
    python evaluate_result.py \
        --lmdb_path  set_Tc/test.lmdb \
        --pred_path  result_Tc/checkpoints/2024-.../results/ocp_predictions.npz \
        --target_property "Tc (K)" \
        --material_id UUID \
        --output_dir ./eval_results
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from fairchem.core.datasets import LmdbDataset


def collect_result(lmdb_path, pred_path, target_property, material_id):
    """
    Merge ground-truth values from test.lmdb with ML predictions from
    ocp_predictions.npz.

    Args:
        lmdb_path (str): Path to test.lmdb
        pred_path (str): Path to ocp_predictions.npz
        target_property (str): Property key stored in the LMDB data objects
        material_id (str): Attribute name used as the material identifier

    Returns:
        pd.DataFrame: Columns_id, material_id, <target> ML, <target> DFT
    """
    dataset = LmdbDataset({"src": lmdb_path})

    prd_raw = np.load(pred_path)

    # Re-order predictions to match the original dataset order
    ids = [int(i.split('_')[1]) for i in prd_raw['ids']]
    inverse_ids = np.argsort(ids)
    prd = np.array([i[0] for i in prd_raw['energy']])[inverse_ids]

    dft = np.array([data[target_property] for data in dataset])

    data_list = []
    for ind, data in enumerate(dataset):
        data_list.append({
            "id": data.id,
            "material_id": data[material_id],
            target_property + " ML": prd[ind],
            target_property + " DFT": dft[ind],
        })

    return pd.DataFrame(data_list)


def plot_performance(df, target_property, output_path):
    """
    Create and save a DFT-vs-ML scatter plot with R2, MAE, and RMSE annotations.

    Args:
        df (pd.DataFrame): Output of collect_result()
        target_property (str): Property name (used for axis labels)
        output_path (str): File path for the saved PNG (without extension)
    """
    dft_col = target_property + " DFT"
    ml_col  = target_property + " ML"

    r2   = r2_score(df[dft_col], df[ml_col])
    mae  = mean_absolute_error(df[dft_col], df[ml_col])
    rmse = np.sqrt(np.mean((df[dft_col] - df[ml_col]) ** 2))

    plt.figure(figsize=(8, 8))
    plt.scatter(df[dft_col], df[ml_col], alpha=0.7, edgecolor='k', s=50)

    lims = [
        min(df[dft_col].min(), df[ml_col].min()),
        max(df[dft_col].max(), df[ml_col].max()),
    ]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction (y = x)')

    plt.xlabel(target_property + " (True Value)", fontsize=14)
    plt.ylabel(target_property + " (ML Prediction)", fontsize=14)
    plt.title("DFT vs ML " + target_property + " Prediction (Test Set)", fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')

    metrics_text = '$R^2$ = ' + str(round(r2, 3)) + '\nMAE = ' + str(round(mae, 3)) + '\nN = ' + str(len(df))
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = output_path + ".png"
    plt.savefig(png_path, bbox_inches='tight')
    print(f"Performance plot saved to: {png_path}")

    print(f"\nPerformance Summary for {target_property}:")
    print(f"{'Metric':<20}{'Value':<10}")
    print("-" * 30)
    print(f"{'R² Score':<20}{r2:.4f}")
    print(f"{'MAE':<20}{mae:.4f}")
    print(f"{'RMSE':<20}{rmse:.4f}")
    print(f"{'Data Points':<20}{len(df)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ML predictions against ground truth from test.lmdb"
    )
    parser.add_argument(
        '--lmdb_path',
        type=str,
        required=True,
        help='Path to test.lmdb'
    )
    parser.add_argument(
        '--pred_path',
        type=str,
        required=True,
        help='Path to ocp_predictions.npz'
    )
    parser.add_argument(
        '--target_property',
        type=str,
        required=True,
        help='Target property key stored in the LMDB (e.g. "Tc (K)(KKR-FULL)")'
    )
    parser.add_argument(
        '--material_id',
        type=str,
        required=True,
        help='Attribute name used as identifier (e.g. UUID, material_id)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for output CSV and PNG (default: same directory as pred_path)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    for path, label in [(args.lmdb_path, 'lmdb_path'), (args.pred_path, 'pred_path')]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            return

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.pred_path))
    os.makedirs(args.output_dir, exist_ok=True)

    tag = args.target_property.replace(' ', '_').replace('/', '_')

    print(f"Loading predictions from : {args.pred_path}")
    print(f"Loading ground truth from: {args.lmdb_path}")

    df = collect_result(
        lmdb_path=args.lmdb_path,
        pred_path=args.pred_path,
        target_property=args.target_property,
        material_id=args.material_id,
    )

    csv_path = os.path.join(args.output_dir, f"performance_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to          : {csv_path}")

    plot_base = os.path.join(args.output_dir, f"performance_{tag}")
    plot_performance(df, args.target_property, plot_base)


if __name__ == "__main__":
    main()
