#!/usr/bin/env python
"""
Standalone evaluation script for finished or partially-finished training jobs.

Takes a test LMDB or source CSV and an ocp_predictions.npz file, then produces a CSV of
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

    # Match a larger source CSV by material identifier (requires material_ids in NPZ)
    python evaluate_result.py \
        --csv_path database.csv --pred_path ocp_predictions.npz \
        --target_property "Tc (K)" --material_id UUID

Atomic CSV targets must preserve the atom order used during prediction.
Older NPZ files remain usable with --lmdb_path. Regenerate predictions with the
updated MLIP_FTL exporter to use --csv_path; retraining is not required.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error

from fairchem.core.datasets import LmdbDataset
from fairchem.core.modules.local_property import (
    aligned_prediction,
    collect_local_results,
    parse_local_target,
    plot_local_performance,
)


def collect_csv_results(
    csv_path,
    pred_path,
    target_property,
    material_id,
    property_level,
    allow_global_sign_flip,
    loss,
):
    """Match reference rows by material ID, keeping prediction archive order."""
    reference = pd.read_csv(csv_path, dtype={material_id: str}, keep_default_na=False)
    missing_columns = {material_id, target_property} - set(reference.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    with np.load(pred_path) as archive:
        if "material_ids" not in archive:
            raise ValueError(
                "Prediction archive has no material_ids. Regenerate predictions with "
                "the updated MLIP_FTL exporter and --material_id (no retraining is "
                "required), or evaluate this archive using --lmdb_path."
            )
        if "ids" not in archive:
            raise ValueError("Prediction archive is missing ids")
        ids = archive["ids"]
        materials = archive["material_ids"]
        if ids.ndim != 1 or materials.ndim != 1 or len(ids) != len(materials):
            raise ValueError(
                "ids and material_ids must be one-dimensional with equal lengths"
            )
        if not len(ids):
            raise ValueError("Prediction archive is empty")
        if materials.dtype.kind not in "US":
            raise ValueError("material_ids must contain strings")
        materials = materials.astype(str)
        blank = [str(ids[i]) for i, value in enumerate(materials) if not value.strip()]
        if blank:
            raise ValueError(f"Blank material identifiers for prediction IDs: {blank}")
        duplicated = pd.Series(materials).duplicated(keep=False)
        if duplicated.any():
            raise ValueError(
                f"Duplicate predicted material identifiers: {sorted(set(materials[duplicated]))}"
            )

        matched = reference[reference[material_id].isin(materials)]
        duplicated = matched[material_id].duplicated(keep=False)
        if duplicated.any():
            raise ValueError(
                f"Duplicate CSV matches for material identifiers: {sorted(set(matched.loc[duplicated, material_id]))}"
            )
        missing = sorted(set(materials) - set(matched[material_id]))
        if missing:
            raise ValueError(f"No CSV match for material identifiers: {missing}")
        matched = matched.set_index(material_id).loc[materials]

        if property_level == "atom":
            required = {"local_property", "local_property_chunk_idx"}
            if not required.issubset(archive.files):
                raise ValueError(
                    f"Prediction archive is missing atomic fields: {sorted(required - set(archive.files))}"
                )
            values = archive["local_property"]
            chunks = archive["local_property_chunk_idx"]
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
                raise ValueError("local_property must contain one scalar per atom")
            if (
                chunks.ndim != 1
                or len(chunks) != len(ids) - 1
                or chunks.dtype.kind not in "iu"
                or np.any(chunks <= 0)
                or np.any(chunks >= len(values))
                or np.any(np.diff(chunks.astype(np.int64)) <= 0)
            ):
                raise ValueError(
                    "Invalid local_property_chunk_idx boundaries or structure count"
                )
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError("Atomic predictions must be finite numeric scalars")
            predictions = np.split(values.reshape(-1), chunks)
        else:
            if "energy" not in archive:
                raise ValueError("Prediction archive is missing energy")
            values = archive["energy"]
            if values.shape not in ((len(ids),), (len(ids), 1)):
                raise ValueError("energy must contain one scalar per prediction ID")
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError("System predictions must be finite numeric scalars")
            predictions = values.reshape(-1)

    rows = []
    for prediction_id, material, pred, reference_value in zip(
        ids, materials, predictions, matched[target_property]
    ):
        row = {"id": prediction_id, "material_id": material}
        try:
            if property_level == "atom":
                # Preparation stores float32 local labels; use the same values
                # for CSV and LMDB evaluation, including sign selection.
                truth = parse_local_target(reference_value, len(pred)).reshape(-1)
                row[target_property + " ML"] = json.dumps(pred.tolist())
                row[target_property + " DFT"] = json.dumps(truth.tolist())
                if allow_global_sign_flip:
                    aligned, signs = aligned_prediction(
                        torch.tensor(pred[:, None], dtype=torch.float64),
                        torch.tensor(truth[:, None], dtype=torch.float64),
                        torch.zeros(len(pred), dtype=torch.long),
                        1,
                        loss,
                        True,
                    )
                    row[target_property + " ML aligned"] = json.dumps(
                        aligned[:, 0].tolist()
                    )
                    row["alignment_sign"] = int(signs.item())
            else:
                truth = float(reference_value)
                if not np.isfinite(truth):
                    raise ValueError("reference must be a finite numeric scalar")
                row[target_property + " ML"] = float(pred)
                row[target_property + " DFT"] = truth
        except (ValueError, TypeError, SyntaxError, OverflowError) as error:
            raise ValueError(
                f"Invalid target for material {material!r}: {error}"
            ) from error
        rows.append(row)
    return pd.DataFrame(rows)


def collect_result(lmdb_path, pred_path, target_property, material_id,
                   property_level='system', allow_global_sign_flip=False, loss='mae',
                   csv_path=None):
    """
    Merge ground-truth values from an LMDB or CSV with ML predictions from
    ocp_predictions.npz.

    Args:
        lmdb_path (str | None): Path to test.lmdb; None when using csv_path
        pred_path (str): Path to ocp_predictions.npz
        target_property (str): Property key stored in the LMDB data objects
        material_id (str): Attribute name used as the material identifier
        csv_path (str | None): Source CSV, mutually exclusive with lmdb_path

    Returns:
        pd.DataFrame: Columns_id, material_id, <target> ML, <target> DFT
    """
    if (lmdb_path is None) == (csv_path is None):
        raise ValueError("Provide exactly one of lmdb_path or csv_path")
    if property_level not in ('system', 'atom'):
        raise ValueError("property_level must be system or atom")
    if allow_global_sign_flip and property_level != 'atom':
        raise ValueError("Global sign alignment requires property_level atom")
    if loss not in ('mae', 'mse'):
        raise ValueError("loss must be mae or mse")
    if csv_path is not None:
        return collect_csv_results(csv_path, pred_path, target_property, material_id,
                                   property_level, allow_global_sign_flip, loss)
    if property_level == 'atom':
        return collect_local_results(lmdb_path, pred_path, target_property, material_id,
                                     allow_global_sign_flip=allow_global_sign_flip, loss=loss)
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
    if len(df) and isinstance(df[dft_col].iloc[0], str) and df[dft_col].iloc[0].startswith('['):
        return plot_local_performance(df, target_property, output_path + '.png')
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
        description="Evaluate ML predictions against ground truth from an LMDB or CSV"
    )
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        '--lmdb_path',
        type=str,
        help='Path to test.lmdb'
    )
    reference.add_argument('--csv_path', type=str,
                           help='Source CSV; matches predictions using material identifiers')
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
        help='Target property column or LMDB key (e.g. "Tc (K)(KKR-FULL)")'
    )
    parser.add_argument(
        '--material_id',
        type=str,
        required=True,
        help='CSV column or LMDB attribute used as identifier (e.g. UUID, material_id)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for output CSV and PNG (default: same directory as pred_path)'
    )
    parser.add_argument('--property_level', choices=['system', 'atom'], default='system')
    parser.add_argument('--allow_global_sign_flip', action='store_true')
    parser.add_argument('--loss', choices=['mae', 'mse'], default='mae',
                        help='Use the same loss as training for global sign alignment')
    args = parser.parse_args()
    if args.allow_global_sign_flip and args.property_level != 'atom':
        parser.error('--allow_global_sign_flip requires --property_level atom')
    return args


def main():
    args = parse_args()

    # Validate inputs
    reference_path = args.lmdb_path if args.lmdb_path is not None else args.csv_path
    for path, label in [(reference_path, 'reference'), (args.pred_path, 'pred_path')]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            return

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.pred_path))

    tag = args.target_property.replace(' ', '_').replace('/', '_')

    print(f"Loading predictions from : {args.pred_path}")
    print(f"Loading ground truth from: {reference_path}")

    df = collect_result(
        lmdb_path=args.lmdb_path,
        pred_path=args.pred_path,
        target_property=args.target_property,
        material_id=args.material_id,
        property_level=args.property_level,
        allow_global_sign_flip=args.allow_global_sign_flip,
        loss=args.loss,
        csv_path=args.csv_path,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"performance_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to          : {csv_path}")

    plot_base = os.path.join(args.output_dir, f"performance_{tag}")
    plot_performance(df, args.target_property, plot_base)


if __name__ == "__main__":
    main()
