"""Shared scalar-site alignment, structure weighting, and result handling."""

from __future__ import annotations

import ast
import json
import math

import numpy as np
import torch

from fairchem.core.common import distutils


def parse_local_target(value, natoms: int) -> np.ndarray:
    """Validate a complete list of scalars without changing its site order."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("expected a list with one scalar per atom")
    array = np.asarray(value)
    if array.shape != (natoms,) or natoms == 0:
        raise ValueError(f"expected {natoms} scalars, got shape {array.shape}")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("local labels must be numeric scalars, not booleans")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("local labels must be finite numeric scalars")
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("local labels exceed float32 range")
    return array.reshape(natoms, 1)


def structure_mean(values, membership, nstructures):
    """Reduce site values using their actual atom-to-structure membership."""
    values = values.reshape(values.shape[0], -1).mean(dim=1)
    totals = values.new_zeros(nstructures).index_add(0, membership, values)
    counts = torch.bincount(membership, minlength=nstructures)
    if (counts == 0).any():
        raise ValueError("each structure must contain supervised atoms")
    return totals / counts.to(values.dtype)


def aligned_prediction(
    pred, target, membership, nstructures, loss="mae", enabled=False
):
    """Choose a single physical sign per structure using the selected loss."""
    if loss not in {"mae", "mse"}:
        raise ValueError("local scalar loss must be mae or mse")
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 1:
        raise ValueError("local predictions and labels must both have shape [N, 1]")
    signs = pred.new_ones(nstructures)
    if enabled:
        # Match exported-file alignment precision. Gradients flow through the
        # selected prediction, rather than through this discrete decision.
        with torch.no_grad():
            plus = pred.double() - target.double()
            minus = -pred.double() - target.double()
            plus = plus.abs() if loss == "mae" else plus.square()
            minus = minus.abs() if loss == "mae" else minus.square()
            flip = structure_mean(minus, membership, nstructures) < structure_mean(
                plus, membership, nstructures
            )
        signs = torch.where(flip, -signs, signs)
    return pred * signs[membership, None], signs


def local_errors(pred, target, membership, nstructures, loss):
    residual = pred - target
    error = residual.abs() if loss == "mae" else residual.square()
    return structure_mean(error, membership, nstructures)


def distributed_structure_mean(errors):
    """DDP averages gradients; compensate using the global structure count."""
    count = distutils.all_reduce(errors.numel(), device=errors.device)
    return errors.sum() * distutils.get_world_size() / count


def normalization_statistics(values, allow_global_sign_flip=False):
    """Each list has equal statistical weight, independent of its length."""
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in values]
    if not arrays or any(not a.size or not np.isfinite(a).all() for a in arrays):
        raise ValueError("normalization requires nonempty finite training labels")
    mean = 0.0 if allow_global_sign_flip else float(np.mean([a.mean() for a in arrays]))
    variance = float(np.mean([np.mean((a - mean) ** 2) for a in arrays]))
    return mean, math.sqrt(variance) if variance > 0 else 1.0


def metric_statistics(pred, target, membership, nstructures):
    """Additive statistics for structure-weighted MAE, RMSE, and weighted R²."""
    pred, target = pred.double(), target.double()
    return {
        "count": nstructures,
        "absolute": local_errors(pred, target, membership, nstructures, "mae")
        .sum()
        .item(),
        "squared": local_errors(pred, target, membership, nstructures, "mse")
        .sum()
        .item(),
        "target_sum": structure_mean(target, membership, nstructures).sum().item(),
        "target_squared_sum": structure_mean(target.square(), membership, nstructures)
        .sum()
        .item(),
    }


def statistics_metrics(stats):
    count = stats["count"]
    variance = stats["target_squared_sum"] - stats["target_sum"] ** 2 / count
    tolerance = (
        np.finfo(float).eps
        * max(abs(stats["target_squared_sum"]), np.finfo(float).tiny)
        * 8
    )
    return {
        "mae": stats["absolute"] / count,
        "rmse": math.sqrt(max(stats["squared"] / count, 0.0)),
        "r2": 1 - stats["squared"] / variance if variance > tolerance else float("nan"),
    }


def local_settings(config):
    target = config.get("outputs", {}).get("local_property", {})
    if target.get("level") != "atom":
        return {
            "property_level": "system",
            "allow_global_sign_flip": False,
            "loss": "mae",
        }
    loss = config["loss_functions"][0]["local_property"]["fn"]
    return {
        "property_level": "atom",
        "allow_global_sign_flip": target.get("allow_global_sign_flip", False),
        "loss": loss,
    }


def collect_local_results(
    lmdb_path,
    pred_path,
    target,
    material_id,
    application=False,
    allow_global_sign_flip=False,
    loss="mae",
):
    """Split flattened predictions before associating them with dataset rows."""
    import pandas as pd

    from fairchem.core.datasets import LmdbDataset

    dataset = LmdbDataset({"src": lmdb_path})
    try:
        with np.load(pred_path) as archive:
            arrays = np.split(
                archive["local_property"], archive["local_property_chunk_idx"]
            )
            ids = [int(str(sid).split("_")[-1]) for sid in archive["ids"]]
        if len(ids) != len(set(ids)) or set(ids) != set(range(len(dataset))):
            raise ValueError("prediction IDs do not match dataset rows")
        predictions = dict(zip(ids, arrays))
        rows = []
        for index in range(len(dataset)):
            data = dataset[index]
            pred = np.asarray(predictions[index]).reshape(-1)
            if len(pred) != int(data.natoms) or not np.isfinite(pred).all():
                raise ValueError(f"invalid atomic prediction for dataset row {index}")
            row = {
                "id": int(data.id),
                "material_id": data[material_id],
                target + " ML": json.dumps(pred.tolist()),
            }
            if not application:
                truth = np.asarray(data[target]).reshape(-1)
                parse_local_target(truth, len(pred))
                row[target + " DFT"] = json.dumps(truth.tolist())
                if allow_global_sign_flip:
                    aligned, signs = aligned_prediction(
                        torch.tensor(pred[:, None], dtype=torch.float64),
                        torch.tensor(truth[:, None], dtype=torch.float64),
                        torch.zeros(len(pred), dtype=torch.long),
                        1,
                        loss,
                        True,
                    )
                    row[target + " ML aligned"] = json.dumps(aligned[:, 0].tolist())
                    row["alignment_sign"] = int(signs.item())
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        del dataset


def plot_local_performance(df, target, output_path):
    """Report the same weighted metrics used by the local trainer."""
    import matplotlib.pyplot as plt

    truth = [json.loads(value) for value in df[target + " DFT"]]
    column = target + (" ML aligned" if target + " ML aligned" in df else " ML")
    pred = [json.loads(value) for value in df[column]]
    sizes = torch.tensor([len(value) for value in truth])
    membership = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    target_tensor = torch.tensor(np.concatenate(truth)[:, None], dtype=torch.float64)
    pred_tensor = torch.tensor(np.concatenate(pred)[:, None], dtype=torch.float64)
    result = statistics_metrics(
        metric_statistics(pred_tensor, target_tensor, membership, len(sizes))
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(target_tensor.numpy(), pred_tensor.numpy(), alpha=0.5, s=12)
    low = min(target_tensor.min().item(), pred_tensor.min().item())
    high = max(target_tensor.max().item(), pred_tensor.max().item())
    ax.plot([low, high], [low, high], "r--")
    ax.set(
        xlabel=target + " (True Value)",
        ylabel=target + " (ML Prediction)",
        title="Local property prediction",
    )
    label = "Sign-aligned" if "alignment_sign" in df else "Signed"
    ax.text(
        0.05,
        0.95,
        f"{label}, equal structure weights\nMAE = {result['mae']:.5g}\nRMSE = {result['rmse']:.5g}\nR² = {result['r2']:.5g}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(
        f"{label} performance for {target} ({len(df)} structures, {sum(sizes).item()} sites): {result}"
    )
    return result
