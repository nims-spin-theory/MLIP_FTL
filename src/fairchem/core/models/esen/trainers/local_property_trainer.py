"""Structure-balanced training for a single scalar atomic property."""

from __future__ import annotations

import math

import torch

from fairchem.core.common import distutils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, match_state_dict
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.modules.local_property import (
    aligned_prediction,
    distributed_structure_mean,
    local_errors,
    metric_statistics,
    statistics_metrics,
    structure_mean,
)
from fairchem.core.modules.normalization.normalizer import create_normalizer

from .trainer import MLIPTrainer


@registry.register_trainer("mlip_local_property_trainer")
class LocalPropertyTrainer(MLIPTrainer):
    """Opt-in local semantics; the ordinary MLIP trainer remains unchanged."""

    def __init__(self, *args, **kwargs):
        self._local_inference_only = kwargs.get("inference_only", False)
        super().__init__(*args, **kwargs)
        # BaseTrainer stores the task name ("ocp") here. Checkpoint-only
        # application must reconstruct this trainer's local semantics instead.
        self.config["trainer"] = "mlip_local_property_trainer"

    def load_task(self):
        super().load_task()
        if set(self.output_targets) != {"local_property"}:
            raise ValueError("local trainer requires exactly one local_property target")
        output = self.output_targets["local_property"]
        if output.get("level") != "atom" or output.get("shape", 1) != 1:
            raise ValueError("local_property must be one scalar per atom")
        if output.get("train_on_free_atoms", False) or output.get(
            "eval_on_free_atoms", False
        ):
            raise ValueError("local scalar targets must include all atoms")
        self.allow_global_sign_flip = output.get("allow_global_sign_flip", False)

    def load_loss(self):
        super().load_loss()
        entries = self.config["loss_functions"]
        if len(entries) != 1 or set(entries[0]) != {"local_property"}:
            raise ValueError("local trainer requires one scalar loss")
        info = entries[0]["local_property"]
        self.local_loss = info["fn"]
        if self.local_loss not in {"mae", "mse"}:
            raise ValueError("local scalar loss must be mae or mse")
        if info.get("reduction", "mean") != "mean":
            raise ValueError("local scalar loss requires a mean over structures")
        self.local_coefficient = info["coefficient"]

    def reset_transfer_ema(self):
        decay = self.config["optim"].get("ema_decay")
        self.ema = (
            ExponentialMovingAverage(self.model.parameters(), decay) if decay else None
        )
        self.config["optim"]["local_property_transfer"] = False

    def validate(self, split="val", disable_tqdm=False):
        # DistributedSampler can pad validation with repeated structures.
        # Keep only compact per-structure statistics and deduplicate at gather.
        self._validation_records = {}
        try:
            return super().validate(split, disable_tqdm)
        finally:
            self._validation_records = None

    def load_checkpoint(self, checkpoint_path, checkpoint=None, inference_only=False):
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.config["optim"].get("local_property_transfer", False) and not (
            inference_only or getattr(self, "_local_inference_only", False)
        ):
            # A base checkpoint supplies representations, never a new target's
            # statistics, head, EMA, optimizer, or epoch counters.
            source = {
                k: v for k, v in checkpoint["state_dict"].items() if "backbone." in k
            }
            if not source:
                raise ValueError("base checkpoint contains no compatible backbone")
            weights = match_state_dict(self.model.state_dict(), source)
            load_state_dict(self.model, weights, strict=True)
            self.reset_transfer_ema()
            return
        super().load_checkpoint(checkpoint_path, checkpoint, inference_only)
        if checkpoint.get("elementrefs", {}).get("local_property") is not None:
            raise ValueError(
                "local scalar properties do not support element references"
            )
        if "local_property" in checkpoint.get("normalizers", {}):
            self.normalizers["local_property"] = create_normalizer(
                state_dict=checkpoint["normalizers"]["local_property"]
            ).to(self.device)

    def _physical(self, out, batch):
        if self.elementrefs:
            raise ValueError(
                "local scalar properties do not support element references"
            )
        target = batch.local_property.view(-1, 1)
        pred = self._denorm_preds(
            "local_property", out["local_property"].float(), batch
        )
        if not torch.isfinite(target).all() or not torch.isfinite(pred).all():
            raise ValueError("local predictions and labels must be finite")
        membership = batch.batch
        count = batch.natoms.numel()
        aligned, _ = aligned_prediction(
            pred,
            target,
            membership,
            count,
            self.local_loss,
            self.allow_global_sign_flip,
        )
        return aligned, target, membership, count

    def _compute_loss(self, out, batch):
        pred, target, membership, count = self._physical(out, batch)
        if "local_property" in self.normalizers:
            normalizer = self.normalizers["local_property"]
            pred, target = normalizer.norm(pred), normalizer.norm(target)
        errors = local_errors(pred, target, membership, count, self.local_loss)
        return self.local_coefficient * distributed_structure_mean(errors)

    @torch.no_grad()
    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        metrics = {} if metrics is None else metrics
        pred, target, membership, count = self._physical(out, batch)
        stats = metric_statistics(pred, target, membership, count)
        records = getattr(self, "_validation_records", None)
        if records is not None:
            absolute = local_errors(
                pred.double(), target.double(), membership, count, "mae"
            )
            squared = local_errors(
                pred.double(), target.double(), membership, count, "mse"
            )
            target_mean = structure_mean(target.double(), membership, count)
            target_squared_mean = structure_mean(
                target.double().square(), membership, count
            )
            normalized_pred, normalized_target = pred, target
            if "local_property" in self.normalizers:
                normalizer = self.normalizers["local_property"]
                normalized_pred, normalized_target = (
                    normalizer.norm(pred),
                    normalizer.norm(target),
                )
            objective = self.local_coefficient * local_errors(
                normalized_pred, normalized_target, membership, count, self.local_loss
            )
            for i in range(count):
                if "fid" in batch:
                    identity = (str(batch.sid[i].item()), str(batch.fid[i].item()))
                else:
                    identity = str(batch.id[i])
                records[identity] = {
                    "count": 1,
                    "absolute": absolute[i].item(),
                    "squared": squared[i].item(),
                    "target_sum": target_mean[i].item(),
                    "target_squared_sum": target_squared_mean[i].item(),
                    "objective": objective[i].item(),
                }
        for name, field in (
            ("mae", "absolute"),
            ("rmse", "squared"),
            ("r2", "squared"),
        ):
            key = "local_property_" + name
            state = metrics.setdefault(key, {"total": 0.0, "numel": 0})
            state["total"] += stats[field]
            state["numel"] += count
            if name == "r2":
                for item in ("target_sum", "target_squared_sum"):
                    state[item] = state.get(item, 0.0) + stats[item]
            self._finalize_metric(name, state)
        return metrics

    @staticmethod
    def _finalize_metric(name, state):
        if name == "rmse":
            state["metric"] = math.sqrt(max(state["total"] / state["numel"], 0.0))
        elif name == "r2":
            state["metric"] = statistics_metrics(
                {
                    "count": state["numel"],
                    "absolute": 0.0,
                    "squared": state["total"],
                    "target_sum": state["target_sum"],
                    "target_squared_sum": state["target_squared_sum"],
                }
            )["r2"]
        else:
            state["metric"] = state["total"] / state["numel"]

    def _aggregate_metrics(self, metrics):
        records = getattr(self, "_validation_records", None)
        if records is not None:
            gathered = distutils.gather_objects(records)
            result = None
            if distutils.is_master():
                unique = {}
                for rank_records in gathered:
                    unique.update(rank_records)
                stats = {
                    key: sum(row[key] for row in unique.values())
                    for key in (
                        "count",
                        "absolute",
                        "squared",
                        "target_sum",
                        "target_squared_sum",
                        "objective",
                    )
                }
                values = statistics_metrics(stats)
                result = {
                    "local_property_" + key: {
                        "metric": value,
                        "total": stats["absolute"]
                        if key == "mae"
                        else stats["squared"],
                        "numel": stats["count"],
                    }
                    for key, value in values.items()
                }
                for key in ("target_sum", "target_squared_sum"):
                    result["local_property_r2"][key] = stats[key]
                result["loss"] = {
                    "metric": stats["objective"] / stats["count"],
                    "total": stats["objective"],
                    "numel": stats["count"],
                }
            payload = [result]
            distutils.broadcast_object_list(payload, src=0, device=self.device)
            return payload[0]
        result = {}
        for key, state in metrics.items():
            reduced = {
                name: distutils.all_reduce(value, device=self.device)
                for name, value in state.items()
                if name != "metric"
            }
            self._finalize_metric(key.removeprefix("local_property_"), reduced)
            result[key] = reduced
        return result
