"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import re

import torch

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import get_checkpoint_format, load_state_dict
from fairchem.core.trainers import OCPTrainer


class BaseTask:
    def __init__(self, config) -> None:
        self.config = config

    def _log_parameter_flexibility_summary(self) -> None:
        """Log total/trainable/frozen parameter counts."""
        flexible_params = 0
        unflexible_params = 0

        for _, param in self.trainer.model.named_parameters():
            param_count = param.numel()
            if param.requires_grad:
                flexible_params += param_count
            else:
                unflexible_params += param_count

        total_params = flexible_params + unflexible_params
        logging.info("Parameter flexibility summary:")
        logging.info("  total params      : %s", f"{total_params:,}")
        logging.info("  flexible params   : %s", f"{flexible_params:,}")
        logging.info("  unflexible params : %s", f"{unflexible_params:,}")

    @staticmethod
    def _match_module_prefix(model_state_dict, checkpoint_state_dict):
        """Align module prefixes between model and checkpoint state dicts."""
        ckpt_key_count = next(iter(checkpoint_state_dict)).count("module")
        mod_key_count = next(iter(model_state_dict)).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            return {
                key_count_diff * "module." + k: v
                for k, v in checkpoint_state_dict.items()
            }
        if key_count_diff < 0:
            return {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint_state_dict.items()
            }
        return checkpoint_state_dict

    @staticmethod
    def _backbone_block_idx(param_name: str):
        match = re.search(r"backbone\.blocks\.(\d+)", param_name)
        if match is None:
            return None
        return int(match.group(1))

    def _count_copied_parameters_from_checkpoint(
        self, checkpoint_path: str, tl_mode: str | None
    ) -> tuple[int, int]:
        """Count tensors/parameters that can be copied from checkpoint."""
        map_location = torch.device("cpu") if self.trainer.cpu else self.trainer.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        checkpoint_state = checkpoint["state_dict"]

        model_state = self.trainer.model.state_dict()
        checkpoint_state = self._match_module_prefix(model_state, checkpoint_state)

        transfer_layers = self.config["optim"].get("TL_transfer_layers", None)
        copied_tensors = 0
        copied_params = 0

        for name, tensor in checkpoint_state.items():
            if name not in model_state:
                continue
            if model_state[name].shape != tensor.shape:
                continue

            if tl_mode == "partial":
                block_idx = self._backbone_block_idx(name)
                is_embedding = "embedding" in name
                is_transferred_block = (
                    block_idx is not None
                    and transfer_layers is not None
                    and block_idx < transfer_layers
                )
                if not (is_embedding or is_transferred_block):
                    continue

            copied_tensors += 1
            copied_params += tensor.numel()

        return copied_tensors, copied_params

    def _load_partial_transfer_checkpoint(self, checkpoint_path: str) -> None:
        transfer_layers = self.config["optim"].get("TL_transfer_layers", None)
        if transfer_layers is None:
            raise ValueError(
                "Partial transfer mode requires optim.TL_transfer_layers to be set."
            )

        if transfer_layers <= 0:
            raise ValueError(
                f"optim.TL_transfer_layers must be > 0, got {transfer_layers}."
            )

        map_location = torch.device("cpu") if self.trainer.cpu else self.trainer.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        checkpoint_state = checkpoint["state_dict"]

        model_state = self.trainer.model.state_dict()
        checkpoint_state = self._match_module_prefix(model_state, checkpoint_state)

        transferred = {}
        transferred_names = []
        for name, tensor in checkpoint_state.items():
            if name not in model_state:
                continue
            if model_state[name].shape != tensor.shape:
                continue

            block_idx = self._backbone_block_idx(name)
            is_embedding = "embedding" in name
            is_transferred_block = block_idx is not None and block_idx < transfer_layers
            if not (is_embedding or is_transferred_block):
                continue

            transferred[name] = tensor
            transferred_names.append(name)

        model_state.update(transferred)
        # We load a full state dict here after updating selected params to avoid
        # unexpected missing-key noise while still only transferring chosen keys.
        load_state_dict(self.trainer.model, model_state, strict=True)

        logging.info(
            "Partial TL loaded %s tensors (transfer layers=%s) from %s",
            len(transferred),
            transfer_layers,
            checkpoint_path,
        )
        copied_params = sum(tensor.numel() for tensor in transferred.values())
        logging.info("Partial TL copied parameters: %s", f"{copied_params:,}")
        if transferred_names:
            logging.info("Partial TL keys: %s",) 
            for name in transferred_names:
                logging.info("  %s", name)

    def _freeze_layers(self) -> None:
        frozen_layers = self.config["optim"].get("FL", None)
        if frozen_layers is None:
            self._log_parameter_flexibility_summary()
            return

        tl_mode = self.config["optim"].get("TL_mode", None)
        transfer_layers = self.config["optim"].get("TL_transfer_layers", None)
        if tl_mode == "partial" and transfer_layers is not None and frozen_layers > transfer_layers:
            raise ValueError(
                "In partial TL mode, optim.FL (frozen layers) must be less than or equal to "
                f"optim.TL_transfer_layers. Got FL={frozen_layers}, "
                f"TL_transfer_layers={transfer_layers}."
            )

        logging.info("Frozen layers: %s", frozen_layers)
        for name, param in self.trainer.model.named_parameters():
            if "embedding" in name:
                param.requires_grad = False

        for frozen_idx in range(frozen_layers):
            prefix = f"backbone.blocks.{frozen_idx}."
            for name, param in self.trainer.model.named_parameters():
                if prefix in name:
                    param.requires_grad = False

        logging.info("List of layers:")
        for name, param in self.trainer.model.named_parameters():
            logging.info("%s: requires_grad=%s", name, param.requires_grad)
        self._log_parameter_flexibility_summary()

    def setup(self, trainer) -> None:
        self.trainer = trainer

        format = get_checkpoint_format(self.config)
        if format == "pt":
            self.chkpt_path = os.path.join(
                self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
            )
        else:
            self.chkpt_path = self.trainer.config["cmd"]["checkpoint_dir"]

        # if the supplied checkpoint exists, then load that, ie: when user specifies the --checkpoint option
        # OR if the a job was preempted correctly and the submitit checkpoint function was called
        # (https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py#L44), then we should attempt to
        # load that checkpoint
        if self.config["checkpoint"] is not None:
            logging.info(
                f"Attemping to load user specified checkpoint at {self.config['checkpoint']}"
            )
            tl_mode = self.config["optim"].get("TL_mode", None)
            copied_tensors, copied_params = self._count_copied_parameters_from_checkpoint(
                self.config["checkpoint"], tl_mode
            )
            logging.info(
                "Copied from checkpoint: tensors=%s, parameters=%s",
                copied_tensors,
                f"{copied_params:,}",
            )
            if tl_mode == "partial":
                self._load_partial_transfer_checkpoint(self.config["checkpoint"])
            else:
                self.trainer.load_checkpoint(checkpoint_path=self.config["checkpoint"])
        # if the supplied checkpoint doesn't exist and there exists a previous checkpoint in the checkpoint path, this
        # means that the previous job didn't terminate "nicely" (due to node failures, crashes etc), then attempt
        # to load the last found checkpoint
        elif (
            os.path.isfile(self.chkpt_path)
            or (os.path.isdir(self.chkpt_path) and len(os.listdir(self.chkpt_path))) > 0
        ):
            logging.info(
                f"Previous checkpoint found at {self.chkpt_path}, resuming job from this checkecpoint"
            )
            self.trainer.load_checkpoint(checkpoint_path=self.chkpt_path)
        self._freeze_layers()


    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError) -> None:
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self) -> None:
        try:
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False)
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self) -> None:
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self) -> None:
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("run-relaxations")
class RelaxationTask(BaseTask):
    def run(self) -> None:
        assert isinstance(
            self.trainer, OCPTrainer
        ), "Relaxations are only possible for ForcesTrainer"
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()
