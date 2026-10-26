"""CSV, export, and real training smoke checks using a small local fixture."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch_geometric.data import Batch, Data

from fairchem.core.datasets import LmdbDataset
from fairchem.core.modules.local_property import (
    collect_local_results,
    local_settings,
    plot_local_performance,
)
from fairchem.core.trainers.ocp_trainer import OCPTrainer

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "sites.csv"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(monkeypatch, output, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--csv_file",
            str(FIXTURE),
            "--material_id",
            "material_id",
            "--target_property",
            "spin",
            "--property_level",
            "atom",
            "--output_dir",
            str(output),
            "--no_analysis",
            *extra,
        ],
    )
    script("prepare_data").main()


def test_preparation_order_apply_splits_and_rejection(tmp_path, monkeypatch):
    prepare(monkeypatch, tmp_path / "data", "--split_ratios", "0.5", "0.25", "0.25")
    original = pd.read_csv(FIXTURE).set_index("material_id")
    for split in ("train", "val", "test"):
        dataset = LmdbDataset({"src": str(tmp_path / "data" / f"{split}.lmdb")})
        for index in range(len(dataset)):
            data = dataset[index]
            assert data.spin.shape == (int(data.natoms), 1)
            assert data.spin.dtype == torch.float32
            assert np.allclose(
                data.spin[:, 0], json.loads(original.loc[data.material_id, "spin"])
            )
        del dataset
    prepare(monkeypatch, tmp_path / "apply", "--apply")
    dataset = LmdbDataset({"src": str(tmp_path / "apply" / "apply.lmdb")})
    assert "spin" not in dataset[0]
    assert len(dataset) == 6
    del dataset
    prepare(
        monkeypatch,
        tmp_path / "two_way",
        "--split_style",
        "two_way",
        "--split_ratios",
        "0.5",
        "0.5",
    )
    assert not (tmp_path / "two_way" / "val.lmdb").exists()
    prepare(
        monkeypatch,
        tmp_path / "explicit",
        "--test_csv_file",
        str(FIXTURE),
        "--val_csv_file",
        str(FIXTURE),
    )
    assert (tmp_path / "explicit" / "val.lmdb").exists()
    with pytest.raises(SystemExit):
        prepare(monkeypatch, tmp_path / "symmetrized", "--symmetrize")
    bad = pd.read_csv(FIXTURE)
    bad.loc[1, "spin"] = "[2]"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="material two"):
        prepare(
            monkeypatch,
            tmp_path / "bad_output",
            "--csv_file",
            str(tmp_path / "bad.csv"),
        )
    assert not (tmp_path / "bad_output").exists()


def test_prediction_archive_reorders_lists_and_csv_roundtrip(tmp_path, monkeypatch):
    prepare(
        monkeypatch,
        tmp_path / "data",
        "--split_style",
        "two_way",
        "--split_ratios",
        "0.5",
        "0.5",
    )
    path = str(tmp_path / "data" / "test.lmdb")
    dataset = LmdbDataset({"src": path})
    truth = [dataset[i].spin.numpy() for i in range(len(dataset))]
    del dataset
    order = list(reversed(range(len(truth))))
    arrays = [-truth[i] for i in order]
    archive = tmp_path / "predictions.npz"
    np.savez(
        archive,
        ids=[f"0_{i}" for i in order],
        local_property=np.concatenate(arrays),
        local_property_chunk_idx=np.cumsum([len(a) for a in arrays])[:-1],
    )
    result = collect_local_results(
        path, archive, "spin", "material_id", allow_global_sign_flip=True
    )
    for i in range(len(truth)):
        assert np.allclose(json.loads(result.loc[i, "spin ML"]), -truth[i][:, 0])
        assert np.allclose(json.loads(result.loc[i, "spin ML aligned"]), truth[i][:, 0])
    result.to_csv(tmp_path / "result.csv", index=False)
    reread = pd.read_csv(tmp_path / "result.csv")
    metrics = plot_local_performance(reread, "spin", tmp_path / "result.png")
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    training = script("MLIP_FTL").collect_result(
        path,
        archive,
        "spin",
        "material_id",
        property_level="atom",
        allow_global_sign_flip=True,
    )
    standalone = script("evaluate_result").collect_result(
        path,
        archive,
        "spin",
        "material_id",
        property_level="atom",
        allow_global_sign_flip=True,
    )
    pd.testing.assert_frame_equal(training, standalone)


@pytest.mark.parametrize(("only_free", "expected"), [(None, 1), (True, 1), (False, 2)])
def test_fixed_atom_export_option(only_free, expected):
    trainer = OCPTrainer.__new__(OCPTrainer)
    trainer.model = nn.Identity()
    trainer.is_debug = False
    trainer.ema = trainer.scaler = None
    trainer.normalizers = trainer.elementrefs = {}
    trainer.config = {
        "outputs": {"local_property": {"level": "atom", "prediction_dtype": "float32"}},
        "task": {},
    }
    if only_free is not None:
        trainer.config["outputs"]["local_property"]["predict_on_free_atoms"] = only_free
    trainer._forward = lambda batch: {"local_property": torch.tensor([[2.0], [-2.0]])}
    batch = Batch.from_data_list(
        [
            Data(
                natoms=2,
                num_nodes=2,
                fixed=torch.tensor([0, 1]),
                sid=torch.tensor([0]),
                fid=torch.tensor([0]),
            )
        ]
    )
    result = trainer.predict(batch, disable_tqdm=True)
    assert result["local_property"][0].shape == (expected, 1)
    assert result["local_property_chunk_idx"] == [expected]


@pytest.mark.parametrize("device", ["cpu", "cuda", "two_gpus"])
def test_training_checkpoint_and_unlabeled_application(tmp_path, monkeypatch, device):
    required = 2 if device == "two_gpus" else 1
    if device != "cpu" and torch.cuda.device_count() < required:
        pytest.skip("required GPUs unavailable")
    prepare(monkeypatch, tmp_path / "data", "--split_ratios", "0.5", "0.25", "0.25")
    prepare(monkeypatch, tmp_path / "apply", "--apply")
    mlip = script("MLIP_FTL")
    config_path = tmp_path / "local.yml"
    mlip.create_config_file(
        str(config_path),
        str(tmp_path / "data"),
        "spin",
        property_level="atom",
        allow_global_sign_flip=True,
        loss="mse",
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        cpu_only=device == "cpu",
    )
    config = yaml.safe_load(config_path.read_text())
    config["model"]["backbone"].update(
        sphere_channels=8,
        hidden_channels=8,
        edge_channels=8,
        num_distance_basis=8,
        lmax=1,
        mmax=1,
    )
    config["optim"]["scheduler_params"]["warmup_epochs"] = 1
    config_path.write_text(yaml.safe_dump(config))
    env = dict(
        os.environ,
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "--mode",
        "train",
        "--config-yml",
        str(config_path),
        "--run-dir",
        str(tmp_path / "run"),
        "--identifier",
        "local_smoke",
    ]
    if device == "cpu":
        cmd.append("--cpu")
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "two_gpus":
        cmd += ["--num-gpus", "2"]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=150, check=False
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-7000:]
    checkpoint_path = next((tmp_path / "run" / "checkpoints").glob("*/checkpoint.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["trainer"] == "mlip_local_property_trainer"
    assert local_settings(checkpoint["config"])["loss"] == "mse"
    # Prediction uses only the checkpoint, never the source YAML or training LMDBs.
    config_path.unlink()
    (tmp_path / "data").rename(tmp_path / "unused_training_data")
    monkeypatch.chdir(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cpdir = mlip.run_application(
        str(checkpoint_path),
        str(tmp_path / "apply" / "apply.lmdb"),
        str(tmp_path / "prediction"),
        "apply_local",
        cpu_only=device == "cpu",
    )
    assert cpdir is not None
    archive = cpdir.replace("checkpoints", "results") + "/ocp_predictions.npz"
    output = mlip.collect_result(
        str(tmp_path / "apply" / "apply.lmdb"),
        archive,
        "spin",
        "material_id",
        application=True,
        **local_settings(checkpoint["config"]),
    )
    expected = pd.read_csv(FIXTURE)
    assert [len(json.loads(value)) for value in output["spin ML"]] == [
        len(json.loads(value)) for value in expected["numbers"]
    ]
    assert "spin DFT" not in output


def test_legacy_global_workflow_and_local_frozen_transfer(tmp_path, monkeypatch):
    """A global checkpoint remains usable and initializes both local TL modes."""
    prepare(
        monkeypatch,
        tmp_path / "global_data",
        "--property_level",
        "system",
        "--target_property",
        "global_value",
        "--split_ratios",
        "0.5",
        "0.25",
        "0.25",
    )
    prepare(
        monkeypatch, tmp_path / "local_data", "--split_ratios", "0.5", "0.25", "0.25"
    )
    prepare(monkeypatch, tmp_path / "apply", "--apply")
    mlip = script("MLIP_FTL")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        MPLCONFIGDIR=str(tmp_path / "mpl"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    base_checkpoint = None
    for mode in ("global", "full", "partial"):
        config_path = tmp_path / f"{mode}.yml"
        options = (
            {}
            if mode == "global"
            else {
                "property_level": "atom",
                "allow_global_sign_flip": mode == "partial",
                "transfer_learning": True,
                "tl_mode": mode,
                "transfer_layers": 1,
                "frozen_layers": 1,
            }
        )
        mlip.create_config_file(
            str(config_path),
            str(tmp_path / ("global_data" if mode == "global" else "local_data")),
            "global_value" if mode == "global" else "spin",
            num_layers=1,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            cpu_only=True,
            **options,
        )
        config = yaml.safe_load(config_path.read_text())
        config["model"]["backbone"].update(
            sphere_channels=8,
            hidden_channels=8,
            edge_channels=8,
            num_distance_basis=8,
            lmax=1,
            mmax=1,
        )
        config_path.write_text(yaml.safe_dump(config))
        run_dir = tmp_path / mode
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "--mode",
            "train",
            "--config-yml",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--identifier",
            mode,
            "--cpu",
        ]
        if base_checkpoint is not None:
            cmd += ["--checkpoint", str(base_checkpoint)]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=150, check=False
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-7000:]
        checkpoint_path = next((run_dir / "checkpoints").glob("*/checkpoint.pt"))
        if mode == "global":
            base_checkpoint = checkpoint_path
            monkeypatch.chdir(tmp_path)
            for key, value in env.items():
                monkeypatch.setenv(key, value)
            cpdir = mlip.run_application(
                str(base_checkpoint),
                str(tmp_path / "apply" / "apply.lmdb"),
                str(tmp_path / "global_apply"),
                "global_apply",
                cpu_only=True,
            )
            assert cpdir is not None
            archive = cpdir.replace("checkpoints", "results") + "/ocp_predictions.npz"
            frame = mlip.collect_result(
                str(tmp_path / "apply" / "apply.lmdb"),
                archive,
                "global_value",
                "material_id",
                application=True,
            )
            assert len(frame) == 6
            assert np.issubdtype(frame["global_value ML"].dtype, np.number)
        else:
            base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
            local = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            for key, value in local["state_dict"].items():
                if "backbone.blocks.0." in key:
                    assert torch.equal(value, base["state_dict"][key]), key
            assert set(local["normalizers"]) == {"local_property"}
