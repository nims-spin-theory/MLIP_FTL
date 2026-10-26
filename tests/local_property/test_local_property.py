"""Local-property contracts, independent of the development dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from fairchem.core.models.esen.esen import MLPScalarSiteHead, eSEN_Backbone
from fairchem.core.models.esen.trainers.local_property_trainer import (
    LocalPropertyTrainer,
)
from fairchem.core.modules.local_property import (
    aligned_prediction,
    distributed_structure_mean,
    local_errors,
    local_settings,
    metric_statistics,
    normalization_statistics,
    parse_local_target,
    statistics_metrics,
)
from fairchem.core.modules.normalization.normalizer import create_normalizer

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "value",
    [
        "[1]",
        "[[1], [2]]",
        "[1, None]",
        "[True, False]",
        "[1, True]",
        "[1, '2']",
        "1",
        [1, float("nan")],
    ],
)
def test_reject_invalid_targets(value):
    with pytest.raises((ValueError, SyntaxError)):
        parse_local_target(value, 2)


def test_target_order_and_type():
    result = parse_local_target("[-2, 1, 0]", 3)
    assert result.dtype == np.float32
    assert result.shape == (3, 1)
    assert result[:, 0].tolist() == [-2, 1, 0]


def test_head_retains_single_atom_and_permutation():
    torch.manual_seed(10)
    head = MLPScalarSiteHead(SimpleNamespace(sphere_channels=4, hidden_channels=8))
    embedding = torch.randn(5, 4, 4)
    out = head(None, {"node_embedding": embedding})["local_property"]
    assert out.shape == (5, 1)
    permutation = torch.tensor([4, 2, 0, 1, 3])
    assert torch.allclose(
        head(None, {"node_embedding": embedding[permutation]})["local_property"],
        out[permutation],
    )
    assert head(None, {"node_embedding": embedding[:1]})["local_property"].shape == (
        1,
        1,
    )
    # Spatial rotations mix the higher-order channels; the scalar channel is unchanged.
    rotated = embedding.clone()
    rotated[:, 1:] = torch.randn_like(rotated[:, 1:])
    assert torch.equal(head(None, {"node_embedding": rotated})["local_property"], out)


def test_full_esen_local_scalar_rotation_and_permutation():
    from ase import Atoms

    from fairchem.core.preprocessing import AtomsToGraphs

    torch.manual_seed(3)
    backbone = eSEN_Backbone(
        num_layers=1,
        sphere_channels=8,
        hidden_channels=8,
        edge_channels=8,
        num_distance_basis=8,
        lmax=1,
        mmax=1,
        act_type="gate",
        mlp_type="spectral",
        use_envelope=True,
        use_pbc=True,
        use_pbc_single=True,
        otf_graph=True,
        regress_forces=False,
        regress_stress=False,
        direct_forces=False,
    ).eval()
    head = MLPScalarSiteHead(backbone).eval()
    atoms = Atoms(
        numbers=[26, 8, 8],
        positions=[[0, 0, 0], [1, 1.1, 1], [2, 2, 2]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    converter = AtomsToGraphs(r_edges=False, r_pbc=True)

    def predict(structure):
        batch = Batch.from_data_list([converter.convert(structure)])
        with torch.no_grad():
            return head(batch, backbone(batch))["local_property"]

    original = predict(atoms)
    rotated = atoms.copy()
    rotation, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    rotated.positions = atoms.positions @ rotation
    rotated.set_cell(atoms.cell.array @ rotation)
    assert torch.allclose(original, predict(rotated), atol=1e-5, rtol=1e-4)
    permutation = [2, 0, 1]
    assert torch.allclose(
        original[permutation], predict(atoms[permutation]), atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("loss", ["mae", "mse"])
def test_whole_structure_alignment(loss):
    target = torch.tensor([[2.0], [-2.0], [1.0], [-3.0], [0.0]])
    pred = target * torch.tensor([[-1.0], [-1.0], [1.0], [1.0], [1.0]])
    membership = torch.tensor([0, 0, 1, 1, 2])
    aligned, signs = aligned_prediction(pred, target, membership, 3, loss, True)
    assert signs.tolist() == [-1, 1, 1]
    assert torch.equal(aligned, target)
    unchanged, _ = aligned_prediction(pred, target, membership, 3, loss, False)
    assert torch.equal(unchanged, pred)
    bad = pred.clone()
    bad[0] *= -1
    aligned, _ = aligned_prediction(bad, target, membership, 3, loss, True)
    assert local_errors(aligned, target, membership, 3, loss).sum() > 0


def test_mae_and_mse_can_choose_different_signs():
    pred = torch.tensor([[100.0], [1.0], [1.0]])
    target = torch.tensor([[-0.1], [2.0], [2.0]])
    membership = torch.zeros(3, dtype=torch.long)
    _, mae_sign = aligned_prediction(pred, target, membership, 1, "mae", True)
    _, mse_sign = aligned_prediction(pred, target, membership, 1, "mse", True)
    assert mae_sign.item() == 1
    assert mse_sign.item() == -1


def test_structure_weights_and_gradients():
    pred = torch.tensor([[4.0], [2.0], [2.0], [2.0]], requires_grad=True)
    target = torch.zeros_like(pred)
    membership = torch.tensor([0, 1, 1, 1])
    errors = local_errors(pred, target, membership, 2, "mae")
    loss = distributed_structure_mean(errors)
    assert loss.item() == 3
    loss.backward()
    assert torch.allclose(pred.grad[:, 0], torch.tensor([0.5, 1 / 6, 1 / 6, 1 / 6]))
    stats = metric_statistics(pred, target, membership, 2)
    metrics = statistics_metrics(stats)
    assert metrics["mae"] == 3
    assert metrics["rmse"] == pytest.approx(np.sqrt(10))
    assert np.isnan(metrics["r2"])


def test_r2_for_small_nonconstant_targets():
    target = torch.tensor([[1e-12], [2e-12]], dtype=torch.float64)
    stats = metric_statistics(target, target, torch.tensor([0, 1]), 2)
    assert statistics_metrics(stats)["r2"] == 1


def test_normalization_structure_weights_and_sign_invariance():
    mean, scale = normalization_statistics([[4], [0, 0, 0]])
    assert mean == 2
    assert scale == 2
    assert normalization_statistics([[0], [0, 0]], True) == (0, 1)
    a = normalization_statistics([[2, -3], [5]], True)
    assert a == normalization_statistics([[-2, 3], [5]], True)
    assert a[0] == 0
    assert a[1] == pytest.approx(np.sqrt((6.5 + 25) / 2))


def minimal_trainer(loss="mae", flip=True, mean=0, scale=1):
    trainer = LocalPropertyTrainer.__new__(LocalPropertyTrainer)
    trainer.local_loss = loss
    trainer.allow_global_sign_flip = flip
    trainer.local_coefficient = 1
    trainer.normalizers = {
        "local_property": create_normalizer(mean=float(mean), rmsd=float(scale))
    }
    trainer.elementrefs = {}
    trainer.device = torch.device("cpu")
    return trainer


def test_alignment_in_physical_units_with_nonzero_mean():
    trainer = minimal_trainer(mean=10, scale=2)
    batch = Batch.from_data_list(
        [Data(local_property=torch.tensor([[2.0], [-2.0]]), num_nodes=2, natoms=2)]
    )
    physical = torch.tensor([[-2.0], [2.0]])
    normalized = trainer.normalizers["local_property"].norm(physical).requires_grad_()
    out = {"local_property": normalized}
    loss = trainer._compute_loss(out, batch)
    assert loss.item() == 0
    loss.backward()
    metrics = trainer._compute_metrics(out, batch, None)
    assert metrics["local_property_mae"]["metric"] == 0
    assert metrics["local_property_rmse"]["metric"] == 0
    assert metrics["local_property_r2"]["metric"] == 1


def test_metrics_are_batch_partition_independent():
    trainer = minimal_trainer(flip=False)
    structures = [
        Data(local_property=torch.tensor([[1.0]]), natoms=1, num_nodes=1),
        Data(local_property=torch.tensor([[2.0], [3.0], [4.0]]), natoms=3, num_nodes=3),
    ]
    batch = Batch.from_data_list(structures)
    prediction = torch.tensor([[2.0], [5.0], [5.0], [5.0]])
    combined = trainer._compute_metrics({"local_property": prediction}, batch, None)
    separate = None
    for data, pred in zip(structures, (prediction[:1], prediction[1:])):
        separate = trainer._compute_metrics(
            {"local_property": pred}, Batch.from_data_list([data]), None, separate
        )
    assert combined == separate
    assert trainer._aggregate_metrics(combined) == combined


def test_local_config_and_legacy_defaults(tmp_path):
    script = load_script("MLIP_FTL")
    common = {
        "config_path": str(tmp_path / "config.yml"),
        "data_dir": str(tmp_path),
        "target_property": "spin",
        "auto_normalize": False,
    }
    script.create_config_file(**common)
    import yaml

    legacy = yaml.safe_load((tmp_path / "config.yml").read_text())
    assert legacy["trainer"] == "mlip_trainer"
    assert legacy["outputs"] == {"energy": {"level": "system", "property": "energy"}}
    assert (
        legacy["dataset"]["train"]["transforms"]["normalizer"]["energy"]["mean"] == 350
    )
    script.create_config_file(
        **common, property_level="atom", allow_global_sign_flip=True, loss="mse"
    )
    local = yaml.safe_load((tmp_path / "config.yml").read_text())
    assert local_settings(local) == {
        "property_level": "atom",
        "allow_global_sign_flip": True,
        "loss": "mse",
    }
    assert local["dataset"]["train"]["transforms"]["normalizer"]["local_property"] == {
        "mean": 0,
        "stdev": 1,
    }
    assert local["outputs"]["local_property"]["predict_on_free_atoms"] is False
    with pytest.raises(ValueError):
        script.create_config_file(**common, allow_global_sign_flip=True)
    with pytest.raises(ValueError):
        script.create_config_file(
            **common, property_level="atom", manual_mean=1, manual_stdev=0
        )


def test_local_checkpoint_restores_normalization_and_transfer_keeps_new_head(tmp_path):
    trainer = minimal_trainer()
    trainer.cpu = True
    trainer.model = nn.Module()
    trainer.model.backbone = nn.Linear(2, 2)
    trainer.model.output_heads = nn.Linear(2, 1)
    trainer.config = {
        "optim": {"ema_decay": 0.9, "local_property_transfer": True},
        "task": {},
    }
    trainer.ema = None
    original_head = trainer.model.output_heads.weight.detach().clone()
    checkpoint = {
        "state_dict": {
            "backbone.weight": torch.full((2, 2), 7.0),
            "backbone.bias": torch.ones(2),
            "output_heads.weight": torch.full((1, 2), 42.0),
        },
        "normalizers": {
            "local_property": create_normalizer(mean=99.0, rmsd=3.0).state_dict()
        },
        "elementrefs": {},
    }
    trainer.load_checkpoint("unused", checkpoint=checkpoint)
    assert torch.equal(trainer.model.output_heads.weight, original_head)
    assert torch.all(trainer.model.backbone.weight == 7)
    assert trainer.normalizers["local_property"].mean == 0
    assert torch.equal(trainer.ema.shadow_params[0], trainer.model.backbone.weight)
    trainer.config["optim"]["local_property_transfer"] = False
    trainer.scaler = None
    checkpoint["state_dict"] = trainer.model.state_dict()
    trainer.load_checkpoint("unused", checkpoint=checkpoint, inference_only=True)
    assert trainer.normalizers["local_property"].mean == 99


def test_validation_sampler_duplicates_do_not_change_structure_weights():
    trainer = minimal_trainer(flip=False)
    trainer._validation_records = {}
    one = Data(
        local_property=torch.tensor([[1.0]]),
        num_nodes=1,
        natoms=1,
        sid=torch.tensor([0]),
        fid=torch.tensor([0]),
    )
    two = Data(
        local_property=torch.tensor([[2.0], [3.0]]),
        num_nodes=2,
        natoms=2,
        sid=torch.tensor([0]),
        fid=torch.tensor([1]),
    )
    batch = Batch.from_data_list([one, two, one])
    pred = torch.tensor([[2.0], [5.0], [5.0], [2.0]])
    metrics = trainer._compute_metrics({"local_property": pred}, batch, None)
    result = trainer._aggregate_metrics(metrics)
    assert result["local_property_mae"]["numel"] == 2
    assert result["local_property_mae"]["metric"] == pytest.approx((1 + 2.5) / 2)
    assert result["loss"]["metric"] == pytest.approx((1 + 2.5) / 2)


def _distributed_worker(rank, rendezvous, result_path):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Rank zero has one 1-site structure; rank one has two structures.
        weight = torch.tensor(1.0, requires_grad=True)
        values = torch.tensor([2.0]) if rank == 0 else torch.tensor([4.0, 6.0])
        loss = distributed_structure_mean(weight * values)
        loss.backward()
        gradient = weight.grad.clone()
        dist.all_reduce(gradient)
        if rank == 0:
            torch.save(gradient / 2, result_path)
    finally:
        dist.destroy_process_group()


def test_distributed_structure_reduction(tmp_path):
    import torch.multiprocessing as mp

    result = str(tmp_path / "gradient.pt")
    mp.spawn(
        _distributed_worker,
        args=("file://" + str(tmp_path / "rendezvous"), result),
        nprocs=2,
        join=True,
    )
    assert torch.load(result, weights_only=True).item() == pytest.approx(4.0)
