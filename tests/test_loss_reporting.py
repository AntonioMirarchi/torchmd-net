"""Epoch reporting must preserve element counts without changing batch gradients."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from lightning.pytorch import Trainer

from torchmdnet.module import LNNP
from torchmdnet.loss import loss_class_mapping
from utils import load_example_args


def make_module(loss_name="mse_loss", reduction="mean", alpha=1.0):
    """Build a small wrapper with nontrivial energy/force weights and optional EMA."""
    args = load_example_args("tensornet", remove_prior=True)
    args.update(
        embedding_dimension=8,
        num_rbf=4,
        num_layers=0,
        derivative=True,
        train_loss=loss_name,
        train_loss_arg={"reduction": reduction},
        ema_alpha_y=alpha,
        ema_alpha_neg_dy=alpha,
        y_weight=2.0,
        neg_dy_weight=0.5,
        test_interval=-1,
    )
    return LNNP(args)


def structures():
    """Five differently sized structures with known energy and force errors."""
    return [
        Data(
            z=torch.ones(n, dtype=torch.long),
            pos=torch.full((n, 3), float(error)),
            y=torch.zeros(1, 1),
            neg_dy=torch.zeros(n, 3),
        )
        for n, error in zip([2, 4, 3, 5, 1], [1, 3, 2, 6, 8])
    ]


def predictions(batch):
    """Use the first bead's coordinate as energy, and all coordinates as forces."""
    return batch.pos[batch.ptr[:-1], :1].clone(), batch.pos.clone()


@pytest.mark.parametrize("stage", ["train", "val", "test"])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("loss_name", ["mse_loss", "l1_loss", "huber_loss"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_epoch_loss_matches_whole_dataset(stage, batch_size, loss_name, reduction):
    """Unequal batches and graph sizes give the same pooled metric, even with EMA."""
    module = make_module(loss_name, reduction, alpha=0.3)
    if stage == "test":
        loss_name, reduction = "l1_loss", "mean"
        loss_fn = loss_class_mapping[loss_name]
    else:
        loss_fn = module.train_loss_fn
    for batch in DataLoader(structures(), batch_size=batch_size):
        energy, forces = predictions(batch)
        module._compute_losses(energy, forces, batch, loss_fn, loss_name, stage)
    metrics = module._get_epoch_loss_dict(stage)
    whole = Batch.from_data_list(structures())
    energy, forces = predictions(whole)
    expected_y = loss_class_mapping[loss_name](energy, whole.y, reduction=reduction)
    expected_f = loss_class_mapping[loss_name](
        forces, whole.neg_dy, reduction=reduction
    )
    assert metrics[f"{stage}_y_{loss_name}"].item() == pytest.approx(expected_y.item())
    assert metrics[f"{stage}_neg_dy_{loss_name}"].item() == pytest.approx(
        expected_f.item()
    )
    assert metrics[f"{stage}_total_{loss_name}"].item() == pytest.approx(
        2 * expected_y.item() + 0.5 * expected_f.item()
    )


@pytest.mark.parametrize("alpha", [1.0, 0.3])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_reporting_preserves_batch_loss_and_gradients(alpha, reduction):
    """Detached metric collection must not alter the existing batch-loss/EMA path."""
    module = make_module(reduction=reduction, alpha=alpha)
    previous = {}
    for batch in DataLoader(structures(), batch_size=2):
        energy, forces = [x.requires_grad_() for x in predictions(batch)]
        losses = module._compute_losses(
            energy, forces, batch, module.train_loss_fn, "mse_loss", "train"
        )
        expected = {}
        for name, prediction, target in [
            ("y", energy, batch.y),
            ("neg_dy", forces, batch.neg_dy),
        ]:
            raw = torch.nn.functional.mse_loss(prediction, target, reduction=reduction)
            expected[name] = alpha * raw + (1 - alpha) * previous.get(
                name, raw.detach()
            )
            previous[name] = expected[name].detach()
            torch.testing.assert_close(losses[name], expected[name])
        actual_loss = 2 * losses["y"] + 0.5 * losses["neg_dy"]
        expected_loss = 2 * expected["y"] + 0.5 * expected["neg_dy"]
        actual_grads = torch.autograd.grad(
            actual_loss, (energy, forces), retain_graph=True
        )
        expected_grads = torch.autograd.grad(expected_loss, (energy, forces))
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference)
        assert not module.losses["train"]["mse_loss"].requires_grad


def test_epoch_loss_combines_rank_sums_before_dividing():
    """Simulate unequal rank workloads using the same strategy.reduce interface."""
    modules = [make_module(), make_module()]
    for module, samples in zip(modules, [structures()[:2], structures()[2:]]):
        batch = Batch.from_data_list(samples)
        module._compute_losses(
            *predictions(batch), batch, module.train_loss_fn, "mse_loss", "train"
        )
    global_statistics = sum(m.losses["train"]["mse_loss"] for m in modules)
    for module in modules:
        strategy = Mock()
        strategy.reduce.return_value = global_statistics.clone()
        module._trainer = SimpleNamespace(strategy=strategy)
        metrics = module._get_epoch_loss_dict("train")
        strategy.reduce.assert_called_once()
        assert strategy.reduce.call_args.kwargs == {"reduce_op": "sum"}
        assert metrics["train_y_mse_loss"].item() == pytest.approx(114 / 5)
        assert metrics["train_neg_dy_mse_loss"].item() == pytest.approx(294 / 15)


class FixedPredictions(nn.Module):
    """Deterministic predictions for testing Lightning's validation/test lifecycle."""

    def forward(self, z, pos, batch, **kwargs):
        first = torch.cat([torch.ones(1, dtype=torch.bool), batch[1:] != batch[:-1]])
        return pos[first, :1], pos


def test_lightning_reporting_and_stage_resets(tmp_path):
    """Actual Lightning runs retain metric names and reset repeated evaluation runs."""
    module = make_module("l1_loss", alpha=0.3)
    module.model = FixedPredictions()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        inference_mode=False,
    )
    for batch_size in [2, 5]:
        loader = DataLoader(structures(), batch_size=batch_size)
        result = trainer.validate(module, dataloaders=loader, verbose=False)[0]
        assert result["val_y_l1_loss"] == pytest.approx(4)
        assert result["val_neg_dy_l1_loss"] == pytest.approx(58 / 15)
        assert result["val_total_l1_loss"] == pytest.approx(8 + 0.5 * 58 / 15)
        assert module.losses["val"]["l1_loss"][0, 1].item() == 5
        result = trainer.test(module, dataloaders=loader, verbose=False)[0]
        assert result["test_y_l1_loss"] == pytest.approx(4)

    # Validation start/end must not reset the training epoch's statistics.
    module._trainer = None
    batch = Batch.from_data_list(structures())
    module._compute_losses(
        *predictions(batch), batch, module.train_loss_fn, "l1_loss", "train"
    )
    before = module.losses["train"]["l1_loss"].clone()
    module.on_validation_epoch_start()
    torch.testing.assert_close(module.losses["train"]["l1_loss"], before)
    assert module._get_epoch_loss_dict("val") == {}
    module.on_train_epoch_start()
    assert module._get_epoch_loss_dict("train") == {}


@pytest.mark.parametrize("target", ["y", "neg_dy"])
def test_single_target_reporting(target):
    """Force-only and energy-only datasets retain the configured metric names."""
    module = make_module()
    missing = "neg_dy" if target == "y" else "y"
    module.hparams[f"{missing}_weight"] = 0.0
    batch = Batch.from_data_list(structures())
    del batch[missing]
    module._compute_losses(
        *predictions(batch), batch, module.train_loss_fn, "mse_loss", "train"
    )
    result = module._get_epoch_loss_dict("train")
    assert f"train_{missing}_mse_loss" not in result
    assert result["train_total_mse_loss"] == (
        result[f"train_{target}_mse_loss"] * module.hparams[f"{target}_weight"]
    )
