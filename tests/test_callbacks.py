# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

"""Regression tests for EMA scheduling, validation, and checkpoint lifecycle."""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pytest
import torch
from torch.utils.data import DataLoader

from torchmdnet.callbacks import EMAWeightAveraging


class ConstantGradientModel(pl.LightningModule):
    """Take unit optimizer steps so EMA has a known, independent reference."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("offset", torch.tensor(2.0))
        self.validation_weights = []

    def training_step(self, batch, batch_idx):
        return self.weight

    def validation_step(self, batch, batch_idx):
        self.validation_weights.append(self.weight.item())
        self.log("val_loss", self.weight, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1.0)


def make_trainer(tmp_path, ema, max_epochs=2, accumulate_grad_batches=1):
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path, filename="epoch={epoch}", save_last=True, monitor="val_loss"
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[ema, checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    return trainer, checkpoint


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"decay": 0.0}, "decay"),
        ({"decay": 1.0}, "decay"),
        ({"decay": float("nan")}, "decay"),
        ({"decay": float("inf")}, "decay"),
        ({"decay": "0.9"}, "decay"),
        ({"decay": True}, "decay"),
        ({"start_step": -1}, "start_step"),
        ({"start_step": 0.5}, "start_step"),
        ({"start_step": True}, "start_step"),
        ({"update_every": 0}, "update_every"),
        ({"update_every": -1}, "update_every"),
        ({"update_every": 1.5}, "update_every"),
        ({"update_every": True}, "update_every"),
    ],
)
def test_ema_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EMAWeightAveraging(**{"decay": 0.5, **kwargs})


def test_ema_optimizer_step_schedule():
    ema = EMAWeightAveraging(decay=0.5, start_step=3, update_every=2)
    assert [step for step in range(9) if ema.should_update(step_idx=step)] == [3, 5, 7]
    assert not ema.should_update()
    assert not ema.should_update(epoch_idx=3)


@pytest.mark.parametrize("start_step", [1, 10])
def test_ema_validation_and_checkpoint_with_gradient_accumulation(tmp_path, start_step):
    """Skipped batches do not update EMA; pre-start validation keeps trained weights."""
    ema = EMAWeightAveraging(decay=0.5, start_step=start_step, update_every=2)
    model = ConstantGradientModel()
    trainer, checkpoint = make_trainer(tmp_path, ema, accumulate_grad_batches=2)
    loader = DataLoader(torch.zeros(4), batch_size=1)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 4
    state = torch.load(checkpoint.last_model_path, weights_only=False)
    torch.testing.assert_close(
        state["current_model_state"]["weight"], torch.tensor(-4.0)
    )
    if start_step == 1:
        # Updates after optimizer steps 2 and 4: seed -2, then 0.5*(-2) + 0.5*(-4).
        expected = -3.0
        assert ema._average_model.n_averaged.item() == 2
        assert model.validation_weights == [-2.0] * 4 + [-3.0] * 4
    else:
        expected = -4.0
        assert ema._average_model.n_averaged.item() == 0
        assert model.validation_weights == [-2.0] * 4 + [-4.0] * 4
    assert model.weight.item() == expected
    assert state["state_dict"]["weight"].item() == expected
    torch.testing.assert_close(
        state["averaging_state"]["n_averaged"], ema._average_model.n_averaged
    )
    assert state["state_dict"]["offset"].item() == 2.0


@pytest.mark.parametrize("start_step", [1, 4])
def test_ema_resume_matches_uninterrupted_training(tmp_path, start_step):
    """Resume both before and after the first EMA update without losing history."""
    loader = DataLoader(torch.zeros(4), batch_size=1)
    full_ema = EMAWeightAveraging(decay=0.5, start_step=start_step, update_every=2)
    full_model = ConstantGradientModel()
    full_trainer, _ = make_trainer(tmp_path / "full", full_ema)
    full_trainer.fit(full_model, train_dataloaders=loader, val_dataloaders=loader)

    first_ema = EMAWeightAveraging(decay=0.5, start_step=start_step, update_every=2)
    first_trainer, checkpoint = make_trainer(tmp_path / "first", first_ema, max_epochs=1)
    first_trainer.fit(
        ConstantGradientModel(), train_dataloaders=loader, val_dataloaders=loader
    )
    resumed_ema = EMAWeightAveraging(decay=0.5, start_step=start_step, update_every=2)
    resumed_model = ConstantGradientModel()
    resumed_trainer, _ = make_trainer(tmp_path / "resumed", resumed_ema)
    resumed_trainer.fit(
        resumed_model,
        train_dataloaders=loader,
        val_dataloaders=loader,
        ckpt_path=checkpoint.last_model_path,
    )
    assert resumed_trainer.global_step == full_trainer.global_step == 8
    torch.testing.assert_close(resumed_model.weight, full_model.weight)
    torch.testing.assert_close(
        resumed_ema._average_model.n_averaged, full_ema._average_model.n_averaged
    )


def test_ema_checkpoint_loads_for_torchmdnet_inference(tmp_path):
    """EMA training checkpoints expose averaged parameters through load_model."""
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    from torchmdnet.models.model import load_model
    from torchmdnet.module import LNNP
    from torchmdnet.scripts.train import get_argparse
    from utils import DummyDataset

    args = vars(get_argparse().parse_args([]))
    args.update(
        model="tensornet2",
        q_dim=2,
        embedding_dimension=8,
        num_layers=1,
        num_rbf=4,
        max_num_neighbors=16,
        derivative=True,
        decay_rate_ema_weights=0.5,
        ema_start_step=0,
        ema_update_steps=1,
        log_dir=str(tmp_path),
    )
    model = LNNP(args)
    ema = EMAWeightAveraging(decay=0.5, start_step=0, update_every=1)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_total_mse_loss")
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        callbacks=[ema, checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        inference_mode=False,
    )
    loader = GeometricDataLoader(
        DummyDataset(num_samples=4, min_atoms=3, max_atoms=4), batch_size=2
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert ema._average_model.n_averaged.item() == trainer.global_step == 2
    loaded = load_model(checkpoint.best_model_path, derivative=False).eval()
    batch = next(iter(loader))
    expected_energy, expected_forces = model(
        batch.z, batch.pos.detach().clone(), batch=batch.batch
    )
    pos = batch.pos.detach().clone().requires_grad_(True)
    energy, _ = loaded(batch.z, pos, batch=batch.batch)
    forces = -torch.autograd.grad(energy.sum(), pos)[0]
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(forces, expected_forces)
