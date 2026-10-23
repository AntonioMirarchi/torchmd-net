"""Regression checks for latent NQE without an explicit Coulomb energy head."""

import pytest
import torch

from torchmdnet.models.model import create_model
from torchmdnet.models.tensornet2 import ChargePredict
from torchmdnet.scripts.train import get_argparse
from utils import create_example_batch


def model_args(**kwargs):
    """Use a small force-predicting TensorNet2 with neutral inputs by default."""
    args = vars(get_argparse().parse_args([]))
    args.update(
        model="tensornet2",
        embedding_dimension=16,
        num_rbf=8,
        num_layers=2,
        q_dim=4,
        derivative=True,
        output_model="Scalar",
        prior_model=None,
        max_num_neighbors=32,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("q_dim", [0, 4])
def test_scalar_nqe_backward_and_script(static_shapes, q_dim):
    """Force training uses every parameter, including the latent charge predictors."""
    torch.manual_seed(123)
    model = create_model(model_args(static_shapes=static_shapes, q_dim=q_dim))
    z, pos, batch = create_example_batch()
    energy, forces = model(z, pos, batch)
    assert torch.isfinite(forces).all()
    # The zero energy term retains the scalar bias in force-only autograd graphs.
    (forces.square().mean() + energy.sum() * 0).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert isinstance(model.representation_model.charge_predicts[-1], torch.nn.Identity)
    scripted = torch.jit.script(model)
    scripted_energy, scripted_forces = scripted(z, pos.detach().clone(), batch)
    torch.testing.assert_close(scripted_energy, energy)
    torch.testing.assert_close(scripted_forces, forces)


def test_legacy_unused_predictor_checkpoint():
    """Discard only obsolete unused predictors when loading scalar checkpoints."""
    model = create_model(model_args())
    state = model.state_dict()
    predictor = ChargePredict(16, "silu", 4)
    state.update(
        {
            f"representation_model.charge_predicts.1.{k}": v
            for k, v in predictor.state_dict().items()
        }
    )
    restored = create_model(model_args())
    restored.load_state_dict(state, strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    state["unrelated_parameter"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(state, strict=True)


@pytest.mark.parametrize("include_loop", [False, True])
@pytest.mark.parametrize("model_name", ["tensornet", "tensornet2"])
def test_neighbor_strategies(include_loop, model_name):
    """Both neighbor algorithms produce matching energies and forces."""
    args = model_args(
        model=model_name,
        q_dim=4 if model_name == "tensornet2" else 0,
        distance_include_loop=include_loop,
        static_shapes=False,
    )
    brute = create_model(args)
    cell = create_model(dict(args, distance_strategy="cell"))
    # The nonperiodic cell grid has its own bounding box, independent of weights.
    cell_box = cell.representation_model.distance.box.clone()
    cell.load_state_dict(brute.state_dict())
    cell.representation_model.distance.box.copy_(cell_box)
    z, pos, batch = create_example_batch()
    edges = cell.representation_model.distance(pos, batch)[0]
    assert bool((edges[0] == edges[1]).any()) == include_loop
    expected = brute(z, pos.clone(), batch)
    actual = cell(z, pos.clone(), batch)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


def test_nqe_arguments(tmp_path):
    """Boolean CLI/YAML options work and incompatible NQE settings fail early."""
    parser = get_argparse()
    args = parser.parse_args(
        [
            "--model",
            "tensornet2",
            "--q-dim",
            "8",
            "--distance-strategy",
            "cell",
            "--no-distance-include-loop",
        ]
    )
    assert args.q_dim == 8 and args.distance_strategy == "cell"
    assert args.distance_include_loop is False
    config = tmp_path / "input.yaml"
    config.write_text("distance_include_loop: false\n")
    assert parser.parse_args(["--conf", str(config)]).distance_include_loop is False
    with pytest.raises(ValueError, match="q_dim"):
        create_model(model_args(q_dim=-1))
    with pytest.raises(ValueError, match="q_dim"):
        create_model(model_args(model="tensornet"))
    with pytest.raises(ValueError, match="q_dim"):
        create_model(model_args(q_dim=0, output_model="ScalarPlusWeightedCoulomb"))


def test_force_only_lightning_training(tmp_path):
    """Optimize latent NQE using forces alone and report finite validation loss."""
    import lightning as pl
    from torch_geometric.loader import DataLoader
    from torchmdnet.module import LNNP
    from utils import DummyDataset

    module = LNNP(
        model_args(
            y_weight=0.0, neg_dy_weight=1.0, lr_warmup_steps=0, log_dir=str(tmp_path)
        )
    )
    predictor = module.model.representation_model.charge_predict_0
    before = predictor.q_mlp.layers[0].weight.detach().clone()
    loader = DataLoader(DummyDataset(num_samples=4, energy=False), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        inference_mode=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["val_total_mse_loss"])
    assert not torch.equal(before, predictor.q_mlp.layers[0].weight)
