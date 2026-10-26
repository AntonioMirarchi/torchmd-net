# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

"""Normalization semantics for transfer learning, inference, and post-fit tests."""

from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from torchmdnet.data import DataModule
from torchmdnet.models.model import create_model, create_prior_models, load_model
from torchmdnet.scripts.train import fix_state_dict, get_argparse, main
from utils import DummyDataset, create_example_batch


@pytest.fixture
def model_args(tmp_path):
    args = get_argparse().parse_args([])
    vars(args).update(
        model="tensornet2",
        q_dim=2,
        embedding_dimension=8,
        num_layers=1,
        num_rbf=4,
        max_num_neighbors=16,
        derivative=True,
        log_dir=str(tmp_path),
    )
    return args


@pytest.mark.parametrize("shape", [(), (1,), (2,)])
@pytest.mark.parametrize("prefix", ["", "model."])
def test_inference_restores_normalization_shapes_and_predictions(
    tmp_path, model_args, shape, prefix
):
    """Standalone and Lightning weights retain scalar or vector normalization."""
    mean, std = torch.full(shape, 2.5), torch.full(shape, 1.5)
    model = create_model(vars(model_args), mean=mean, std=std).eval()
    path = tmp_path / "normalization.ckpt"
    torch.save(
        {
            "hyper_parameters": vars(model_args),
            "state_dict": {prefix + k: v for k, v in model.state_dict().items()},
        },
        path,
    )
    loaded = load_model(str(path)).eval()
    torch.testing.assert_close(loaded.mean, mean)
    torch.testing.assert_close(loaded.std, std)
    z, pos, batch = create_example_batch()
    expected = model(z, pos.detach().clone(), batch=batch)
    actual = loaded(z, pos.detach().clone(), batch=batch)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_fix_state_dict_excludes_only_root_normalization():
    """Transfer normalization is excluded without changing the source checkpoint."""
    state = {
        "model.mean": torch.tensor([2.5]),
        "model.std": torch.tensor([1.5]),
        "model.representation_model.weight": torch.ones(2),
        "model.output_model.output_network.0.weight": torch.ones(2, 2),
        "model.unexpected_parameter": torch.zeros(1),
    }
    adapted = fix_state_dict({"state_dict": state})
    assert "mean" not in adapted and "std" not in adapted
    assert "model.mean" in state and "model.std" in state
    assert set(adapted) == {
        "representation_model.weight",
        "output_model.output_network.layers.0.weight",
        "unexpected_parameter",
    }
    kept = fix_state_dict({"state_dict": state}, reset_normalization=False)
    torch.testing.assert_close(kept["mean"], state["model.mean"])
    torch.testing.assert_close(kept["std"], state["model.std"])


@pytest.mark.parametrize("invalid_weights", [False, True])
def test_load_weights_keeps_target_statistics_and_strict_loading(
    tmp_path, model_args, invalid_weights
):
    """Transfer learned parameters but retain the target data's normalization."""
    source = create_model(
        vars(model_args), mean=torch.tensor([9.0, 10.0]), std=torch.tensor([4.0, 5.0])
    )
    state = {"model." + k: v for k, v in source.state_dict().items()}
    if invalid_weights:
        state.pop("model.representation_model.linear.weight")
    path = tmp_path / "pretrained.ckpt"
    torch.save({"state_dict": state}, path)
    model_args.load_weights = str(path)
    model_args.load_weights_normalization = "dataset"
    model_args.standardize = True
    data = Mock(spec=["dataset", "mean", "std", "prepare_data", "setup"])
    data.dataset = None
    data.mean, data.std = torch.tensor([1.0]), torch.tensor([2.0])

    def fit(model, datamodule, **kwargs):
        torch.testing.assert_close(model.model.mean, data.mean)
        torch.testing.assert_close(model.model.std, data.std)
        for name, parameter in model.model.named_parameters():
            torch.testing.assert_close(parameter, dict(source.named_parameters())[name])
        raise RuntimeError("Verified transferred weights and target statistics")

    with (
        patch("torchmdnet.scripts.train.get_args", return_value=model_args),
        patch("torchmdnet.scripts.train.DataModule", return_value=data),
        patch("torchmdnet.scripts.train.pl.Trainer") as trainer,
    ):
        trainer.return_value.fit.side_effect = fit
        message = "Missing key" if invalid_weights else "Verified transferred weights"
        with pytest.raises(RuntimeError, match=message):
            main()
        if invalid_weights:
            trainer.assert_not_called()


@pytest.mark.parametrize("use_ema", [False, True])
@pytest.mark.parametrize("prior_model", [None, "Atomref"])
@pytest.mark.parametrize("normalization_policy", [None, "checkpoint", "dataset"])
def test_standardized_training_reloads_best_checkpoint_for_test(
    tmp_path, model_args, use_ema, prior_model, normalization_policy
):
    """Exercise the real fit/reload/test utility with dataset-shaped statistics."""
    vars(model_args).update(
        standardize=True,
        num_epochs=1,
        batch_size=2,
        inference_batch_size=2,
        num_workers=0,
        train_size=4,
        val_size=2,
        test_size=2,
        ngpus=1,
        save_interval=1,
        early_stopping_monitor=None,
        decay_rate_ema_weights=0.5 if use_ema else None,
        ema_start_step=0,
        ema_update_steps=1,
        prior_model=prior_model,
    )
    if normalization_policy is not None:
        model_args.load_weights = str(tmp_path / "pretrained.ckpt")
        model_args.load_weights_normalization = normalization_policy
    input_path = tmp_path / "input.yaml"
    input_path.write_text(
        yaml.safe_dump({k: v for k, v in vars(model_args).items() if k != "conf"})
    )
    data = DataModule(
        vars(model_args),
        DummyDataset(
            num_samples=8, min_atoms=3, max_atoms=4, has_atomref=bool(prior_model)
        ),
    )
    source_mean, source_std = torch.tensor([7.0]), torch.tensor([3.0])
    if normalization_policy is not None:
        source = create_model(
            vars(model_args),
            prior_model=create_prior_models(vars(model_args), data.dataset),
            mean=source_mean,
            std=source_std,
        )
        torch.save(
            {"state_dict": {"model." + k: v for k, v in source.state_dict().items()}},
            model_args.load_weights,
        )
    from torchmdnet.module import LNNP

    with (
        patch("torchmdnet.scripts.train.get_args", return_value=model_args),
        patch("torchmdnet.scripts.train.DataModule", return_value=data),
        patch.object(
            LNNP, "load_from_checkpoint", wraps=LNNP.load_from_checkpoint
        ) as reload_model,
    ):
        main()
    path = reload_model.call_args.args[0]
    assert reload_model.call_args.kwargs["hparams_file"] == str(input_path)
    if normalization_policy == "checkpoint":
        expected_mean, expected_std = source_mean, source_std
    else:
        expected_mean, expected_std = data.mean, data.std
    torch.testing.assert_close(reload_model.call_args.kwargs["mean"], expected_mean)
    torch.testing.assert_close(reload_model.call_args.kwargs["std"], expected_std)
    loaded = load_model(path).eval()
    torch.testing.assert_close(loaded.mean, expected_mean)
    torch.testing.assert_close(loaded.std, expected_std)
    if prior_model:
        assert reload_model.call_args.kwargs["prior_args"][0]["max_z"] == 100


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("policy", ["checkpoint", "dataset"])
def test_load_weights_normalization_policy(tmp_path, model_args, standardize, policy):
    """Checkpoint statistics preserve initial predictions even without standardize."""
    source_mean, source_std = torch.tensor([2.5]), torch.tensor([1.5])
    source = create_model(
        vars(model_args), mean=source_mean, std=source_std
    ).eval()
    path = tmp_path / "pretrained.ckpt"
    torch.save(
        {"state_dict": {"model." + k: v for k, v in source.state_dict().items()}}, path
    )
    model_args.load_weights = str(path)
    model_args.load_weights_normalization = policy
    model_args.standardize = standardize
    data = Mock(spec=["dataset", "mean", "std", "prepare_data", "setup"])
    data.dataset = None
    data.mean = torch.tensor([1.0]) if standardize else None
    data.std = torch.tensor([4.0]) if standardize else None
    if policy == "checkpoint":
        expected_mean, expected_std = source_mean, source_std
    elif standardize:
        expected_mean, expected_std = data.mean, data.std
    else:
        expected_mean, expected_std = torch.tensor(0.0), torch.tensor(1.0)

    def fit(model, datamodule, **kwargs):
        torch.testing.assert_close(model.model.mean, expected_mean)
        torch.testing.assert_close(model.model.std, expected_std)
        z, pos, batch = create_example_batch()
        expected_energy, expected_forces = source(z, pos.detach().clone(), batch)
        energy, forces = model.model.eval()(z, pos.detach().clone(), batch)
        if policy == "checkpoint":
            torch.testing.assert_close(energy, expected_energy)
            torch.testing.assert_close(forces, expected_forces)
        else:
            # Replacing std rescales the learned forces before any optimizer step.
            torch.testing.assert_close(
                forces, expected_forces * expected_std / source_std
            )
        raise RuntimeError("Verified normalization policy")

    with (
        patch("torchmdnet.scripts.train.get_args", return_value=model_args),
        patch("torchmdnet.scripts.train.DataModule", return_value=data),
        patch("torchmdnet.scripts.train.pl.Trainer") as trainer,
    ):
        trainer.return_value.fit.side_effect = fit
        with pytest.raises(RuntimeError, match="Verified normalization policy"):
            main()


def test_checkpoint_normalization_requires_saved_statistics(tmp_path, model_args):
    source = create_model(vars(model_args))
    path = tmp_path / "weights_only.ckpt"
    torch.save({"state_dict": fix_state_dict({"state_dict": source.state_dict()})}, path)
    model_args.load_weights = str(path)
    data = Mock(spec=["dataset", "mean", "std", "prepare_data", "setup"])
    data.dataset = data.mean = data.std = None
    with (
        patch("torchmdnet.scripts.train.get_args", return_value=model_args),
        patch("torchmdnet.scripts.train.DataModule", return_value=data),
    ):
        with pytest.raises(ValueError, match="--load-weights-normalization dataset"):
            main()
