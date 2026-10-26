# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import json
from unittest.mock import Mock, patch

import pytest

from torchmdnet.scripts.train import get_argparse, _requires_pre_fit_setup, main


def test_dataset_arg_cli_precedence(tmp_path):
    """Argument order controls YAML/CLI precedence; dataset dictionaries replace whole."""
    config = tmp_path / "input.yaml"
    config.write_text(
        "batch_size: 1\ndataset_arg:\n  paths: inputs.json\n  stride: 2\n"
    )
    override = {"paths": "other.json", "properties": []}
    cli = ["--batch-size", "4", "--dataset-arg", json.dumps(override)]
    args = get_argparse().parse_args(["--conf", str(config), *cli])
    assert args.batch_size == 4
    assert args.dataset_arg == override  # The dictionary is replaced, not merged.
    args = get_argparse().parse_args([*cli, "--conf", str(config)])
    assert args.batch_size == 1
    assert args.dataset_arg == {"paths": "inputs.json", "stride": 2}


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"dataset": "HFDataset"}, False),
        ({"dataset": "HFDataset", "prior_model": []}, False),
        ({"dataset": "HFDataset", "standardize": True}, True),
        ({"dataset": "HFDataset", "remove_ref_energy": True}, True),
        ({"dataset": "HFDataset", "prior_model": ["Atomref"]}, True),
        ({"dataset": "QM9"}, True),
    ],
)
def test_requires_pre_fit_setup(options, expected):
    """Defer HF setup except when model initialization needs statistics or priors."""
    args = get_argparse().parse_args([])
    vars(args).update(options)
    assert _requires_pre_fit_setup(args) == expected


@pytest.mark.parametrize("use_ema", [False, True])
def test_hf_setup_is_deferred_to_trainer(tmp_path, use_ema):
    """Reach Trainer.fit without touching the HF dataset or assuming splits exist."""
    args = get_argparse().parse_args(
        ["--dataset", "HFDataset", "--log-dir", str(tmp_path)]
    )
    if use_ema:
        args.decay_rate_ema_weights = 0.9
        args.ema_start_step = 2
        args.ema_update_steps = 3
    # Deliberately omit idx_test, which does not exist until Lightning calls setup.
    data = Mock(spec=["dataset", "mean", "std", "prepare_data", "setup"])
    data.dataset = data.mean = data.std = None

    def fit(model, datamodule, **kwargs):
        """Check the startup contract and stop before launching a training loop."""
        assert datamodule is data
        data.prepare_data.assert_not_called()
        data.setup.assert_not_called()
        from torchmdnet.callbacks import EMAWeightAveraging

        ema = [
            callback
            for callback in trainer.call_args.kwargs["callbacks"]
            if isinstance(callback, EMAWeightAveraging)
        ]
        assert len(ema) == int(use_ema)
        if use_ema:
            assert (ema[0].decay, ema[0].start_step, ema[0].update_every) == (0.9, 2, 3)
        raise RuntimeError("Reached Trainer.fit without eager setup")

    with (
        patch("torchmdnet.scripts.train.get_args", return_value=args),
        patch("torchmdnet.scripts.train.DataModule", return_value=data),
        patch("torchmdnet.scripts.train.LNNP"),
        patch("torchmdnet.scripts.train.pl.Trainer") as trainer,
    ):
        trainer.return_value.fit.side_effect = fit
        with pytest.raises(
            RuntimeError, match="Reached Trainer.fit without eager setup"
        ):
            main()


def test_ema_weight_config(tmp_path):
    """Keep the custom repo option names and respect YAML/CLI precedence."""
    parser = get_argparse()
    args = parser.parse_args([])
    assert args.decay_rate_ema_weights is None
    assert args.ema_start_step == 1000
    assert args.ema_update_steps == 10
    config = tmp_path / "ema.yaml"
    config.write_text(
        "decay_rate_ema_weights: 0.9\nema_start_step: 5\nema_update_steps: 2\n"
    )
    args = parser.parse_args(
        ["--conf", str(config), "--ema-start-step", "0", "--ema-update-steps", "3"]
    )
    assert args.decay_rate_ema_weights == 0.9
    assert args.ema_start_step == 0
    assert args.ema_update_steps == 3


def test_load_weights_normalization_config(tmp_path):
    parser = get_argparse()
    assert parser.parse_args([]).load_weights_normalization == "checkpoint"
    config = tmp_path / "normalization.yaml"
    config.write_text("load_weights_normalization: dataset\n")
    args = parser.parse_args(["--conf", str(config)])
    assert args.load_weights_normalization == "dataset"
    args = parser.parse_args(
        ["--conf", str(config), "--load-weights-normalization", "checkpoint"]
    )
    assert args.load_weights_normalization == "checkpoint"
    with pytest.raises(SystemExit):
        parser.parse_args(["--load-weights-normalization", "unknown"])
    config.write_text("load_weights_normalization: unknown\n")
    args = parser.parse_args(["--conf", str(config)])
    with (
        patch("torchmdnet.scripts.train.get_args", return_value=args),
        patch("torchmdnet.scripts.train.DataModule") as data,
    ):
        with pytest.raises(ValueError, match="load_weights_normalization"):
            main()
        data.assert_not_called()


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"decay_rate_ema_weights": 1.0}, "decay"),
        ({"ema_start_step": -1}, "start_step"),
        ({"ema_update_steps": 0}, "update_every"),
    ],
)
def test_ema_invalid_config_fails_before_data_setup(settings, message):
    args = get_argparse().parse_args([])
    args.decay_rate_ema_weights = 0.9
    vars(args).update(settings)
    with (
        patch("torchmdnet.scripts.train.get_args", return_value=args),
        patch("torchmdnet.scripts.train.DataModule") as data,
    ):
        with pytest.raises(ValueError, match=message):
            main()
        data.assert_not_called()


def test_cutoff_config(tmp_path):
    """Accept the YAML option, retain the cosine default, and allow CLI overrides."""
    from torchmdnet.models.model import create_model

    parser = get_argparse()
    assert parser.parse_args([]).cutoff_fn_type == "cosine"
    config = tmp_path / "input.yaml"
    config.write_text("cutoff_fn_type: polynomial\n")
    assert parser.parse_args(["--conf", str(config)]).cutoff_fn_type == "polynomial"
    args = parser.parse_args(["--conf", str(config), "--cutoff-fn-type", "cosine"])
    assert args.cutoff_fn_type == "cosine"
    with pytest.raises(SystemExit):
        parser.parse_args(["--cutoff-fn-type", "unknown"])
    # YAML values bypass argparse choices; model construction validates them too.
    config.write_text("cutoff_fn_type: unknown\n")
    args = parser.parse_args(["--conf", str(config)])
    with pytest.raises(ValueError, match="Unknown cutoff function"):
        create_model(vars(args))


def test_force_reduction_config(tmp_path):
    """Parse YAML/CLI overrides and loss kwargs without requiring reduction=none."""
    parser = get_argparse()
    assert parser.parse_args([]).force_loss_reduction == "atom_mean"
    config = tmp_path / "input.yaml"
    config.write_text(
        "force_loss_reduction: graph_mean\ntrain_loss: huber_loss\ntrain_loss_arg:\n  delta: 2.5\n"
    )
    args = parser.parse_args(["--conf", str(config)])
    assert args.force_loss_reduction == "graph_mean"
    assert args.train_loss_arg == {"delta": 2.5}
    args = parser.parse_args(
        [
            "--conf",
            str(config),
            "--force-loss-reduction",
            "graph_mean_inv_sqrt",
            "--train-loss-arg",
            '{"delta": 1.5}',
        ]
    )
    assert args.force_loss_reduction == "graph_mean_inv_sqrt"
    assert args.train_loss_arg == {"delta": 1.5}
    with pytest.raises(SystemExit):
        parser.parse_args(["--force-loss-reduction", "unknown"])


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"force_loss_reduction": "unknown"}, "Unknown force_loss_reduction"),
        ({"train_loss_arg": "not a dictionary"}, "must be a dictionary"),
        ({"train_loss_arg": []}, "must be a dictionary"),
        (
            {"train_loss_arg": {"reduction": "none"}},
            "unreduced force errors are computed internally",
        ),
        (
            {"force_loss_reduction": "graph_mean", "train_loss_arg": {"reduce": True}},
            "deprecated",
        ),
    ],
)
def test_force_reduction_invalid_config(settings, message):
    """Invalid YAML/checkpoint arguments fail before constructing the model."""
    from torchmdnet.module import LNNP

    args = vars(get_argparse().parse_args([]))
    args.update(settings)
    with pytest.raises(ValueError, match=message):
        LNNP(args)
