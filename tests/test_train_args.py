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


def test_hf_setup_is_deferred_to_trainer(tmp_path):
    """Reach Trainer.fit without touching the HF dataset or assuming splits exist."""
    args = get_argparse().parse_args(
        ["--dataset", "HFDataset", "--log-dir", str(tmp_path)]
    )
    # Deliberately omit idx_test, which does not exist until Lightning calls setup.
    data = Mock(spec=["dataset", "mean", "std", "prepare_data", "setup"])
    data.dataset = data.mean = data.std = None

    def fit(model, datamodule, **kwargs):
        """Check the startup contract and stop before launching a training loop."""
        assert datamodule is data
        data.prepare_data.assert_not_called()
        data.setup.assert_not_called()
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
