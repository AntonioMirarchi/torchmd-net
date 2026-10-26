# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

"""Accept only the obsolete output-head std key across checkpoint loaders."""

from copy import deepcopy
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch

from torchmdnet.callbacks import EMAWeightAveraging
from torchmdnet.models.model import create_model, load_model
from torchmdnet.module import LNNP
from torchmdnet.scripts.train import fix_state_dict, get_argparse
from utils import create_example_batch


@pytest.fixture
def module():
    args = vars(get_argparse().parse_args([]))
    args.update(
        model="tensornet2",
        embedding_dimension=8,
        num_layers=1,
        num_rbf=4,
        q_dim=2,
        max_num_neighbors=16,
        derivative=True,
    )
    return LNNP(args, mean=torch.tensor(2.5), std=torch.tensor(1.5)).eval()


def assert_predictions_match(model, reference):
    z, pos, batch = create_example_batch()
    expected = reference(z, pos.detach().clone(), batch=batch)
    actual = model.eval()(z, pos.detach().clone(), batch=batch)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target)
    torch.testing.assert_close(model.mean, reference.mean)
    torch.testing.assert_close(model.std, reference.std)


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize("unrelated_key", [False, True])
def test_load_model_legacy_output_std(tmp_path, module, prefix, unrelated_key):
    state = {prefix + k: v for k, v in module.model.state_dict().items()}
    state[prefix + "output_model.output_std"] = torch.tensor(999.0)
    if unrelated_key:
        state[prefix + "output_model.output_std_extra"] = torch.ones(1)
    path = tmp_path / "legacy.ckpt"
    torch.save({"hyper_parameters": dict(module.hparams), "state_dict": state}, path)
    if unrelated_key:
        with pytest.raises(RuntimeError, match="Unexpected key.*output_std_extra"):
            load_model(str(path))
    else:
        loaded = load_model(str(path)).eval()
        assert_predictions_match(loaded, module.model)
        assert "output_model.output_std" not in loaded.state_dict()


@pytest.mark.parametrize("unrelated_key", [False, True])
def test_lightning_load_legacy_output_std(tmp_path, module, unrelated_key):
    state = module.state_dict()
    state["model.output_model.output_std"] = torch.tensor(999.0)
    if unrelated_key:
        state["model.output_model.output_std_extra"] = torch.ones(1)
    path = tmp_path / "lightning.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(module.hparams),
            "hparams_name": "hparams",
            "state_dict": state,
        },
        path,
    )
    if unrelated_key:
        with pytest.raises(RuntimeError, match="Unexpected key.*output_std_extra"):
            LNNP.load_from_checkpoint(path)
    else:
        loaded = LNNP.load_from_checkpoint(path).eval()
        assert_predictions_match(loaded.model, module.model)


@pytest.mark.parametrize("unrelated_key", [False, True])
def test_transfer_load_legacy_output_std(module, unrelated_key):
    state = module.state_dict()
    state["model.output_model.output_std"] = torch.tensor(999.0)
    if unrelated_key:
        state["model.output_model.output_std_extra"] = torch.ones(1)
    state = fix_state_dict({"state_dict": state}, reset_normalization=False)
    loaded = create_model(dict(module.hparams))
    if unrelated_key:
        with pytest.raises(RuntimeError, match="Unexpected key.*output_std_extra"):
            loaded.load_state_dict(state)
    else:
        loaded.load_state_dict(state)
        assert_predictions_match(loaded, module.model)


def test_ema_restore_cleans_averaged_and_current_weights(module):
    """EMA resume loads both state dictionaries through the model's compatibility hook."""
    ema = EMAWeightAveraging(decay=0.5, start_step=0, update_every=1)
    ema.setup(None, module, "fit")
    ema.on_train_batch_end(SimpleNamespace(global_step=1), module, None, None, 0)
    # Make current and averaged parameters distinct to check both restore paths.
    with torch.no_grad():
        next(module.parameters()).add_(0.25)
    checkpoint = {"state_dict": deepcopy(module.state_dict())}
    ema.on_save_checkpoint(None, module, checkpoint)
    checkpoint["state_dict"]["model.output_model.output_std"] = torch.tensor(999.0)
    checkpoint["current_model_state"]["model.output_model.output_std"] = torch.tensor(123.0)

    restored = LNNP(dict(module.hparams))
    restored.load_state_dict(checkpoint["state_dict"])
    resumed_ema = EMAWeightAveraging(decay=0.5, start_step=0, update_every=1)
    resumed_ema.setup(None, restored, "fit")
    resumed_ema.load_state_dict(ema.state_dict())
    resumed_ema.on_load_checkpoint(None, restored, checkpoint)
    for name, value in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    for name, value in ema._average_model.state_dict().items():
        torch.testing.assert_close(resumed_ema._average_model.state_dict()[name], value)
