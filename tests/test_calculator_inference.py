"""Regression tests for the ASE inference path used alongside OpenMM."""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from torchmdnet.calculators import TMDNETCalculator
from torchmdnet.models.model import create_model, load_model
from torchmdnet.scripts.train import get_argparse
from utils import load_example_args


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, "float64"])
def test_calculator_dtype_and_neutral_default(dtype):
    directory = Path(__file__).parent
    atoms = read(directory / "caffeine.pdb")
    calc = TMDNETCalculator(str(directory / "example_tensornet.ckpt"), dtype=dtype)
    atoms.calc = calc
    with torch.no_grad():
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    expected_dtype = torch.float32 if dtype == torch.float32 else torch.float64
    assert forces.dtype == (
        np.float32 if expected_dtype == torch.float32 else np.float64
    )
    assert not calc.model.training
    pos = torch.tensor(atoms.positions, dtype=expected_dtype, requires_grad=True)
    z = torch.tensor(atoms.numbers)
    reference, _ = calc.model(z, pos, q=torch.zeros(1, dtype=expected_dtype))
    reference_forces = -torch.autograd.grad(reference.sum(), pos)[0]
    np.testing.assert_allclose(
        forces, reference_forces.detach().numpy(), rtol=1e-5, atol=1e-5
    )
    assert energy == pytest.approx(reference.item())
    atoms.info["charge"] = 0.5
    assert calc.check_state(atoms) == ["info"]
    atoms.get_potential_energy()
    assert calc.evals == 2
    # Unrelated array metadata must not trigger ambiguous dictionary equality.
    atoms.info["metadata"] = np.arange(3)
    atoms.get_potential_energy()
    assert calc.evals == 2
    calc.calculate()
    assert calc.evals == 3


def test_load_model_disables_enabled_atomref(tmp_path):
    args = load_example_args("tensornet", remove_prior=True)
    args.pop("check_errors", None)
    args.update(prior_model=[{"Atomref": {"max_z": 100, "enable": True}}])
    model = create_model(args)
    path = tmp_path / "atomref.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": model.state_dict()}, path)
    loaded = load_model(str(path), remove_ref_energy=True)
    assert not loaded.prior_model[0].enable
    loaded = load_model(str(path))
    assert loaded.prior_model[0].enable


def test_reaction_field_dielectric_reaches_output_model():
    args = load_example_args("tensornet2", remove_prior=True)
    args.update(
        output_model="ScalarPlusWeightedCoulomb",
        coulomb_cutoff=8.0,
        coulomb_epsilon_solvent=40.0,
        q_weights=[1.0] * (args["num_layers"] + 1),
    )
    model = create_model(args)
    assert model.output_model.epsilon_solvent == 40.0


@pytest.mark.parametrize("bias", [False, True])
def test_scalar_output_checkpoint_bias(tmp_path, bias):
    option = (
        "--output-mlp-bias-on-last-layer"
        if bias
        else "--no-output-mlp-bias-on-last-layer"
    )
    args = vars(get_argparse().parse_args(["--model", "tensornet", option]))
    args.update(
        embedding_dimension=8, num_layers=1, num_rbf=4, output_mlp_num_layers=1
    )
    model = create_model(args)
    path = tmp_path / "output.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": model.state_dict()}, path)
    loaded = load_model(str(path))
    layers = loaded.output_model.output_network.layers
    assert len(layers) == 5
    assert (layers[-1].bias is not None) == bias
